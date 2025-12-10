"""
Frequency-Aware MedCLIP-SAMv2 Integration Script

This script integrates the frequency-aware module with MedCLIP-SAMv2's
saliency map generation pipeline to create enhanced segmentation masks.

Usage:
    python frequency_aware_integration.py --config config.yaml --data_dir data/ --output_dir output/
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image
import yaml
import logging

# Import frequency-aware modules
try:
    from frequency_aware import (
        DualStreamPreprocessor,
        FrequencyAwareSaliencyGenerator,
        MultiScaleSaliencyGenerator,
        FrequencyAwarePipeline,
        make_frequency_aware
    )
except ImportError:
    print("Error: frequency_aware module not found. Make sure it's in the correct location.")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrequencyAwareMedCLIPSAM:
    """
    Complete frequency-aware MedCLIP-SAMv2 integration pipeline.
    
    Combines:
    1. Wavelet-based preprocessing for boundary detection
    2. Feature fusion in vision encoder
    3. Refined saliency map generation
    4. Post-processing and SAM prompt generation
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        """
        Initialize the frequency-aware pipeline.
        
        Args:
            config: Configuration dictionary
            device: Device to run on ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device
        self.dtype = torch.float32
        
        # Initialize components
        self.preprocessor = DualStreamPreprocessor(
            wavelet_type=config.get('wavelet_type', 'haar'),
            image_size=config.get('image_size', 224)
        )
        
        self.saliency_generator = self._init_saliency_generator(config)
        self.postprocessor = FrequencyAwarePipeline(
            prompt_type=config.get('prompt_type', 'bbox'),
            refine_masks=config.get('refine_masks', True)
        )
        
        logger.info("Frequency-aware pipeline initialized successfully")
    
    def _init_saliency_generator(self, config: Dict):
        """Initialize the appropriate saliency generator."""
        use_multi_scale = config.get('use_multi_scale', False)
        
        if use_multi_scale:
            scales = config.get('scales', [0.5, 1.0, 1.5])
            return MultiScaleSaliencyGenerator(
                scales=scales,
                aggregation=config.get('aggregation', 'weighted_mean')
            )
        else:
            return FrequencyAwareSaliencyGenerator(
                blur_kernel=config.get('blur_kernel', 5),
                morphology_kernel=config.get('morphology_kernel', 5),
                frequency_weight=config.get('frequency_weight', 0.3)
            )
    
    def preprocess(self, image_path: str) -> Dict:
        """
        Preprocess image using dual-stream approach.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Dict with preprocessed data
        """
        logger.info(f"Preprocessing image: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Dual-stream preprocessing
        result = self.preprocessor(image_np)
        
        # Convert to tensors and move to device
        original_stream = result['original_stream'].unsqueeze(0).to(self.device)
        high_freq_stream = result['high_freq_enhanced'].unsqueeze(0).to(self.device)
        
        return {
            'original_stream': original_stream,
            'high_freq_stream': high_freq_stream,
            'raw_image': image_np,
            'image_tensor': torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device),
            'wavelet_components': result['wavelet_components']
        }
    
    def generate_saliency_map(
        self,
        image_features: torch.Tensor,
        high_freq_features: torch.Tensor,
        text_embedding: torch.Tensor,
        image_tensor: torch.Tensor,
        image_size: Tuple[int, int]
    ) -> Dict:
        """
        Generate refined saliency map.
        
        Args:
            image_features: Vision encoder features
            high_freq_features: High-frequency wavelet features
            text_embedding: Text embedding from language model
            image_tensor: Original image tensor
            image_size: Target output size
        
        Returns:
            Dict with saliency maps and confidence scores
        """
        logger.info("Generating refined saliency map with frequency awareness")
        
        result = self.saliency_generator(
            image_features=image_features,
            high_freq_features=high_freq_features,
            text_embedding=text_embedding,
            image_tensor=image_tensor,
            target_size=image_size
        )
        
        return result
    
    def extract_sam_prompts(
        self,
        saliency_map: np.ndarray,
        binary_mask: np.ndarray,
        confidence_map: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Extract SAM prompts from saliency map.
        
        Args:
            saliency_map: Generated saliency map
            binary_mask: Binary segmentation mask
            confidence_map: Confidence scores
        
        Returns:
            Dict with SAM prompts and post-processing info
        """
        logger.info("Extracting SAM prompts from saliency map")
        
        result = self.postprocessor(
            saliency_map=saliency_map,
            binary_mask=binary_mask,
            confidence_map=confidence_map
        )
        
        return result
    
    def process_single_image(
        self,
        image_path: str,
        image_features: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Process a single image end-to-end (preprocessing only, feature generation needs external model).
        
        Args:
            image_path: Path to input image
            image_features: Optional pre-computed image features
            text_embedding: Optional pre-computed text embedding
        
        Returns:
            Dict with all processing results
        """
        # Preprocessing
        preprocessed = self.preprocess(image_path)
        
        logger.info(f"Preprocessing complete. Wavelet components extracted.")
        logger.info(f"  - Original stream shape: {preprocessed['original_stream'].shape}")
        logger.info(f"  - High-freq stream shape: {preprocessed['high_freq_stream'].shape}")
        
        return {
            'preprocessed': preprocessed,
            'image_features': image_features,
            'text_embedding': text_embedding
        }
    
    def process_batch(
        self,
        image_dir: str,
        output_dir: str,
        text_embeddings: Optional[Dict[str, np.ndarray]] = None,
        save_intermediates: bool = True
    ):
        """
        Process a batch of images.
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save outputs
            text_embeddings: Dict mapping image names to text embeddings
            save_intermediates: Whether to save intermediate results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        logger.info(f"Found {len(image_files)} images to process")
        
        for idx, image_file in enumerate(image_files):
            logger.info(f"Processing {idx+1}/{len(image_files)}: {image_file}")
            
            image_path = os.path.join(image_dir, image_file)
            
            try:
                # Preprocess
                result = self.process_single_image(image_path)
                preprocessed = result['preprocessed']
                
                # Save preprocessing results if requested
                if save_intermediates:
                    save_path = os.path.join(output_dir, f"{Path(image_file).stem}_preprocessed.npz")
                    np.savez(
                        save_path,
                        raw_image=preprocessed['raw_image'],
                        ll=preprocessed['wavelet_components']['ll'],
                        lh=preprocessed['wavelet_components']['lh'],
                        hl=preprocessed['wavelet_components']['hl'],
                        hh=preprocessed['wavelet_components']['hh'],
                        high_freq_merged=preprocessed['wavelet_components']['high_freq_merged']
                    )
                    logger.info(f"  Saved preprocessing results to {save_path}")
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {str(e)}")
                continue
        
        logger.info(f"Batch processing complete. Results saved to {output_dir}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Frequency-Aware MedCLIP-SAMv2 Integration')
    parser.add_argument('--config', type=str, default='config/freq_aware_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--image_dir', type=str, default='data/images/',
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='output/freq_aware/',
                       help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_process', action='store_true',
                       help='Process entire directory of images')
    parser.add_argument('--save_intermediates', action='store_true',
                       help='Save intermediate preprocessing results')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        logger.warning(f"Config file not found: {args.config}. Using defaults.")
        config = {
            'wavelet_type': 'haar',
            'image_size': 224,
            'frequency_weight': 0.3,
            'prompt_type': 'bbox',
            'refine_masks': True
        }
    
    # Initialize pipeline
    pipeline = FrequencyAwareMedCLIPSAM(config, device=args.device)
    
    # Process
    if args.batch_process and os.path.isdir(args.image_dir):
        pipeline.process_batch(
            args.image_dir,
            args.output_dir,
            save_intermediates=args.save_intermediates
        )
    else:
        logger.info("Single image mode: Use FrequencyAwareMedCLIPSAM class for programmatic use")


if __name__ == '__main__':
    main()
