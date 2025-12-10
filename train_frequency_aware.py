"""
Frequency-Aware MedCLIP-SAMv2 Training Script

3-stage training strategy:
- Stage 1: Warm-up frequency modules (BiomedCLIP frozen)
- Stage 2: Fine-tune vision encoder with frequency injection
- Stage 3: End-to-end optimization with SAM feedback
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from PIL import Image
from torchvision import transforms
import csv

# Import frequency-aware modules
from frequency_aware import (
    DualStreamPreprocessor,
    HighFreqProjection,
    FeatureFusionGate,
    FrequencyAwareSaliencyGenerator,
    FrequencyAwarePipeline
)

# Import BiomedCLIP (adjust path as needed)
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (
            pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth
        )
        
        return 1 - dice.mean()


class BoundaryLoss(nn.Module):
    """Boundary-aware loss to encourage sharp edges."""
    def __init__(self):
        super().__init__()
        # Sobel kernels for edge detection
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        """
        Encourage sharp boundaries by maximizing gradient at GT boundaries.
        """
        # Compute gradients
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
        
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
        
        # Loss: minimize difference in gradient magnitude
        return F.mse_loss(pred_grad, target_grad)


class MedPixDataset(Dataset):
    """MedPix caption-only dataset for BiomedCLIP fine-tuning."""

    def __init__(self, csv_path, tokenizer, image_size=224, max_text_len=64):
        self.csv_path = Path(csv_path)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_text_len = max_text_len

        self.samples = []
        with self.csv_path.open('r', newline='') as f:
            reader = csv.DictReader(f)
            if 'Caption' not in reader.fieldnames or 'filename' not in reader.fieldnames:
                raise ValueError("CSV must have columns: Caption, filename")
            for row in reader:
                caption = row['Caption']
                img_path = row['filename']
                self.samples.append((caption, img_path))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        caption, img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        text_tokens = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt'
        )
        # Squeeze batch dimension from tokenizer outputs
        text_tokens = {k: v.squeeze(0) for k, v in text_tokens.items()}

        # No segmentation mask available for MedPix
        mask = None
        return img, text_tokens, mask


def create_medpix_dataloaders(config, tokenizer):
    """Create train/val dataloaders for MedPix caption-only dataset."""
    csv_path = config.get('csv_path', 'data/medpix_dataset/medpix_dataset.csv')
    val_ratio = config.get('val_ratio', 0.1)
    batch_size = config.get('batch_size', 8)
    num_workers = config.get('num_workers', 4)

    dataset = MedPixDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        image_size=config.get('image_size', 224),
        max_text_len=config.get('max_text_len', 64)
    )

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


class FreqAwareModel(nn.Module):
    """
    Complete frequency-aware model combining BiomedCLIP and frequency modules.
    """
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.config = config
        self.device = device
        
        # Load BiomedCLIP
        logger.info("Loading BiomedCLIP model...")
        # Priority: local checkpoint at checkpoints/pytorch_model.bin if present
        local_ckpt = Path(config.get('biomedclip_ckpt', 'checkpoints/pytorch_model.bin'))
        if local_ckpt.exists():
            logger.info(f"Found local BiomedCLIP checkpoint: {local_ckpt}")
            self.biomedclip = AutoModel.from_pretrained(
                local_ckpt.parent,
                trust_remote_code=True
            ).to(device)
        elif config.get('finetuned_biomedclip', False):
            self.biomedclip = AutoModel.from_pretrained(
                "./saliency_maps/model",
                trust_remote_code=True
            ).to(device)
        else:
            self.biomedclip = AutoModel.from_pretrained(
                "chuhac/BiomedCLIP-vit-bert-hf",
                trust_remote_code=True
            ).to(device)
        
        # Initialize frequency-aware modules
        self.preprocessor = DualStreamPreprocessor(
            wavelet_type=config.get('wavelet_type', 'haar'),
            image_size=config.get('image_size', 224)
        )
        
        self.high_freq_proj = HighFreqProjection(
            embedding_dim=config.get('embedding_dim', 768),
            num_patches=config.get('num_patches', 196)
        ).to(device)
        
        self.fusion_gate = FeatureFusionGate(
            embedding_dim=config.get('embedding_dim', 768),
            fusion_ratio=config.get('fusion_ratio', 0.1)
        ).to(device)
        
        self.saliency_gen = FrequencyAwareSaliencyGenerator(
            blur_kernel=config.get('blur_kernel', 5),
            morphology_kernel=config.get('morphology_kernel', 5),
            frequency_weight=config.get('frequency_weight', 0.3)
        ).to(device)
        
        logger.info("Model initialized successfully")
    
    def forward(self, images, texts, return_intermediates=False):
        """
        Forward pass through the complete pipeline.
        
        Args:
            images: Batch of images
            texts: Batch of text prompts (already tokenized)
            return_intermediates: Whether to return intermediate features
        
        Returns:
            Saliency maps and optionally intermediate features
        """
        batch_size = images.shape[0]
        
        # Dual-stream preprocessing (this happens on CPU, then move to GPU)
        original_streams = []
        high_freq_streams = []
        
        for img in images:
            prep_result = self.preprocessor(img.cpu().numpy().transpose(1, 2, 0))
            original_streams.append(prep_result['original_stream'])
            high_freq_streams.append(prep_result['high_freq_enhanced'])
        
        original_stream = torch.stack(original_streams).to(self.device)
        high_freq_stream = torch.stack(high_freq_streams).to(self.device)
        
        # Get BiomedCLIP features
        with torch.set_grad_enabled(self.training):
            # Vision features
            vision_output = self.biomedclip.vision_model(
                original_stream,
                output_hidden_states=True
            )
            image_features = vision_output['hidden_states'][-1]  # Last layer
            
            # Text features
            text_output = self.biomedclip.text_model(
                input_ids=texts['input_ids'].to(self.device),
                attention_mask=texts['attention_mask'].to(self.device),
                output_hidden_states=True
            )
            text_features = text_output['pooler_output']
        
        # Project high-frequency features
        high_freq_proj = self.high_freq_proj(high_freq_stream)
        
        # Feature fusion
        fused_features = self.fusion_gate(image_features, high_freq_proj)
        
        # Generate saliency maps
        saliency_result = self.saliency_gen(
            image_features=fused_features,
            high_freq_features=high_freq_proj,
            text_embedding=text_features,
            image_tensor=original_stream,
            target_size=(images.shape[2], images.shape[3])
        )
        
        if return_intermediates:
            return saliency_result, {
                'image_features': image_features,
                'text_features': text_features,
                'high_freq_proj': high_freq_proj,
                'fused_features': fused_features,
                'fusion_alpha': self.fusion_gate.fusion_alpha.item()
            }
        
        return saliency_result


class Trainer:
    """
    Main trainer class for 3-stage training.
    """
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.current_stage = 1
        self.task = config.get('task', 'clip')  # 'clip' (caption-only) or 'seg' (segmentation)
        
        # Initialize model
        self.model = FreqAwareModel(config, device)
        
        # Loss functions
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.boundary_loss = BoundaryLoss().to(device)
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Metrics tracking
        self.best_dice = 0.0
        self.train_losses = []
        self.val_metrics = []
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    def setup_stage(self, stage):
        """Setup optimizer and scheduler for each stage."""
        self.current_stage = stage

        # Caption-only CLIP fine-tuning on MedPix
        if self.task == 'clip':
            logger.info("=== CLIP fine-tuning on MedPix (caption-only) ===")
            for param in self.model.biomedclip.parameters():
                param.requires_grad = True

            self.optimizer = AdamW(
                self.model.biomedclip.parameters(),
                lr=self.config.get('clip_lr', 1e-5),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            self.scheduler = None
            self.epochs = self.config.get('clip_epochs', 5)

            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
            logger.info(f"Training for {self.epochs} epochs")
            return
        
        if stage == 1:
            # Stage 1: Freeze BiomedCLIP, train frequency modules only
            logger.info("=== STAGE 1: Warm-up Training ===")
            logger.info("Freezing BiomedCLIP, training frequency modules")
            
            for param in self.model.biomedclip.parameters():
                param.requires_grad = False
            
            trainable_params = (
                list(self.model.high_freq_proj.parameters()) +
                list(self.model.fusion_gate.parameters())
            )
            
            self.optimizer = Adam(
                trainable_params,
                lr=self.config.get('stage1_lr', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
            self.scheduler = None
            self.epochs = self.config.get('stage1_epochs', 5)
            
        elif stage == 2:
            # Stage 2: Fine-tune last transformer blocks
            logger.info("=== STAGE 2: Fine-tuning Vision Encoder ===")
            logger.info("Unfreezing last transformer blocks")
            
            # Unfreeze last N blocks
            N = self.config.get('unfreeze_blocks', 2)
            blocks = self.model.biomedclip.vision_model.trunk.blocks
            
            for i, block in enumerate(blocks):
                if i >= len(blocks) - N:
                    for param in block.parameters():
                        param.requires_grad = True
                        logger.info(f"  Unfrozen block {i}")
            
            # Different learning rates for different modules
            param_groups = [
                {
                    'params': self.model.high_freq_proj.parameters(),
                    'lr': self.config.get('stage2_freq_lr', 5e-5)
                },
                {
                    'params': self.model.fusion_gate.parameters(),
                    'lr': self.config.get('stage2_freq_lr', 5e-5)
                },
                {
                    'params': [p for p in self.model.biomedclip.vision_model.parameters()
                              if p.requires_grad],
                    'lr': self.config.get('stage2_vision_lr', 1e-5)
                }
            ]
            
            self.optimizer = AdamW(
                param_groups,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('stage2_epochs', 10)
            )
            self.epochs = self.config.get('stage2_epochs', 10)
            
        elif stage == 3:
            # Stage 3: End-to-end fine-tuning
            logger.info("=== STAGE 3: End-to-End Fine-tuning ===")
            logger.info("Unfreezing vision encoder (keeping text encoder frozen)")
            
            for param in self.model.biomedclip.vision_model.parameters():
                param.requires_grad = True
            for param in self.model.biomedclip.text_model.parameters():
                param.requires_grad = False
            
            param_groups = [
                {
                    'params': self.model.high_freq_proj.parameters(),
                    'lr': self.config.get('stage3_freq_lr', 2e-5)
                },
                {
                    'params': self.model.fusion_gate.parameters(),
                    'lr': self.config.get('stage3_freq_lr', 2e-5)
                },
                {
                    'params': self.model.biomedclip.vision_model.parameters(),
                    'lr': self.config.get('stage3_vision_lr', 5e-6)
                }
            ]
            
            self.optimizer = AdamW(
                param_groups,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('stage3_epochs', 20)
            )
            self.epochs = self.config.get('stage3_epochs', 20)
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        logger.info(f"Training for {self.epochs} epochs")
    
    def compute_loss(self, saliency_result, masks_gt):
        """Compute multi-component loss."""
        saliency_map = saliency_result['saliency_map_refined']
        binary_mask = saliency_result['binary_mask']
        
        # Resize GT masks if needed
        if masks_gt.shape[-2:] != binary_mask.shape[-2:]:
            masks_gt = F.interpolate(
                masks_gt.float(),
                size=binary_mask.shape[-2:],
                mode='nearest'
            )
        
        # Base losses
        dice = self.dice_loss(binary_mask, masks_gt)
        bce = self.bce_loss(binary_mask, masks_gt)
        
        loss = dice + bce
        loss_dict = {'dice': dice.item(), 'bce': bce.item()}
        
        # Stage-specific losses
        if self.current_stage >= 2:
            # Boundary loss
            boundary = self.boundary_loss(saliency_map.unsqueeze(1), masks_gt)
            loss += 0.2 * boundary
            loss_dict['boundary'] = boundary.item()
        
        if self.current_stage >= 3:
            # Consistency loss
            consistency = F.mse_loss(saliency_map, binary_mask)
            loss += 0.1 * consistency
            loss_dict['consistency'] = consistency.item()
            
            # Fusion regularization
            fusion_alpha = self.model.fusion_gate.fusion_alpha
            reg = torch.abs(fusion_alpha - 0.2)  # Encourage alpha around 0.2
            loss += 0.01 * reg
            loss_dict['fusion_reg'] = reg.item()
        
        loss_dict['total'] = loss.item()
        return loss, loss_dict

    def compute_clip_loss(self, images, texts):
        """Contrastive CLIP loss for caption-only MedPix training."""
        outputs = self.model.biomedclip(
            pixel_values=images,
            input_ids=texts['input_ids'].to(self.device),
            attention_mask=texts['attention_mask'].to(self.device),
            return_loss=False
        )

        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        labels = torch.arange(logits_per_image.size(0), device=self.device)
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        loss = 0.5 * (loss_img + loss_txt)

        return loss, {
            'clip_loss': loss.item(),
            'loss_img': loss_img.item(),
            'loss_txt': loss_txt.item()
        }
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        loss_components = {}
        
        pbar = tqdm(dataloader, desc=f"Stage {self.current_stage} Training")
        
        for batch_idx, batch in enumerate(pbar):
            images, texts, masks_gt = batch
            images = images.to(self.device)
            masks_gt = masks_gt.to(self.device) if masks_gt is not None else None
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.task == 'clip':
                loss, loss_dict = self.compute_clip_loss(images, texts)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                self.optimizer.step()
            else:
                if masks_gt is None:
                    raise RuntimeError("Segmentation task requires masks_gt but dataset provides None.")

                if self.use_amp:
                    with autocast():
                        saliency_result = self.model(images, texts)
                        loss, loss_dict = self.compute_loss(saliency_result, masks_gt)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    saliency_result = self.model(images, texts)
                    loss, loss_dict = self.compute_loss(saliency_result, masks_gt)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=1.0
                    )
                    self.optimizer.step()
            
            total_loss += loss.item()
            
            # Accumulate loss components
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'alpha': f"{self.model.fusion_gate.fusion_alpha.item():.3f}"
            })
        
        # Average losses
        avg_loss = total_loss / len(dataloader)
        for key in loss_components:
            loss_components[key] /= len(dataloader)
        
        return avg_loss, loss_components
    
    @torch.no_grad()
    def validate(self, dataloader):
        """Validation pass."""
        self.model.eval()
        total_loss = 0

        if self.task == 'clip':
            for batch in tqdm(dataloader, desc="Validation"):
                images, texts, _ = batch
                images = images.to(self.device)
                loss, loss_dict = self.compute_clip_loss(images, texts)
                total_loss += loss.item()
            metrics = {
                'val_loss': total_loss / len(dataloader),
                'clip_loss': total_loss / len(dataloader)
            }
            return metrics

        dice_scores = []
        iou_scores = []

        for batch in tqdm(dataloader, desc="Validation"):
            images, texts, masks_gt = batch
            images = images.to(self.device)
            masks_gt = masks_gt.to(self.device)
            
            # Forward pass
            saliency_result = self.model(images, texts)
            loss, _ = self.compute_loss(saliency_result, masks_gt)
            
            total_loss += loss.item()
            
            # Compute metrics
            binary_mask = saliency_result['binary_mask']
            
            if masks_gt.shape[-2:] != binary_mask.shape[-2:]:
                masks_gt = F.interpolate(
                    masks_gt.float(),
                    size=binary_mask.shape[-2:],
                    mode='nearest'
                )
            
            dice = self.compute_dice(binary_mask, masks_gt)
            iou = self.compute_iou(binary_mask, masks_gt)
            
            dice_scores.extend(dice.cpu().numpy())
            iou_scores.extend(iou.cpu().numpy())
        
        metrics = {
            'val_loss': total_loss / len(dataloader),
            'dice': np.mean(dice_scores),
            'iou': np.mean(iou_scores),
            'dice_std': np.std(dice_scores),
            'iou_std': np.std(iou_scores)
        }

        return metrics
    
    def compute_dice(self, pred, target, smooth=1.0):
        """Compute Dice coefficient."""
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (
            pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth
        )
        
        return dice
    
    def compute_iou(self, pred, target, smooth=1.0):
        """Compute IoU."""
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        
        return iou
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'stage': self.current_stage,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dice': self.best_dice,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_dice = checkpoint.get('best_dice', 0.0)
        logger.info(f"Checkpoint loaded: {path}")
    
    def train(self, train_loader, val_loader):
        """Main training loop for current stage."""
        logger.info(f"\nStarting Stage {self.current_stage} training...")
        
        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Train
            train_loss, loss_components = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Loss components: {loss_components}")
            if self.task == 'clip':
                logger.info(f"Val CLIP Loss: {val_metrics['clip_loss']:.4f}")
            else:
                logger.info(f"Val Dice: {val_metrics['dice']:.4f} ± {val_metrics['dice_std']:.4f}")
                logger.info(f"Val IoU: {val_metrics['iou']:.4f} ± {val_metrics['iou_std']:.4f}")
                logger.info(f"Fusion Alpha: {self.model.fusion_gate.fusion_alpha.item():.4f}")
            
            # LR scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Save checkpoints
            self.save_checkpoint(f'stage{self.current_stage}_epoch{epoch+1}.pth')
            
            # Save best model
            if val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                self.save_checkpoint(f'stage{self.current_stage}_best.pth')
                logger.info(f"✓ New best Dice: {self.best_dice:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(f'stage{self.current_stage}_final.pth')
        logger.info(f"Stage {self.current_stage} training complete!")


def main():
    parser = argparse.ArgumentParser(description='Frequency-Aware MedCLIP-SAMv2 Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], default=1,
                       help='Training stage (1, 2, or 3)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--task', type=str, choices=['clip', 'seg'], default='clip',
                        help='Training task: clip (caption-only) or seg (segmentation)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override task from CLI
    config['task'] = args.task
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Tokenizer for captions
    tokenizer = AutoTokenizer.from_pretrained(
        config.get('tokenizer_name', 'chuhac/BiomedCLIP-vit-bert-hf'),
        trust_remote_code=True
    )

    # Build dataloaders
    if config['task'] == 'clip':
        train_loader, val_loader = create_medpix_dataloaders(config, tokenizer)
    else:
        raise RuntimeError("Segmentation task requires a dataset with masks. Please provide a segmentation dataset.")

    # Initialize trainer
    trainer = Trainer(config, device)
    
    # Setup stage
    trainer.setup_stage(args.stage)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
