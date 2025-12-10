# Training Strategy for Frequency-Aware MedCLIP-SAMv2

## 📊 Phân Tích Kiến Trúc & Các Thành Phần Cần Train

### 1. Tổng Quan Các Module

```
┌─────────────────────────────────────────────────────┐
│ PRETRAINED (FROZEN) - Không train                   │
├─────────────────────────────────────────────────────┤
│ ✓ BiomedCLIP Vision Encoder (pretrained)           │
│ ✓ BiomedCLIP Text Encoder (pretrained)             │
│ ✓ Wavelet Transform (không có parameters)          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ TRAINABLE MODULES - CẦN TRAIN                       │
├─────────────────────────────────────────────────────┤
│ 1. HighFreqProjection                               │
│    - Conv layers: 3 layers (3→64→128→768)          │
│    - Linear projection                              │
│    - Parameters: ~1.5M                              │
│                                                     │
│ 2. FeatureFusionGate                                │
│    - fusion_alpha: 1 learnable parameter           │
│    - freq_transform: Linear(768, 768)              │
│    - gate network: Linear layers                   │
│    - Parameters: ~1.2M                              │
│                                                     │
│ 3. (Optional) Fine-tune BiomedCLIP                  │
│    - Last few transformer layers                   │
│    - Parameters: ~20M (if unfrozen)                │
└─────────────────────────────────────────────────────┘
```

---

## 🎯 Chiến Lược Training 3 Giai Đoạn

### **GIAI ĐOẠN 1: Warm-up Training (2-5 epochs)**
**Mục tiêu**: Train frequency-aware modules với BiomedCLIP frozen

#### Freeze/Unfreeze:
```python
# Freeze BiomedCLIP
for param in biomedclip_model.parameters():
    param.requires_grad = False

# Unfreeze frequency-aware modules
high_freq_projection.requires_grad_(True)
fusion_gate.requires_grad_(True)
```

#### Loss Function:
```python
# Supervised segmentation loss
loss = dice_loss(pred_mask, gt_mask) + \
       bce_loss(pred_mask, gt_mask) + \
       boundary_loss(pred_mask, gt_mask)  # Thêm boundary loss

# Regularization cho fusion weight
loss += lambda_reg * torch.abs(fusion_gate.fusion_alpha - 0.1)
```

#### Hyperparameters:
- **Learning rate**: 1e-4 (Adam optimizer)
- **Batch size**: 8-16 (tùy GPU memory)
- **Epochs**: 3-5
- **Weight decay**: 1e-5
- **Gradient clipping**: 1.0

#### Expected Output:
- HighFreqProjection học cách extract boundary features
- FusionGate học fusion_alpha ≈ 0.1-0.3
- Saliency maps bắt đầu sắc nét hơn baseline

---

### **GIAI ĐOẠN 2: Fine-tuning BiomedCLIP (5-10 epochs)**
**Mục tiêu**: Fine-tune vision encoder với frequency injection

#### Freeze/Unfreeze:
```python
# Unfreeze last N transformer blocks of BiomedCLIP
N = 2  # Chỉ unfreeze 2 blocks cuối
for i, block in enumerate(biomedclip_model.vision_model.transformer.blocks):
    if i >= len(blocks) - N:
        for param in block.parameters():
            param.requires_grad = True
    else:
        for param in block.parameters():
            param.requires_grad = False

# Keep frequency modules trainable
high_freq_projection.requires_grad_(True)
fusion_gate.requires_grad_(True)
```

#### Loss Function:
```python
# Multi-task loss
loss = seg_loss(pred_mask, gt_mask) + \
       0.3 * contrastive_loss(image_feat, text_feat) + \
       0.2 * boundary_sharpness_loss(saliency_map)

# Boundary sharpness loss: khuyến khích saliency sắc nét
def boundary_sharpness_loss(saliency):
    # Laplacian for sharpness
    laplacian = compute_laplacian(saliency)
    return -torch.mean(torch.abs(laplacian))
```

#### Hyperparameters:
- **Learning rate**: 
  - BiomedCLIP: 1e-5 (slow)
  - Frequency modules: 5e-5 (faster)
- **Batch size**: 8-12
- **Epochs**: 5-10
- **Warmup**: 500 steps
- **LR scheduler**: Cosine annealing

#### Expected Output:
- Vision encoder adapt để sử dụng frequency information
- Saliency maps sharper, Dice score tăng 3-5%
- Fusion_alpha có thể điều chỉnh cao hơn (0.2-0.4)

---

### **GIAI ĐOẠN 3: End-to-End Fine-tuning (10-20 epochs)**
**Mục tiêu**: Optimize toàn bộ pipeline với real SAM feedback

#### Freeze/Unfreeze:
```python
# Unfreeze toàn bộ (except text encoder - giữ frozen)
biomedclip_model.vision_model.requires_grad_(True)
biomedclip_model.text_model.requires_grad_(False)  # Keep frozen
high_freq_projection.requires_grad_(True)
fusion_gate.requires_grad_(True)
```

#### Loss Function:
```python
# End-to-end loss với SAM feedback
loss = seg_loss(sam_mask, gt_mask) + \
       0.5 * saliency_loss(saliency_map, gt_saliency) + \
       0.3 * prompt_quality_loss(prompts, gt_mask) + \
       0.2 * consistency_loss(saliency_map, sam_mask)

# Prompt quality: bboxes phải tight với GT
def prompt_quality_loss(bboxes, gt_mask):
    bbox_masks = create_bbox_masks(bboxes)
    iou = compute_iou(bbox_masks, gt_mask)
    return 1.0 - iou.mean()
```

#### Hyperparameters:
- **Learning rate**: 
  - Vision encoder: 5e-6 (very slow)
  - Frequency modules: 2e-5
- **Batch size**: 4-8 (SAM requires more memory)
- **Epochs**: 10-20
- **Mixed precision**: FP16 để tiết kiệm memory
- **Gradient accumulation**: 2-4 steps

#### Expected Output:
- Best segmentation performance
- Dice score tăng 8-15% so với baseline
- Tight SAM prompts, minimal false positives

---

## 📝 Training Script Skeleton

```python
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from frequency_aware import (
    DualStreamPreprocessor,
    HighFreqProjection,
    FeatureFusionGate,
    FrequencyAwareSaliencyGenerator
)

class FreqAwareTrainer:
    def __init__(self, config):
        self.device = torch.device('cuda')
        
        # Load BiomedCLIP (pretrained)
        self.biomedclip = load_biomedclip_model().to(self.device)
        
        # Initialize frequency-aware modules
        self.preprocessor = DualStreamPreprocessor()
        self.high_freq_proj = HighFreqProjection(
            embedding_dim=768,
            num_patches=196
        ).to(self.device)
        
        self.fusion_gate = FeatureFusionGate(
            embedding_dim=768,
            fusion_ratio=0.1
        ).to(self.device)
        
        self.saliency_gen = FrequencyAwareSaliencyGenerator().to(self.device)
        
        # Stage-specific setup
        self.current_stage = 1
        
    def setup_stage_1(self):
        """Giai đoạn 1: Warm-up training"""
        # Freeze BiomedCLIP
        for param in self.biomedclip.parameters():
            param.requires_grad = False
        
        # Trainable parameters
        trainable_params = list(self.high_freq_proj.parameters()) + \
                          list(self.fusion_gate.parameters())
        
        self.optimizer = Adam(trainable_params, lr=1e-4, weight_decay=1e-5)
        self.scheduler = None
        self.epochs = 5
        
        print(f"Stage 1: {sum(p.numel() for p in trainable_params):,} trainable parameters")
    
    def setup_stage_2(self):
        """Giai đoạn 2: Fine-tune vision encoder"""
        # Unfreeze last 2 transformer blocks
        N = 2
        blocks = self.biomedclip.vision_model.transformer.blocks
        for i, block in enumerate(blocks):
            if i >= len(blocks) - N:
                for param in block.parameters():
                    param.requires_grad = True
        
        # Different LR for different modules
        param_groups = [
            {'params': self.high_freq_proj.parameters(), 'lr': 5e-5},
            {'params': self.fusion_gate.parameters(), 'lr': 5e-5},
            {'params': [p for p in self.biomedclip.vision_model.parameters() 
                       if p.requires_grad], 'lr': 1e-5}
        ]
        
        self.optimizer = AdamW(param_groups, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10)
        self.epochs = 10
        
        trainable = sum(p.numel() for p in self.biomedclip.parameters() if p.requires_grad)
        print(f"Stage 2: {trainable:,} trainable parameters")
    
    def setup_stage_3(self):
        """Giai đoạn 3: End-to-end fine-tuning"""
        # Unfreeze vision encoder, keep text encoder frozen
        for param in self.biomedclip.vision_model.parameters():
            param.requires_grad = True
        for param in self.biomedclip.text_model.parameters():
            param.requires_grad = False
        
        param_groups = [
            {'params': self.high_freq_proj.parameters(), 'lr': 2e-5},
            {'params': self.fusion_gate.parameters(), 'lr': 2e-5},
            {'params': self.biomedclip.vision_model.parameters(), 'lr': 5e-6}
        ]
        
        self.optimizer = AdamW(param_groups, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=20)
        self.epochs = 20
        
        trainable = sum(p.numel() for p in self.biomedclip.parameters() if p.requires_grad)
        print(f"Stage 3: {trainable:,} trainable parameters")
    
    def train_epoch(self, dataloader, stage):
        self.train()
        total_loss = 0
        
        for batch in dataloader:
            images, texts, masks_gt = batch
            
            # Dual-stream preprocessing
            prep_result = self.preprocessor(images)
            original_stream = prep_result['original_stream'].to(self.device)
            high_freq_stream = prep_result['high_freq_enhanced'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get BiomedCLIP features
            with torch.set_grad_enabled(stage >= 2):
                image_features = self.biomedclip.vision_model(original_stream)
                text_features = self.biomedclip.text_model(texts)
            
            # Project high-freq features
            high_freq_proj = self.high_freq_proj(high_freq_stream)
            
            # Feature fusion
            fused_features = self.fusion_gate(image_features, high_freq_proj)
            
            # Generate saliency
            saliency_result = self.saliency_gen(
                fused_features,
                high_freq_proj,
                text_features,
                original_stream
            )
            
            # Compute loss
            loss = self.compute_loss(saliency_result, masks_gt, stage)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.parameters() if p.requires_grad],
                max_norm=1.0
            )
            self.optimizer.step()
            
            total_loss += loss.item()
        
        if self.scheduler:
            self.scheduler.step()
        
        return total_loss / len(dataloader)
    
    def compute_loss(self, saliency_result, masks_gt, stage):
        """Multi-stage loss computation"""
        saliency_map = saliency_result['saliency_map_refined']
        binary_mask = saliency_result['binary_mask']
        
        # Base segmentation loss
        dice = dice_loss(binary_mask, masks_gt)
        bce = F.binary_cross_entropy(binary_mask, masks_gt)
        
        loss = dice + bce
        
        if stage >= 2:
            # Add boundary sharpness loss
            boundary_loss = self.boundary_sharpness_loss(saliency_map)
            loss += 0.2 * boundary_loss
        
        if stage >= 3:
            # Add consistency regularization
            consistency = self.consistency_loss(saliency_map, binary_mask)
            loss += 0.1 * consistency
        
        return loss
    
    def boundary_sharpness_loss(self, saliency_map):
        """Encourage sharp boundaries"""
        # Compute gradient magnitude
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=saliency_map.device)
        sobel_y = sobel_x.t()
        
        grad_x = F.conv2d(saliency_map.unsqueeze(1), 
                         sobel_x.view(1, 1, 3, 3), padding=1)
        grad_y = F.conv2d(saliency_map.unsqueeze(1), 
                         sobel_y.view(1, 1, 3, 3), padding=1)
        
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Maximize gradient at boundaries
        return -gradient_magnitude.mean()
    
    def consistency_loss(self, saliency_map, binary_mask):
        """Ensure consistency between saliency and binary mask"""
        return F.mse_loss(saliency_map, binary_mask.float())


# Main training loop
def main():
    config = load_config('config/freq_aware_config.yaml')
    trainer = FreqAwareTrainer(config)
    
    # Load datasets
    train_loader = create_dataloader('train', batch_size=8)
    val_loader = create_dataloader('val', batch_size=8)
    
    # Stage 1: Warm-up
    print("\n" + "="*60)
    print("STAGE 1: Warm-up Training (Frequency Modules Only)")
    print("="*60)
    trainer.setup_stage_1()
    for epoch in range(trainer.epochs):
        train_loss = trainer.train_epoch(train_loader, stage=1)
        val_loss = validate(trainer, val_loader)
        print(f"Epoch {epoch+1}/{trainer.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Save checkpoint
    save_checkpoint(trainer, 'checkpoints/stage1_final.pth')
    
    # Stage 2: Fine-tune vision encoder
    print("\n" + "="*60)
    print("STAGE 2: Fine-tuning Vision Encoder")
    print("="*60)
    trainer.setup_stage_2()
    best_dice = 0
    for epoch in range(trainer.epochs):
        train_loss = trainer.train_epoch(train_loader, stage=2)
        val_metrics = validate_with_metrics(trainer, val_loader)
        print(f"Epoch {epoch+1}/{trainer.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Dice: {val_metrics['dice']:.4f}")
        
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            save_checkpoint(trainer, 'checkpoints/stage2_best.pth')
    
    # Stage 3: End-to-end
    print("\n" + "="*60)
    print("STAGE 3: End-to-End Fine-tuning")
    print("="*60)
    trainer.setup_stage_3()
    load_checkpoint(trainer, 'checkpoints/stage2_best.pth')
    
    for epoch in range(trainer.epochs):
        train_loss = trainer.train_epoch(train_loader, stage=3)
        val_metrics = validate_with_metrics(trainer, val_loader)
        print(f"Epoch {epoch+1}/{trainer.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Dice: {val_metrics['dice']:.4f} | "
              f"Val IoU: {val_metrics['iou']:.4f}")
        
        save_checkpoint(trainer, f'checkpoints/stage3_epoch{epoch+1}.pth')

if __name__ == '__main__':
    main()
```

---

## 💾 Yêu Cầu GPU Memory

| Stage | Batch Size | Memory Usage | Recommended GPU |
|-------|-----------|--------------|-----------------|
| Stage 1 | 16 | ~8GB | RTX 3080 (10GB) |
| Stage 2 | 12 | ~12GB | RTX 3090 (24GB) |
| Stage 3 | 8 | ~16GB | A100 (40GB) hoặc V100 (32GB) |

### Memory Optimization:
```python
# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = trainer.compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 📊 Monitoring & Checkpoints

### Metrics to Track:
```python
metrics = {
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_dice': dice_coefficient,
    'val_iou': iou_score,
    'fusion_alpha': fusion_gate.fusion_alpha.item(),
    'saliency_sharpness': compute_sharpness(saliency_maps),
    'boundary_accuracy': boundary_f1_score
}
```

### Checkpointing Strategy:
```
checkpoints/
├── stage1_final.pth           # End of stage 1
├── stage2_best.pth            # Best validation Dice in stage 2
├── stage2_epoch{N}.pth        # Regular checkpoints
├── stage3_best.pth            # Best overall
└── stage3_epoch{N}.pth        # Final checkpoints
```

---

## 🎯 Kết Luận

### Thứ Tự Training Khuyến Nghị:
1. **Stage 1 (3-5 epochs)**: Warm-up frequency modules → fusion_alpha ≈ 0.1-0.3
2. **Stage 2 (5-10 epochs)**: Fine-tune last transformer blocks → Dice +3-5%
3. **Stage 3 (10-20 epochs)**: End-to-end optimization → Dice +8-15%

### Expected Timeline (với 1 GPU):
- Stage 1: 2-4 hours (small dataset), 8-12 hours (large dataset)
- Stage 2: 5-10 hours (medium), 20-30 hours (large)
- Stage 3: 10-20 hours (medium), 40-60 hours (large)
- **Total**: 17-34 hours (medium), 68-102 hours (large dataset)

### Key Success Indicators:
✅ fusion_alpha converges to 0.2-0.4 (not too low, not too high)
✅ Saliency maps visibly sharper than baseline
✅ Dice score improvement ≥5% after stage 2
✅ Boundary F1 score improvement ≥10%
✅ SAM prompts tighter (smaller bboxes, higher IoU with GT)
