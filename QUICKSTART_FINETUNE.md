# Tóm tắt nhanh: Fine-tune BiomedCLIP

## Các file quan trọng để fine-tune

### 1. **Entry point chính**
```
biomedclip_finetuning/open_clip/src/open_clip_train/main.py
```
→ File này orchestrate toàn bộ quá trình training

### 2. **Training loop**
```
biomedclip_finetuning/open_clip/src/open_clip_train/train.py
```
→ Logic huấn luyện: forward pass, backward pass, metrics

### 3. **Data loading**
```
biomedclip_finetuning/open_clip/src/open_clip_train/data.py
```
→ Load CSV, WebDataset, augmentation

### 4. **Loss functions**
```
biomedclip_finetuning/open_clip/src/open_clip/loss.py
```
→ CLIP loss, DHN-NCE loss (cho medical imaging)

### 5. **Scripts mẫu**
- Bash: `biomedclip_finetuning/open_clip/scripts/biomedclip.sh`
- PowerShell: `finetune_biomedclip.ps1` (mới tạo)

---

## Cách chạy (3 bước)

### Bước 1: Chuẩn bị data
Tạo file CSV với 2 cột:
```csv
filename,Caption
path/to/image1.jpg,"Medical description 1"
path/to/image2.jpg,"Medical description 2"
```

### Bước 2: Chỉnh sửa config
Mở file `finetune_biomedclip.ps1`, sửa dòng:
```powershell
$TRAIN_DATA = "data/your_dataset/train.csv"  # <-- Đổi path này
```

### Bước 3: Chạy
```powershell
.\finetune_biomedclip.ps1
```

---

## Tham số cần điều chỉnh

| Tham số | File | Dòng | Mô tả |
|---------|------|------|-------|
| `$TRAIN_DATA` | finetune_biomedclip.ps1 | ~11 | Đường dẫn CSV |
| `$BATCH_SIZE` | finetune_biomedclip.ps1 | ~7 | 8-32 tùy GPU |
| `$NUM_EPOCHS` | finetune_biomedclip.ps1 | ~9 | 20-50 epochs |
| `$LEARNING_RATE` | finetune_biomedclip.ps1 | ~17 | 1e-3 đến 5e-4 |

---

## Xem kết quả

```powershell
# TensorBoard
cd biomedclip_finetuning\open_clip\src
tensorboard --logdir logs
# Mở: http://localhost:6006
```

Checkpoints lưu tại:
```
biomedclip_finetuning/open_clip/src/logs/<experiment_name>/checkpoints/
```

---

## Hướng dẫn chi tiết

Xem file **`FINETUNING_GUIDE.md`** để biết:
- Cấu trúc các file huấn luyện
- Tất cả hyperparameters
- Multi-GPU training
- Troubleshooting
- Inference code sau khi fine-tune
