# Frequency-Aware Integration Guide

## Tổng Quan

Frequency-Aware Integration là cải thiện MedCLIP-SAMv2 bằng cách tích hợp thông tin tần số cao từ FMISeg để tạo saliency maps sắc nét hơn, giúp SAM phân đoạn chính xác hơn.

## Kiến Trúc

### 4 Giai Đoạn Chính

```
Input Image
    ↓
[Giai Đoạn 1] Dual-Stream Preprocessing
    ├─ Stream 1: Tiêu chuẩn (RGB cho BiomedCLIP)
    └─ Stream 2: Wavelet Transform (Biên & cạnh)
    ↓
[Giai Đoạn 2] Feature Fusion trong Vision Encoder
    ├─ High-Freq Projection
    └─ Learnable Fusion Gate
    ↓
[Giai Đoạn 3] Refined Saliency Map Generation
    ├─ Gradient Computation (Semantic + Frequency)
    ├─ Edge-Aware Smoothing
    └─ Confidence Mapping
    ↓
[Giai Đoạn 4] Post-Processing & SAM Prompts
    ├─ ROI Extraction
    ├─ SAM Prompt Generation
    └─ Mask Refinement
    ↓
Final Segmentation Mask
```

## Các Thành Phần Chính

### 1. Wavelet Preprocessor (`wavelet_preprocessor.py`)

**Chức năng**: Thực hiện Discrete Wavelet Transform (DWT) để tách ảnh thành thành phần tần số

```python
from frequency_aware import DualStreamPreprocessor

preprocessor = DualStreamPreprocessor(
    wavelet_type='haar',  # Loại wavelet
    image_size=224        # Kích thước chuẩn
)

result = preprocessor(image_path)
# Trả về:
# - original_stream: Ảnh chuẩn hóa cho BiomedCLIP
# - wavelet_components: LL, LH, HL, HH
# - high_freq_enhanced: Thành phần tần số cao (ranh giới)
```

**Đầu ra Wavelet**:
- **LL (Low-Low)**: Hình dáng tổng quát (tương tự ảnh thu nhỏ)
- **LH (Low-High)**: Chi tiết ngang
- **HL (High-Low)**: Chi tiết dọc
- **HH (High-High)**: Chi tiết chéo
- **High-freq merged**: LH + HL + HH (chứa ranh giới mà BiomedCLIP bỏ sót)

### 2. Feature Fusion (`feature_fusion.py`)

**Chức năng**: Inject thông tin tần số cao vào Vision Encoder của BiomedCLIP

**Ba bước**:

1. **HighFreqProjection**: Chuyển đổi wavelet components (H, W) → patch embeddings (num_patches, embedding_dim)
   
2. **FeatureFusionGate**: Học cách kết hợp features
   - Formula: `Feature_fused = Original + α × HighFreq`
   - α là parameter học được (adaptive)

3. **FrequencyAwareVisionEncoder**: Wrapper cho Vision Encoder

**Sử dụng**:
```python
from frequency_aware import FrequencyAwareVisionEncoder, make_frequency_aware

# Wrap encoder gốc
aware_encoder = make_frequency_aware(
    original_encoder,
    embedding_dim=768,
    fusion_ratio=0.1
)

# Forward pass với frequency injection
output = aware_encoder(
    x=image_tensor,
    high_freq_features=wavelet_features
)
```

### 3. Saliency Map Generation (`saliency_generation.py`)

**Chức năng**: Tạo saliency maps sắc nét nhờ thông tin tần số

**Quy trình**:
1. Compute similarity scores (semantic + frequency-aware)
2. Reshape từ patch-space → spatial map (14×14 → 224×224)
3. Edge-preserving smoothing (bilateral filter)
4. Contrast enhancement (CLAHE)
5. Morphological operations (open, close)

**Hai loại Generator**:

- **FrequencyAwareSaliencyGenerator**: Single-scale
- **MultiScaleSaliencyGenerator**: Multi-scale với aggregation

```python
from frequency_aware import MultiScaleSaliencyGenerator

generator = MultiScaleSaliencyGenerator(
    scales=[0.5, 1.0, 1.5],
    aggregation='weighted_mean'
)

result = generator(
    image_features=encoder_output,
    high_freq_features=freq_features,
    text_embedding=text_emb,
    image_tensor=image,
    target_size=(original_h, original_w)
)

# Trả về:
# - saliency_map: Saliency score [0, 1]
# - saliency_map_refined: Sau refinement
# - binary_mask: Segmentation mask
# - confidence_map: Độ tin cậy của mỗi pixel
```

### 4. Post-Processing (`postprocessing.py`)

**Các bước**:

1. **ROI Extraction**: 
   - Tìm connected components trong binary mask
   - Tính bounding box với padding
   - Sắp xếp theo confidence

2. **SAM Prompt Generation**:
   - Bbox prompts: Hộp bao quanh ROI
   - Point prompts: Tâm và boundary points
   - Combined: Cả hai

3. **Mask Refinement**:
   - Morphological operations (close, open)
   - Boundary refinement dùng saliency
   - Confidence-based filtering
   - Remove small objects

```python
from frequency_aware import FrequencyAwarePipeline

pipeline = FrequencyAwarePipeline(
    prompt_type='bbox',
    refine_masks=True
)

result = pipeline(
    saliency_map=saliency,
    binary_mask=binary_mask,
    confidence_map=confidence,
    sam_mask=sam_output  # Optional
)

# Trả về:
# - prompts: SAMPrompt với bboxes/points
# - refined_mask: Mask sau refinement
# - metrics: Các metrics đánh giá
```

## Sử Dụng

### 1. Preprocessing Batch Images

```bash
python frequency_aware_integration.py \
    --config config/freq_aware_config.yaml \
    --image_dir data/images/ \
    --output_dir output/freq_aware/ \
    --batch_process \
    --save_intermediates
```

### 2. Programmatic Usage

```python
from frequency_aware import (
    DualStreamPreprocessor,
    FrequencyAwareSaliencyGenerator,
    FrequencyAwarePipeline
)
import torch

# 1. Preprocessing
preprocessor = DualStreamPreprocessor()
result = preprocessor('image.jpg')
original_stream = result['original_stream']
high_freq_stream = result['high_freq_enhanced']

# 2. Generate Features (sử dụng BiomedCLIP model)
with torch.no_grad():
    image_features = biomedclip(original_stream)  # From your model
    text_embedding = biomedclip_text('cancer')    # From your model

# 3. Generate Saliency Map
saliency_gen = FrequencyAwareSaliencyGenerator()
saliency_result = saliency_gen(
    image_features=image_features,
    high_freq_features=high_freq_stream,
    text_embedding=text_embedding,
    image_tensor=original_stream,
    target_size=(original_h, original_w)
)

saliency_map = saliency_result['saliency_map_refined']
binary_mask = saliency_result['binary_mask']
confidence = saliency_result['confidence_map']

# 4. Generate SAM Prompts
postprocessor = FrequencyAwarePipeline()
prompt_result = postprocessor(
    saliency_map=saliency_map.numpy(),
    binary_mask=binary_mask.numpy(),
    confidence_map=confidence.numpy()
)

# Sử dụng prompts với SAM
sam_prompts = prompt_result['prompts']
# sam_output = sam_model(image, prompts)
```

### 3. Integration với Existing Code

```python
# Thay đổi hiện tại trong generate_saliency_maps.py
# Thêm frequency-aware preprocessing:

from frequency_aware import DualStreamPreprocessor, FrequencyAwareSaliencyGenerator

preprocessor = DualStreamPreprocessor()
saliency_gen = FrequencyAwareSaliencyGenerator()

# Trong vòng loop xử lý images:
image = Image.open(image_path).convert('RGB')

# Stream preprocessing
prep_result = preprocessor(np.array(image))
original = prep_result['original_stream']
high_freq = prep_result['high_freq_enhanced']

# Saliency generation với frequency awareness
saliency = saliency_gen(
    image_features=image_features,
    high_freq_features=high_freq.unsqueeze(0),
    text_embedding=text_embedding,
    image_tensor=original.unsqueeze(0),
    target_size=image.size[::-1]
)
```

## Cấu Hình

File cấu hình: `config/freq_aware_config.yaml`

**Các parameter quan trọng**:

```yaml
# Wavelet
wavelet_type: 'haar'  # Loại wavelet, có thể thay đổi
frequency_weight: 0.3 # Trọng số tần số trong fusion (0.0-1.0)

# Saliency Generation
use_multi_scale: true # Sử dụng multi-scale aggregation
scales: [0.5, 1.0, 1.5]
aggregation: 'weighted_mean'

# ROI Extraction
min_roi_size: 20  # Kích thước ROI tối thiểu
max_roi_count: 5  # Số ROI tối đa

# SAM Prompts
prompt_type: 'bbox'  # 'bbox', 'points', 'combined'

# Mask Refinement
confidence_threshold: 0.5
```

## Đầu Ra

**Intermediate Results** (nếu `save_intermediates=true`):
- `*_preprocessed.npz`: Wavelet components (LL, LH, HL, HH)

**Final Outputs** (từ pipeline):
- Saliency maps (raw + refined)
- Binary segmentation masks
- Confidence maps
- SAM prompts (bboxes, points)
- Refined masks sau SAM inference

## Performance Tips

1. **Batch Processing**: Xử lý theo batch để tăng speed
   ```python
   from frequency_aware import process_batch_dual_stream
   
   batch_result = process_batch_dual_stream(image_list, preprocessor, device='cuda')
   ```

2. **Multi-scale Aggregation**: Cân bằng giữa accuracy và speed
   - Dùng `scales=[0.5, 1.0]` cho speed
   - Dùng `scales=[0.5, 1.0, 1.5]` cho accuracy

3. **GPU Memory**: Nếu lỗi memory
   - Giảm `image_size` (224 → 128)
   - Tắt `save_intermediates`
   - Giảm `point_count`

## Troubleshooting

### Issue: ImportError cho pywt
```bash
pip install PyWavelets
```

### Issue: CUDA out of memory
```python
# Giảm batch size hoặc image size
preprocessor = DualStreamPreprocessor(image_size=128)
```

### Issue: Saliency map quá mờ
```yaml
# Tăng frequency_weight
frequency_weight: 0.5  # Từ 0.3 lên 0.5
```

### Issue: Quá nhiều false positives
```yaml
# Tăng threshold
threshold: 0.5  # Từ 0.3 lên 0.5
# Tăng min_roi_size
min_roi_size: 50  # Từ 20 lên 50
```

## Metrics & Evaluation

Sau inference với SAM, có thể đánh giá:

```python
from frequency_aware.postprocessing import MaskRefinement

refiner = MaskRefinement()
result = refiner(sam_mask, saliency_map, confidence_map)

metrics = result['metrics']
# - mask_area: Tổng số pixel của mask
# - mask_coverage: % ảnh được cover
# - mean_saliency: Saliency trung bình trong mask
# - mean_confidence: Confidence trung bình
```

## References

- **FMISeg**: Frequency-domain Multi-modal Fusion for Language-guided Medical Image Segmentation
- **MedCLIP-SAMv2**: Towards Universal Text-Driven Medical Image Segmentation
- **Wavelet Transform**: Discrete Wavelet Transform (DWT) cho feature extraction

## Citing

Nếu sử dụng frequency-aware integration, vui lòng cite:
- FMISeg paper
- MedCLIP-SAMv2 paper
- Wavelet transform references
