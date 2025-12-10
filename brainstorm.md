# Thiết Kế Chi Tiết: Cải Thiện MedCLIP-SAMv2 với FMISeg (Frequency-aware Integration)

Tài liệu này mô tả chi tiết luồng xử lý dữ liệu (Data Pipeline) để tích hợp sức mạnh xử lý miền tần số (Wavelet) của FMISeg vào pipeline gốc của MedCLIP-SAMv2. Mục tiêu là tạo ra Saliency Maps sắc nét hơn, giúp SAM phân đoạn chính xác hơn mà không làm mất đi khả năng Zero-shot.

## 1. Tổng quan ý tưởng: "Đôi mắt Frequency-aware"

* **Mô hình gốc (BiomedCLIP):** Giỏi nhận diện ngữ nghĩa (vật thể là "cái gì") nhưng thường tạo ra các vùng kích hoạt mờ (blob), thiếu chính xác về vị trí ranh giới.
* **Mô hình tích hợp (FMISeg):** Giỏi phát hiện chi tiết cạnh, nhiễu và kết cấu thông qua biến đổi Wavelet.
* **Giải pháp:** "Tiêm" (inject) thông tin tần số cao từ FMISeg vào luồng xử lý của BiomedCLIP để định hướng sự chú ý vào đúng ranh giới tổn thương.

---

## 2. Chi tiết Luồng xử lý dữ liệu (Data Pipeline)

Quy trình cải tiến được chia thành 4 giai đoạn chính:

### Giai đoạn 1: Tiền xử lý song song (Dual-Stream Preprocessing)

Thay vì chỉ đưa ảnh gốc vào một luồng duy nhất, dữ liệu đầu vào được tách làm hai ngay từ đầu.

* **Input:** Ảnh y tế gốc (RGB hoặc Grayscale).
* **Luồng 1 (BiomedCLIP Gốc):**
    * Ảnh đi qua quy trình chuẩn hóa (Normalization) tiêu chuẩn của CLIP.
    * Mục đích: Giữ lại thông tin màu sắc/cường độ sáng để nhận diện ngữ nghĩa.
* **Luồng 2 (FMISeg - Wavelet):**
    * Ảnh đi qua hàm `DWT_2D` (sử dụng file `utils/wave.py` từ FMISeg).
    * **Đầu ra:** 4 thành phần tần số:
        * **LL (Low):** Hình dáng tổng quát (tương tự ảnh thu nhỏ).
        * **LH, HL, HH (High):** Chi tiết cạnh ngang, dọc và chéo.
    * **Trọng tâm:** Chúng ta đặc biệt quan tâm đến tổng hợp của **LH + HL + HH**, nơi chứa thông tin ranh giới mà BiomedCLIP thường bỏ sót.

### Giai đoạn 2: Hợp nhất đặc trưng (Feature Fusion) trong Image Encoder

Đây là bước can thiệp vào kiến trúc mô hình (Model Architecture Modification).

* **File liên quan:** `biomedclip_finetuning/open_clip/src/open_clip/transformer.py` (hoặc `model.py` của CLIP).
* **Quy trình hiện tại:** Vision Transformer (ViT) chia ảnh thành các ô vuông (patches) và nhúng (embed) chúng thành vector.
* **Quy trình cải tiến:**
    1.  Tạo embedding riêng cho các thành phần tần số cao (High-freq components) từ Giai đoạn 1.
    2.  **Cơ chế Gating/Fusion:** Trước khi đưa vector vào các lớp Transformer Block, thực hiện cộng đặc trưng tần số vào đặc trưng không gian (spatial features).
    
    **Công thức khái quát:**
    $$Feature_{input} = PatchEmbed(Image) + \alpha \times Projection(HighFreqWavelet)$$
    
    *(Lưu ý: $\alpha$ là trọng số nhỏ, ví dụ 0.1, hoặc tham số học được để tránh làm hỏng các trọng số pretrained của CLIP).*

### Giai đoạn 3: Tạo Saliency Map "Sắc nét" (Refined Saliency Map Generation)

Đây là bước quyết định chất lượng đầu vào (Prompt) cho SAM.

* **File liên quan:** `saliency_maps/scripts/generate_saliency_maps.py`.
* **Vấn đề của MedCLIP gốc:** Saliency Map được tạo từ Gradient của văn bản lên ảnh. Nếu đặc trưng ảnh mờ, Gradient sẽ bị lan tỏa ra vùng xung quanh (bleed out), khiến bản đồ nhiệt lớn hơn thực tế.
* **Cơ chế sau cải tiến:**
    * Vì `Feature_input` (từ Giai đoạn 2) đã chứa thông tin cạnh sắc nét từ Wavelet, các nơ-ron chịu trách nhiệm nhận diện biên (boundary neurons) sẽ kích hoạt mạnh hơn.
    * Gradient sẽ tập trung (focus) vào vùng có sự thay đổi tần số cao (tức là viền khối u/tổn thương).
    * **Kết quả:** Saliency Map ít nhiễu nền (background noise) và bám sát rìa vật thể hơn.

### Giai đoạn 4: Hậu xử lý & SAM Inference

* **Thresholding:** Từ Saliency Map đã được tinh chỉnh, thuật toán cắt ngưỡng (thresholding) sẽ tạo ra các vùng quan tâm (ROI).
* **Prompt Generation:** Tạo Bounding Box (Hộp bao) hoặc Points (Điểm) từ ROI. Nhờ Map sắc nét, hộp bao sẽ "khít" (tight) với tổn thương hơn.
* **SAM Inference:** SAM nhận prompt chính xác -> Trả về Mask phân đoạn chuẩn xác.