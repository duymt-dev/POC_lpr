# Kết quả triển khai - AI Vision LPR

> Cập nhật: 2026-04-12

## Tổng quan

Đã triển khai thành công pipeline nhận diện biển số xe Việt Nam từ video, chạy trên Windows với GPU NVIDIA GTX 1650.

## Output cuối cùng

### Video kết quả
- File: `output/result.mp4`
- Frames: `1841`
- FPS: `29.984`
- Kích thước: `1700x720` (video gốc 1280x720 + panel 420px)
- Dung lượng: ~97 MB

### Layout hiển thị
- **Bên trái (1280px)**: Video gốc với bbox biển số
  - Bbox xanh lá: biển số hợp lệ
  - Bbox cam: biển số chưa hợp lệ
- **Bên phải (420px)**: Panel nền tối
  - Text biển số rất to (font size 1.1)
  - Danh sách dọc tất cả biển số trong frame
  - Hiển thị latency từng bước (detect, OCR, frame)

## Pipeline đã triển khai

```
Video Input
    ↓
[1] Detect biển số (YOLOv5 nano)
    ↓
[2] Crop từ ảnh gốc + padding 8%
    ↓
[3] Enhancement (nếu H < 40px)
    - Upscale bằng INTER_CUBIC
    - CLAHE (clipLimit=2.0)
    - detailEnhance
    ↓
[4] OCR từng ký tự (YOLOv5 nano)
    ↓
[5] Ghép ký tự theo dòng và vị trí x
    ↓
[6] Chuẩn hóa + lọc regex biển số VN
    ↓
[7] Voting nhiều frame (window=8)
    ↓
[8] Hiển thị text ổn định
```

## Các tính năng đã thực hiện

### 1. Detection & OCR
- ✅ Load model YOLOv5 cũ qua `vendor/yolov5`
- ✅ Detect biển số trên frame resize 640x640
- ✅ Crop từ ảnh gốc full resolution
- ✅ OCR detect từng ký tự (30 class: 0-9, A-Z trừ I,J,O,Q,R,W)

### 2. Enhancement
- ✅ Upscale crop nhỏ (H < 40px) bằng INTER_CUBIC
- ✅ CLAHE để cân bằng độ sáng
- ✅ detailEnhance để làm nổi ký tự

### 3. Post-processing
- ✅ Chuẩn hóa text (loại ký tự đặc biệt)
- ✅ Format biển số VN: `29A1-123.45` hoặc `30L-999.99`
- ✅ Regex lọc biển số hợp lệ (5 pattern phổ biến)

### 4. Tracking & Stability
- ✅ Track xe theo vị trí bbox giữa các frame
- ✅ Voting text OCR trong cửa sổ 8 frame
- ✅ Hiển thị text ổn định nhất, giảm nhấp nháy

### 5. Debug
- ✅ Lưu crop lỗi vào `debug_crops/` (giới hạn 150 ảnh)
- ✅ Tên file chứa raw text và formatted text để dễ phân tích

## Hiệu suất

### Latency trung bình (GTX 1650)
- Detect: ~15-25ms/frame
- OCR: ~10-30ms/plate (tùy số lượng xe)
- Total: ~30-60ms/frame

### Chất lượng OCR
- **Crop tốt (H ≥ 40px)**: OCR chính xác cao, pass regex VN
- **Crop nhỏ (H < 40px)**: Enhancement giúp tăng từ 18% lên 36% detect được ký tự, nhưng vẫn chưa đủ pass regex
- **Debug crops**: 135 crop lỗi được lưu lại
  - 87 crop EMPTY (không detect được ký tự nào)
  - 48 crop partial (detect được 1-3 ký tự)

## Hạn chế hiện tại

### 1. Crop quá nhỏ
- Nhiều biển số xa camera có chiều cao chỉ 12-22px
- Enhancement giúp cải thiện nhưng chưa đủ để OCR chính xác
- Cần model OCR mạnh hơn hoặc super-resolution tốt hơn (FSRCNN, ESRGAN)

### 2. Góc chụp xấu
- Biển số nghiêng, mờ, bị che khuất
- Ánh sáng chói hoặc quá tối
- CLAHE giúp phần nào nhưng không phải lúc nào cũng đủ

### 3. Regex VN chưa bao phủ hết
- Hiện tại có 5 pattern phổ biến
- Một số biển số đặc biệt (ngoại giao, quân đội) chưa cover

## Cấu trúc file

```
E:\Kenmon\
├── docs/
│   ├── plan.md              # Kế hoạch từng bước
│   ├── model_info.md        # Thông tin model
│   └── results.md           # File này
├── models/
│   ├── LP_detector_nano_61.pt
│   └── LP_ocr_nano_62.pt
├── src/
│   └── inference.py         # Script chính
├── vendor/
│   └── yolov5/              # YOLOv5 compatibility
├── video test/
│   └── pwf2amf0...mp4
├── output/
│   └── result.mp4           # Video output cuối cùng
└── debug_crops/             # 135 ảnh crop lỗi
```

## Cách chạy

### Chạy với cấu hình mặc định
```bash
python src/inference.py
```

### Chạy với tham số tùy chỉnh
```bash
python src/inference.py \
  --video "path/to/video.mp4" \
  --output "output/custom.mp4" \
  --det-conf 0.4 \
  --ocr-conf 0.3 \
  --stable-window 10 \
  --panel-width 500
```

### Các tham số quan trọng
- `--det-conf`: Confidence threshold cho detector (default: 0.35)
- `--ocr-conf`: Confidence threshold cho OCR (default: 0.25)
- `--stable-window`: Số frame để voting text (default: 8)
- `--padding`: Padding ratio quanh bbox (default: 0.08)
- `--panel-width`: Chiều rộng panel bên phải (default: 420)
- `--debug-limit`: Số crop lỗi tối đa lưu (default: 150)

## Đề xuất cải tiến tiếp theo

### 1. Super-resolution mạnh hơn
- Thêm FSRCNN hoặc ESRGAN cho crop nhỏ
- Download pretrained model từ OpenCV zoo

### 2. OCR model tốt hơn
- Retrain OCR model với augmentation mạnh hơn
- Thêm data biển số nhỏ, mờ, nghiêng

### 3. Regex và format
- Bổ sung pattern cho biển ngoại giao, quân đội
- Thêm logic sửa lỗi OCR phổ biến (B↔8, D↔0, etc.)

### 4. Tracking nâng cao
- Dùng DeepSORT thay vì simple centroid tracking
- Giữ history dài hơn cho xe di chuyển chậm

### 5. Deploy Raspberry Pi
- Convert model sang TFLite hoặc ONNX
- Tối ưu inference cho ARM CPU
- Test FPS trên Pi 5
