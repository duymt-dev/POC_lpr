# Kế hoạch triển khai - AI Vision LPR

> Cập nhật: 2026-04-12

## Mục tiêu

Xuất video output từ video test, trong đó:
- **Bbox** vẽ quanh biển số xe
- **Góc phải trên video**: hiển thị text biển số thật to, dạng danh sách dọc

## Pipeline

```
Video Input
    │
    ▼
[1] Đọc frame
    │
    ▼
[2] Detect biển số (YOLOv5 - LP_detector_nano_61.pt)
    │   - Input: frame resize 640x640
    │   - Output: bounding boxes (x, y, w, h, conf)
    │
    ▼
[3] Crop biển số từ ảnh GỐC (full resolution)
    │   - Thêm padding 5-10%
    │
    ▼
[4] OCR từng biển số (YOLOv5 - LP_ocr_nano_62.pt)
    │   - Detect từng ký tự
    │   - Sắp xếp theo vị trí x (trái → phải)
    │   - Ghép thành chuỗi biển số
    │
    ▼
[5] Vẽ kết quả lên frame
    │   - Bbox xanh lá quanh biển số
    │   - Text biển số to ở góc phải trên
    │
    ▼
[6] Ghi frame vào video output
```

## Các bước thực hiện

### Bước 1: Kiểm tra môi trường
- [x] Python 3.11+
- [x] Đã có `ultralytics`, `opencv-python`, `numpy`, `torch`
- [x] Xác nhận môi trường có GPU NVIDIA

### Bước 2: Inspect model
- [x] Load `LP_detector_nano_61.pt` → class `license_plate`
- [x] Load `LP_ocr_nano_62.pt` → OCR detect từng ký tự, 30 class
- [x] Ghi kết quả vào `/docs/model_info.md`
- [x] Thiết lập tương thích YOLOv5 cũ bằng `vendor/yolov5`

### Bước 3: Viết code inference
- [x] Tạo `src/inference.py`
- [x] Pipeline: read video → detect → crop → OCR → draw → write video
- [x] Output: `output/result.mp4`
- [x] Hiển thị kết quả bằng panel bên phải để text biển số to và dễ đọc

### Bước 4: Test & Debug
- [x] Smoke test 10 frame với video test
- [x] Chạy full video test
- [x] Kiểm tra bbox có chính xác không
- [x] Kiểm tra OCR có đọc đúng biển số không
- [x] Lưu crop lỗi vào `/debug_crops/`
- [x] Phân tích debug crops và thêm enhancement

## Kết quả hiện tại

- ✅ Đã tạo file output: `output/result.mp4`
- ✅ Số frame output: `1841`
- ✅ FPS output: `29.984`
- ✅ Kích thước output: `1700x720`
- ✅ Layout output:
  - bên trái: video gốc có bbox biển số
  - bên phải: panel nền tối hiển thị danh sách biển số thật to
- ✅ OCR đã được cải thiện:
  - chuẩn hóa text nhận diện
  - lọc theo regex biển số Việt Nam
  - voting nhiều frame để giảm nhấp nháy text
  - lưu crop lỗi vào `debug_crops/` khi OCR không hợp lệ
  - enhancement (upscale + CLAHE + detailEnhance) cho crop nhỏ
- ✅ Debug crops: 135 ảnh lỗi được lưu để phân tích
  - 87 EMPTY (không detect ký tự)
  - 48 partial (detect 1-3 ký tự)
  - Enhancement giúp tăng từ 18% lên 36% detect được ký tự

## Tài liệu chi tiết

Xem `docs/results.md` để biết thêm chi tiết về:
- Pipeline đầy đủ
- Hiệu suất và chất lượng OCR
- Hạn chế hiện tại
- Đề xuất cải tiến tiếp theo

## Cấu trúc thư mục

```
E:\Kenmon\
├── docs/
│   ├── plan.md              ← file này
│   └── model_info.md        ← thông tin model
├── models/
│   ├── LP_detector_nano_61.pt
│   └── LP_ocr_nano_62.pt
├── src/
│   └── inference.py         ← code chính
├── video test/
│   └── pwf2amf0...mp4
├── output/                  ← video kết quả
└── debug_crops/             ← ảnh lỗi (sẽ tạo)
```

## Ghi chú

- Test trên Windows trước, deploy Raspberry Pi sau
- Model: YOLOv5 nano
- Hiển thị: danh sách dọc tất cả biển số ở panel riêng bên phải video
