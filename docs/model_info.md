# Model Info

> Cập nhật: 2026-04-12

## Kết quả inspect

### 1. Detector
- File: `models/LP_detector_nano_61.pt`
- Định dạng: checkpoint YOLOv5 cũ
- Số class: `1`
- Class names:

```json
[
  "license_plate"
]
```

### 2. OCR
- File: `models/LP_ocr_nano_62.pt`
- Định dạng: checkpoint YOLOv5 cũ
- OCR hoạt động theo kiểu: **detect từng ký tự riêng lẻ**
- Số class: `30`
- Class names:

```json
[
  "1", "2", "3", "4", "5", "6", "7", "8", "9",
  "A", "B", "C", "D", "E", "F", "G", "H",
  "K", "L", "M", "N", "P", "S", "T", "U", "V",
  "X", "Y", "Z", "0"
]
```

## Ghi chú tương thích

- Hai model không load trực tiếp bằng `ultralytics.YOLO(...)` hiện đại.
- Nguyên nhân: checkpoint phụ thuộc module `models.yolo` từ repo YOLOv5 cũ.
- Cách xử lý đã chọn:
  - Clone repo `ultralytics/yolov5` vào `vendor/yolov5`
  - Dùng loader tương thích từ repo này để chạy inference

## Ảnh hưởng đến triển khai

- `src/inference.py` sẽ dùng backend YOLOv5 compatibility thay vì API Ultralytics mới.
- Detector sẽ trả về bbox biển số.
- OCR sẽ trả về bbox từng ký tự, sau đó code sẽ:
  - sắp xếp theo dòng và theo trục `x`
  - ghép thành text biển số
  - hiển thị ở góc phải video
