# AI Vision LPR on Raspberry Pi

## Tổng quan dự án

Dự án tập trung vào việc nhận diện biển số xe (LPR) tại Việt Nam, tối ưu hóa để chạy trên phần cứng Raspberry Pi 5. Hệ thống sử dụng pipeline hai giai đoạn:

- **Detection** (YOLO/Custom)
- **Recognition** (OCR)

## Tech Stack

- **Engine:** Python 3.11+
- **Vision:** OpenCV, NumPy, `cv2.dnn_superres`
- **Deployment:** Raspberry Pi 5 (ARM Architecture)
- **Models:** YOLOv5/v8/v10, Tiny SR (FSRCNN)

## Chiến lược tối ưu hóa

Khi bảo trì hoặc nâng cấp dự án, hãy tuân thủ các nguyên tắc sau để đảm bảo hiệu suất trên Raspberry Pi.

### 1. Xử lý ảnh đầu vào (Preprocessing)

- **Detect trên low-resolution:** Resize frame về `320x320` hoặc `640x640` để chạy model detect nhanh hơn.
- **Crop từ source-resolution:** Khi đã có tọa độ `(x, y, w, h)` từ model detect, **bắt buộc** quay lại ảnh gốc (full resolution) để crop biển số.
- **Không được crop từ ảnh đã resize.**

### 2. Nâng cao chất lượng ảnh (Enhancement)

Nếu ảnh biển số sau khi crop có chiều cao `H < 40 px`:

- Sử dụng Tiny SR (`FSRCNN_x2`) qua `cv2.dnn_superres`
- Áp dụng CLAHE để xử lý ánh sáng chói hoặc quá tối:
  - `clipLimit=2.0`
  - `tileGridSize=(8, 8)`
- Sử dụng `cv2.detailEnhance()` để làm nổi bật nét ký tự trước khi đưa vào OCR

### 3. Logic hậu kỳ (Post-processing)

- Sử dụng Regex để kiểm tra định dạng biển số Việt Nam
  - **Xe máy:** `29-A1 123.45`
  - **Ô tô:** `30L-999.99`
- Thêm padding khoảng `5-10%` xung quanh bounding box trước khi crop để tránh mất rìa ký tự

## Cấu trúc thư mục quan trọng

- `/models`: Chứa các file `.tflite`, `.onnx` và `.pb` (FSRCNN)
- `/src/inference.py`: File thực thi chính
- `/debug_crops`: Thư mục lưu ảnh biển số lỗi để phân tích chất lượng
- `/data`: Chứa video test và annotations

## Hướng dẫn dành cho AI Agent (Claude Code)

Khi được yêu cầu chỉnh sửa code, hãy:

### 1. Tối ưu bộ nhớ và hiệu năng

- Ưu tiên các phép toán vector hóa của NumPy thay vì dùng vòng lặp `for`

### 2. Git Flow

- Sử dụng `git-skill` để tự động commit
- Commit message cần mô tả rõ module nào đã được tối ưu

### 3. Logging

Thêm log về thời gian xử lý (**latency**) của từng bước:

- Detect
- Crop
- SR
- OCR

## Quy trình chạy test video

1. Đọc video từ `/data/test_video.mp4`
2. Chạy pipeline
3. Lưu các kết quả lỗi vào `/debug_crops`
4. Nếu tỉ lệ OCR thành công `< 80%`, tự động đề xuất áp dụng thêm các bước tiền xử lý trong `instruction.md`

## Nguyên tắc ưu tiên

Luôn ưu tiên sự cân bằng giữa:

- **Accuracy** (độ chính xác)
- **FPS** (tốc độ khung hình)

vì thiết bị đầu cuối là **Raspberry Pi 5**.
