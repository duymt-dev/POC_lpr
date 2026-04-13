# Kiểm tra bổ sung - AI Vision LPR

> Cập nhật: 2026-04-12

## Các test bổ sung

Đã chạy thêm 2 phiên bản với các ngưỡng confidence khác nhau để đánh giá tác động đến kết quả OCR.

### Test 1: Confidence cao hơn
- **File**: `output/result_det0.4_ocr0.3_win10.mp4`
- **Tham số**: `--det-conf 0.4 --ocr-conf 0.3 --stable-window 10`
- **Kích thước**: 97,270,571 bytes (~92.8 MB)
- **Mô tả**: Ngưỡng confidence cao hơn giúp giảm false positive nhưng có thể bỏ qua một số biển số yếu

### Test 2: Confidence thấp hơn
- **File**: `output/result_det0.07_ocr0.07_win10.mp4`
- **Tham số**: `--det-conf 0.07 --ocr-conf 0.07 --stable-window 10`
- **Kích thước**: 98,251,783 bytes (~93.7 MB)
- **Mô tả**: Ngưỡng confidence thấp hơn giúp detect được nhiều bbox hơn nhưng tăng nguy cơ false positive

## So sánh kích thước file

| File | Kích thước (bytes) | Kích thước (MB) | Ghi chú |
|------|-------------------|-----------------|---------|
| result.mp4 | 97,150,487 | 92.7 | Cấu hình mặc định (det-conf 0.35, ocr-conf 0.25) |
| result_det0.4_ocr0.3_win10.mp4 | 97,270,571 | 92.8 | Confidence cao hơn |
| result_det0.07_ocr0.07_win10.mp4 | 98,251,783 | 93.7 | Confidence thấp hơn |
| result_enhanced_test.mp4 | 2,382,643 | 2.3 | Test ngắn 50 frame |
| result_stable_test.mp4 | 1,414,305 | 1.3 | Test ngắn 30 frame |

## Đề xuất tiếp theo

1. **Phân tích chi tiết**: So sánh số lượng biển số được detect đúng trong mỗi phiên bản
2. **Tối ưu ngưỡng confidence**: Tìm giá trị cân bằng giữa recall và precision
3. **Thêm metrics**: Tính toán F1-score, precision, recall nếu có ground truth annotation
4. **Deploy Raspberry Pi**: Chuyển model sang TFLite và test trên Pi 5 thực tế

## Cách so sánh trực quan

Mở cùng lúc các file video để quan sát:
- result.mp4 (cấu hình mặc định)
- result_det0.4_ocr0.3_win10.mp4 (confidence cao)
- result_det0.07_ocr0.07_win10.mp4 (confidence thấp)

Lưu ý sự khác biệt về:
- Số bbox được vẽ
- Độ ổn định của text biển số
- Số lượng false positive/false negative