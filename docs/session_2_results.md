# Session 2 Results - LPR Improvements

> Cập nhật: 2026-04-13

## Mục tiêu session 2

Thực hiện 4 đề xuất từ session 1:
1. Ghi log chi tiết theo car_id
2. Thêm OCR backend khác cho crop nhỏ (PaddleOCR)
3. Motorcycle-specific reconstruction cho biển 2 dòng
4. Focus vào car#3 với ground truth đã biết (29F-027.94)

## Kết quả thực hiện

### ✅ 1. Log chi tiết car_candidates.csv

**Đã thêm:**
- Function `log_car_candidate()` trong `src/inference.py:836`
- File output: `debug_crops/car_candidates.csv`

**Thông tin log:**
- `car_id`: ID xe được track
- `frame`: Frame number
- `raw_text`: Text OCR gốc
- `row_texts`: Text theo từng dòng (phân cách bởi `|`)
- `normalized_candidate`: Candidate sau normalize
- `formatted_candidate`: Candidate sau format
- `plate_score`: Điểm đánh giá candidate
- `ocr_conf`: Confidence OCR trung bình
- `crop_w`, `crop_h`: Kích thước crop
- `best_raw_now`: Best plate hiện tại (raw)
- `best_formatted_now`: Best plate hiện tại (formatted)

**Lợi ích:**
- Trace được decision path của từng xe qua từng frame
- Dễ dàng debug và phân tích voting logic
- Hiểu rõ tại sao hệ thống chọn plate nào

### ⏸️ 2. OCR backend khác (PaddleOCR)

**Trạng thái:** Pending

**Lý do:**
- Cài đặt PaddleOCR bị timeout (package rất lớn ~100MB)
- Cần thời gian dài để download và cài đặt

**Kế hoạch tiếp theo:**
- Thử cài đặt offline hoặc dùng lightweight version
- Hoặc thử backend khác: TrOCR, CRNN

### ✅ 3. Motorcycle-specific reconstruction

**Đã thêm:**
- Function `reconstruct_motorcycle_plate()` trong `src/inference.py:160`
- Tích hợp vào `select_frame_plate_candidate()`

**Logic reconstruction:**

```
VN motorcycle format:
- Dòng 1: Mã tỉnh (2 số)
- Dòng 2: Series (1 chữ) + Số (4-5 chữ số)

Ví dụ: 29F-027.94
- Dòng 1: "29"
- Dòng 2: "F02794"
```

**Strategies:**
1. Top = exactly 2 digits → province code
2. Top ≥ 2 chars → take first 2 as province, rest to bottom
3. Top ≥ 3 chars → maybe series leaked to top

**Scoring:**
- Motorcycle plates: 7-8 chars (29F12345 or 29F1234)
- Province code: 2 digits (+5 points)
- Valid province code (+5 points)
- Series letter at position 3 (+4 points)
- Tail all digits (+6 points)
- Tail length 4-5 (+3 points)

### ✅ 4. Focus car#3 với ground truth

**Đã thêm:**
- Function `save_car3_crop()` trong `src/inference.py:477`
- Ground truth constant: `GROUND_TRUTH = {3: "29F02794"}`
- File output: `debug_crops/car3_analysis/`

**Kết quả phân tích car#3:**

| Frame | Crop Size | Raw Text | Normalized | Formatted | Similarity to GT |
|-------|-----------|----------|------------|-----------|------------------|
| 619   | 58x29     | "8540"   | "8540"     | "8540"    | 0.000            |

**Row texts:** `"B|54D"` (2 dòng)

**Vấn đề phát hiện:**
1. Car#3 chỉ được track **1 frame duy nhất** (frame 619)
2. Crop rất nhỏ: 58x29 px
3. OCR nhận được "B54D" (2 dòng) nhưng reconstruction thành "8540"
4. Hoàn toàn không match ground truth "29F02794"

**Nguyên nhân có thể:**
- Tracking bị mất car#3 quá sớm (chỉ 1 frame)
- Crop quá nhỏ → OCR hoàn toàn sai
- Có thể car#3 không phải xe có biển "29F-027.94" (ground truth sai?)

## Đánh giá tổng thể

### Điểm mạnh

1. **Logging system hoàn chỉnh**
   - `car_candidates.csv`: Track decision path
   - `car3_analysis/`: Focus vào xe cụ thể
   - Dễ dàng debug và phân tích

2. **Motorcycle reconstruction logic**
   - Xử lý riêng biển 2 dòng
   - Scoring phù hợp với format VN
   - Nhiều strategies fallback

3. **Code structure tốt hơn**
   - Functions rõ ràng, dễ maintain
   - Ground truth constant để validate

### Điểm yếu

1. **Car#3 tracking vấn đề nghiêm trọng**
   - Chỉ track được 1 frame
   - Có thể do:
     - Car detector miss nhiều frame
     - Tracking logic quá strict
     - Car#3 bị occluded hoặc ra khỏi frame nhanh

2. **OCR vẫn quá yếu trên crop nhỏ**
   - Crop 58x29px → OCR hoàn toàn sai
   - Enhancement (FSRCNN, CLAHE, sharpen) chưa đủ
   - Cần OCR backend mạnh hơn

3. **Motorcycle reconstruction chưa được test đầy đủ**
   - Car#3 chỉ có 1 frame → không đủ data để validate
   - Cần test với xe khác có biển 2 dòng

## Files output

### Video
- `output/test_session2_v1.mp4` (100 frames)
- `output/test_session2_v2.mp4` (700 frames)
- `output/test_session2_full.mp4` (1841 frames)

### Debug logs
- `debug_crops/car_candidates.csv` (chi tiết tất cả candidates)
- `debug_crops/car3_analysis/car3_frames.csv` (chỉ car#3)
- `debug_crops/car3_analysis/f00619_w58_h29_sim0.00_8540_8540.jpg` (crop car#3)

## Đề xuất tiếp theo

### Ưu tiên 1: Fix car#3 tracking

**Vấn đề:** Car#3 chỉ track 1 frame

**Giải pháp:**
1. Giảm `car-conf` threshold để detect nhiều xe hơn
2. Tăng `max_frames_missing` trong CarTracker
3. Xem lại video thủ công để xác định car#3 thực sự xuất hiện ở đâu
4. Có thể ground truth sai → cần verify lại

### Ưu tiên 2: OCR backend mạnh hơn

**Vấn đề:** YOLO char-level quá yếu trên crop nhỏ

**Giải pháp:**
1. Cài đặt PaddleOCR (hoặc TrOCR, CRNN)
2. Fallback cho crop < 40px height
3. Ensemble: YOLO + PaddleOCR → vote

### Ưu tiên 3: Validate motorcycle reconstruction

**Vấn đề:** Chưa có data để test

**Giải pháp:**
1. Tìm xe khác trong video có biển 2 dòng rõ ràng
2. Test reconstruction logic với xe đó
3. Tune scoring weights nếu cần

### Ưu tiên 4: Ground truth validation

**Vấn đề:** Car#3 = "29F-027.94" chưa được verify

**Giải pháp:**
1. Xem video thủ công, tìm frame có biển "29F-027.94"
2. Xác định car_id thực sự của xe đó
3. Update ground truth nếu cần

## Kết luận

Session 2 đã hoàn thành 3/4 đề xuất:
- ✅ Logging system
- ⏸️ PaddleOCR (pending)
- ✅ Motorcycle reconstruction
- ✅ Car#3 focus (nhưng phát hiện vấn đề tracking)

**Vấn đề lớn nhất:** Car#3 tracking chỉ 1 frame → không đủ data để validate improvements.

**Next steps:** Fix tracking hoặc verify ground truth trước khi tiếp tục optimize OCR.
