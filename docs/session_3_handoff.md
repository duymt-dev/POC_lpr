# Session 3 Handoff

> Cập nhật: 2026-04-13

## Tóm tắt session 2

Session 2 đã thực hiện 3/4 đề xuất từ session 1:
- ✅ Thêm logging chi tiết `car_candidates.csv`
- ⏸️ PaddleOCR (pending - cài đặt timeout)
- ✅ Motorcycle-specific reconstruction
- ✅ Car#3 focus (phát hiện vấn đề tracking nghiêm trọng)

## Vấn đề nghiêm trọng phát hiện

### Car#3 tracking chỉ 1 frame

**Hiện trạng:**
- Car#3 chỉ xuất hiện ở frame 619
- Crop size: 58x29 px (rất nhỏ)
- OCR: "8540" (raw_text có row_texts "B|54D")
- Ground truth: "29F02794"
- Similarity: 0.000 (hoàn toàn không match)

**Nguyên nhân có thể:**
1. Tracking bị mất xe quá sớm
2. Car detector miss nhiều frame
3. Ground truth sai (car#3 không phải xe có biển "29F-027.94")
4. Xe bị occluded hoặc ra khỏi frame nhanh

## Files quan trọng

### Code
- `src/inference.py` - Pipeline chính với các improvements mới

### Logs mới
- `debug_crops/car_candidates.csv` - Log chi tiết tất cả candidates theo car_id
- `debug_crops/car3_analysis/car3_frames.csv` - Chỉ car#3
- `debug_crops/car3_analysis/*.jpg` - Crops của car#3

### Video output
- `output/test_session2_full.mp4` - Full video với improvements session 2

### Docs
- `docs/session_2_results.md` - Báo cáo chi tiết session 2
- `docs/session_2_handoff.md` - File này

## Improvements đã thêm

### 1. Logging system

**Function:** `log_car_candidate()` (line 836)

**Output:** `debug_crops/car_candidates.csv`

**Columns:**
- car_id, frame, raw_text, row_texts
- normalized_candidate, formatted_candidate
- plate_score, ocr_conf
- crop_w, crop_h
- best_raw_now, best_formatted_now

### 2. Motorcycle reconstruction

**Function:** `reconstruct_motorcycle_plate()` (line 160)

**Logic:**
- Dòng 1: Province code (2 digits)
- Dòng 2: Series (1 letter) + Numbers (4-5 digits)
- Multiple strategies với scoring riêng cho motorcycle

**Tích hợp:** `select_frame_plate_candidate()` (line 260)

### 3. Car#3 focus

**Function:** `save_car3_crop()` (line 477)

**Ground truth:** `GROUND_TRUTH = {3: "29F02794"}` (line 465)

**Output:** `debug_crops/car3_analysis/`

**Features:**
- Lưu tất cả crops của car#3
- Tính similarity với ground truth
- CSV log chi tiết

## Đề xuất cho session 3

### Ưu tiên 1: Investigate car#3 tracking issue

**Mục tiêu:** Hiểu tại sao car#3 chỉ track 1 frame

**Các bước:**
1. Xem video thủ công, tìm frame có biển "29F-027.94"
2. Check car detector output ở các frame đó
3. Verify ground truth có đúng không
4. Nếu ground truth sai → tìm car_id đúng
5. Nếu tracking sai → tune tracking parameters

**Tools:**
- Xem `output/test_session2_full.mp4`
- Check `car_candidates.csv` để tìm pattern "29F"
- Có thể cần chạy lại với `--car-conf` thấp hơn

### Ưu tiên 2: Cài đặt PaddleOCR

**Mục tiêu:** Thêm OCR backend mạnh hơn cho crop nhỏ

**Các bước:**
1. Thử cài đặt lại PaddleOCR (có thể cần nhiều thời gian)
2. Hoặc thử alternative: TrOCR, EasyOCR, CRNN
3. Implement fallback logic cho crop < 40px
4. Test với car#3 hoặc xe khác có crop nhỏ

### Ưu tiên 3: Validate motorcycle reconstruction

**Mục tiêu:** Test logic với xe thực tế

**Các bước:**
1. Tìm xe khác trong video có biển 2 dòng rõ
2. Check `car_candidates.csv` để tìm candidates có row_texts 2 dòng
3. Validate reconstruction có đúng không
4. Tune scoring weights nếu cần

### Ưu tiên 4: Improve tracking robustness

**Mục tiêu:** Giảm tracking loss

**Các bước:**
1. Tăng `max_frames_missing` trong CarTracker
2. Giảm `car-conf` threshold
3. Thử DeepSORT thay vì centroid tracking
4. Test với video để đảm bảo không bị mất xe

## Cách chạy hiện tại

```bash
# Full video với session 2 improvements
python src/inference.py \
  --output output/test_session2_full.mp4 \
  --det-conf 0.15 \
  --ocr-conf 0.10 \
  --car-conf 0.25 \
  --debug-limit 1000

# Test ngắn
python src/inference.py \
  --output output/test_session3.mp4 \
  --det-conf 0.15 \
  --ocr-conf 0.10 \
  --car-conf 0.25 \
  --max-frames 100
```

## Ground truth hiện tại

```python
GROUND_TRUTH = {
    3: "29F02794",  # car#3 = 29F-027.94 (CHƯA VERIFY!)
}
```

**⚠️ WARNING:** Ground truth này chưa được verify. Car#3 chỉ track 1 frame với OCR hoàn toàn sai.

## Tracking parameters hiện tại

```python
# In main()
car_tracker = CarTracker(max_frames_missing=60)

# In parse_args()
--car-conf 0.25  # Car detector confidence
--det-conf 0.15  # Plate detector confidence
--ocr-conf 0.10  # OCR confidence
```

## Metrics cần theo dõi

1. **Tracking stability:**
   - Số frame mỗi car được track
   - Số xe bị mất giữa chừng

2. **OCR accuracy:**
   - Similarity với ground truth (khi có)
   - Số candidates pass VN plate regex

3. **Reconstruction quality:**
   - Số biển 2 dòng được reconstruct đúng
   - Scoring distribution

## Ghi chú quan trọng

1. **Car#3 là mystery:**
   - Chỉ 1 frame
   - OCR hoàn toàn sai
   - Cần verify ground truth trước khi optimize tiếp

2. **PaddleOCR pending:**
   - Cài đặt bị timeout
   - Cần thử lại hoặc dùng alternative

3. **Motorcycle reconstruction chưa test:**
   - Logic đã implement
   - Nhưng chưa có data để validate

4. **Logging system hoàn chỉnh:**
   - `car_candidates.csv` rất hữu ích
   - Dễ dàng trace decision path

## Next session priorities

1. **CRITICAL:** Verify car#3 ground truth
2. **HIGH:** Cài đặt OCR backend mạnh hơn
3. **MEDIUM:** Test motorcycle reconstruction
4. **LOW:** Tune tracking parameters

Good luck session 3! 🚀
