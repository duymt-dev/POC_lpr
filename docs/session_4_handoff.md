# Session 4 Handoff

> Cập nhật: 2026-04-13

## Tóm tắt session 3

Session 3 đã:
- ✅ Fix FSRCNN super-resolution (cài opencv-contrib-python)
- ✅ Chạy inference với FSRCNN enabled
- ✅ Phát hiện car#3 chỉ track 1 frame (frame 619)
- ✅ Xác nhận ground truth có vấn đề (không tìm thấy "29F" hoặc "34A" trong video)

## Session 4: Vehicle Text Fallback Implementation

### Mục tiêu

Implement fallback mechanism để đọc text lớn trên thân xe (xe khách, xe tải, xe dịch vụ) khi biển số quá nhỏ hoặc không đọc được.

### Ý tưởng chính

Nhiều xe thương mại in biển số hoặc mã số bằng chữ rất lớn trên:
- Kính trước
- Thân xe
- Đuôi xe
- Hông xe

Text này thường lớn hơn plate thật nhiều lần và dễ OCR hơn khi plate quá nhỏ.

### Implementation đã hoàn thành

#### 1. Constants và Configuration

**File:** `src/inference.py` (lines 619-665)

```python
# OCR correction map for vehicle text
OCR_CORRECTIONS = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', ...
}

# Blacklist patterns
VEHICLE_TEXT_BLACKLIST = [
    r'^1900\d+$',      # Hotline
    r'^1800\d+$',      # Hotline
    r'^0[35789]\d{8,9}$',  # Mobile numbers
    ...
]

# Thresholds
VEHICLE_TEXT_MIN_CONF = 0.25
VEHICLE_TEXT_MIN_HEIGHT = 18
VEHICLE_TEXT_MIN_LENGTH = 4
VEHICLE_TEXT_MAX_LENGTH = 12
VEHICLE_TEXT_STRICT_MIN_FRAMES = 2
VEHICLE_TEXT_NEAR_MATCH_MIN_FRAMES = 4

# Fallback triggers
PLATE_SMALL_HEIGHT_THRESH = 28
PLATE_SMALL_WIDTH_THRESH = 70
PLATE_LOW_CONF_THRESH = 0.20
PLATE_MIN_FRAMES_FOR_FALLBACK = 3
```

#### 2. Core Functions

**`extract_vehicle_text_regions(car_bbox, frame_shape)`** - Line 814

Trích xuất 6 vùng từ car bbox:
- Full car bbox
- Upper band (20% from top) - kính trước
- Lower band (20% from bottom) - đuôi xe
- Left side (25% from left) - hông trái
- Right side (25% from right) - hông phải
- Center horizontal (40% middle) - thân xe giữa

**`is_vehicle_text_blacklisted(text)`** - Line 867

Lọc text không phải biển số:
- Số điện thoại (09xx, 03xx, 1900, 1800)
- Text quá dài
- Nhiều chữ liên tiếp
- Keywords: HOTLINE, SERVICE, TAXI, BUS, TRUCK, COM

**`score_vehicle_text_candidate(text, conf, frame_idx, car_id, car_tracker)`** - Line 895

Chấm điểm candidate dựa trên:
- Length score (6-9 chars ideal)
- Prefix score (2 digits at start)
- Character composition (letter at pos 2-3, mostly digits after)
- OCR confidence
- Stability score (lặp lại qua nhiều frame)
- Pattern similarity với VN plate

#### 3. CarTracker Updates

**Thêm vào `__init__`** - Line 1015-1026

```python
self.cars[best_id] = {
    "text_history": [],
    "vehicle_text_history": [],  # NEW
    "last_bbox": bbox,
    ...
    "best_vehicle_text": "",     # NEW
    "best_vehicle_conf": 0.0,    # NEW
}
```

**Method mới: `get_best_vehicle_text(car_id)`** - Line 1177

Logic:
- Lấy vehicle_text_history
- Lọc blacklisted candidates
- Sort by score
- Apply temporal consistency bonus
- Return best candidate với confidence

#### 4. Main Pipeline Integration

**Fallback trigger** - Line 1910-1916

```python
plate_is_weak = (
    bh < PLATE_SMALL_HEIGHT_THRESH or
    bw < PLATE_SMALL_WIDTH_THRESH or
    frame_conf < PLATE_LOW_CONF_THRESH or
    not (is_valid_vn_plate(frame_norm) or is_likely_vn_plate(frame_norm))
)
```

**Vehicle text extraction** - Line 1918-2048

Khi `plate_is_weak`:
1. Extract vehicle regions từ car bbox
2. Enhance mỗi region
3. Run OCR với lower confidence (0.02)
4. Score candidates
5. Store vào `vehicle_text_history`
6. Track best candidate

**Display logic** - Line 2064-2092

```python
# Determine final display
if vehicle_text and plate_is_weak:
    best_vehicle_text, best_vehicle_conf = car_tracker.get_best_vehicle_text(car_id)
    if best_vehicle_text and best_vehicle_conf > 0.3:
        final_text = f"VEHICLE: {best_vehicle_text}"
        final_color = (255, 255, 0)  # Cyan
```

**Panel display** - Line 2143-2162

```python
for car_id, plate_text, confidence, frame_count, source in car_plates[:6]:
    # Display with source label
    source_label = f" [{source.upper()}]"  # [PLATE] or [VEHICLE]
    info_text = f"{label}{source_label} conf:{confidence:.2f}"
```

### Decision Policy Implementation

#### Tier System

**Strict candidates:**
- Pass regex VN đầy đủ
- 2 ký tự đầu là số
- Có ít nhất 1 chữ cái
- OCR conf >= 0.35 (vehicle) hoặc 0.25 (plate)
- Lặp >= 2 frames

**Near-match candidates:**
- Length 6-10
- 2 ký tự đầu gần số (có thể sửa bằng OCR map)
- Có ít nhất 1 chữ cái
- Lỗi OCR <= 2 ký tự
- OCR conf >= 0.28
- Lặp >= 4 frames

**Reject:**
- Length < 6 hoặc > 10
- Toàn số và dài >= 8
- Bắt đầu bằng hotline/phone patterns
- Chỉ 1 frame và conf < 0.35

#### Source Priority

1. `plate strict` (weight 1.00)
2. `plate near_match` (weight 0.80)
3. `vehicle_text strict` (weight 0.72)
4. `vehicle_text near_match` (weight 0.55)

#### Override Rules

Vehicle text chỉ override plate khi:
1. Plate không phải strict
2. Vehicle text score cao hơn >= 0.18
3. Vehicle text lặp >= 4 frames
4. Plate crop nhỏ (h < 28 hoặc w < 70)

### Files Changed

**Modified:**
- `src/inference.py` (+1165 lines, -341 lines)

**Key changes:**
- Added 3 new functions for vehicle text
- Extended CarTracker with vehicle_text_history
- Added fallback logic in main pipeline
- Updated display to show source ([PLATE] vs [VEHICLE])

### Testing Results

**Test run:** 200 frames

```bash
python src/inference.py \
  --output output/test_vehicle_text.mp4 \
  --det-conf 0.15 \
  --ocr-conf 0.10 \
  --car-conf 0.25 \
  --max-frames 200
```

**Status:** ✅ Code chạy thành công, không có lỗi

**Output:**
- `output/test_vehicle_text.mp4` - Video với vehicle text fallback
- Detected 1 active car trong 200 frames đầu

### Post-Commit Runtime Fixes

Sau commit `0917cfe`, đã phát hiện và sửa thêm một số lỗi runtime khi chạy dài hơn:

1. Fixed indentation issues trong `enhance_plate_crop()`
2. Fixed ROI slice types trong `extract_vehicle_text_regions()` bằng cách ép toàn bộ coords về `int`
3. Fixed `vehicle_text_history` tuple unpack trong `score_vehicle_text_candidate()`
4. Fixed thứ tự khởi tạo `box_color` trước khi tính `final_color`
5. Added `log_vehicle_text_candidate()` để log riêng vehicle fallback path

### Verification Sau Runtime Fix

**Short run 300 frames:**
- `output/test_vehicle_text_retry.mp4`
- ✅ chạy thành công

**Full run 1000 frames:**
- `output/test_vehicle_text_full_retry.mp4`
- ✅ chạy thành công
- không còn crash tại đoạn vehicle fallback

**Vehicle text logging:**
- Đã thêm `debug_crops/vehicle_text_candidates.csv`
- Trong 300 frames đầu chưa có candidate vehicle-text nào đủ điều kiện để được log
- Điều này cho thấy fallback path hiện chưa mang lại evidence hữu ích trên đoạn video ngắn đầu tiên, cần test ở các frame có xe lớn hơn hoặc video khác

### Improvements vs Session 3

1. **FSRCNN working:** opencv-contrib-python installed
2. **Vehicle text fallback:** Hoàn toàn mới
3. **Blacklist system:** Lọc phone numbers và text rác
4. **Temporal voting:** Vehicle text cần lặp nhiều frame mới tin
5. **Source attribution:** Panel hiển thị [PLATE] hoặc [VEHICLE]

### Known Limitations

1. **Chưa test với xe thật có text lớn:**
   - Video hiện tại có ít xe khách/xe tải
   - Cần test với video khác để validate

2. **Vehicle text có thể false positive:**
   - Dù có blacklist, vẫn có thể đọc nhầm text quảng cáo
   - Temporal voting giúp giảm nhưng không loại bỏ hoàn toàn

3. **Performance impact:**
   - Mỗi plate yếu trigger 6 ROI extractions
   - Mỗi ROI chạy OCR riêng
   - Có thể chậm hơn khi nhiều xe có plate nhỏ

4. **Ground truth vẫn chưa verify:**
   - Car#3 và Car#5 ground truth có vẻ sai
   - Cần verify bằng cách xem video thủ công

### Metrics cần theo dõi

1. **Fallback usage rate:**
   - Bao nhiêu % xe dùng vehicle text thay vì plate
   - Có phải chỉ xe có plate nhỏ không

2. **False positive rate:**
   - Vehicle text có match với ground truth không
   - Có đọc nhầm hotline/text rác không

3. **Performance:**
   - Latency tăng bao nhiêu khi bật fallback
   - OCR ms per frame

4. **Temporal consistency:**
   - Vehicle text có ổn định qua frames không
   - Có nhảy lung tung không

### Đề xuất cho session tiếp theo

#### Ưu tiên 1: Validate vehicle text với video thật

**Mục tiêu:** Test với video có xe khách/xe tải có text lớn

**Các bước:**
1. Tìm video có xe khách hoặc xe tải rõ
2. Chạy inference với vehicle text fallback
3. Check `debug_crops/car_candidates.csv` để xem vehicle text
4. Verify vehicle text có đúng với biển số thật không

#### Ưu tiên 2: Tune thresholds

**Mục tiêu:** Optimize precision/recall balance

**Các bước:**
1. Analyze false positives trong output
2. Adjust blacklist patterns nếu cần
3. Tune min_frames thresholds
4. Tune confidence thresholds

#### Ưu tiên 3: Add vehicle text logging

**Mục tiêu:** Debug và analyze vehicle text candidates

**Các bước:**
1. Tạo `debug_crops/vehicle_text_candidates.csv`
2. Log tất cả vehicle text detections
3. Include: car_id, frame, region, raw_text, normalized, score
4. Dễ dàng analyze false positives

#### Ưu tiên 4: Verify ground truth

**Mục tiêu:** Fix ground truth cho car#3 và car#5

**Các bước:**
1. Xem video thủ công
2. Tìm xe có biển "29F-027.94" và "34A-592.73"
3. Note frame numbers
4. Update `docs/manual_correction.txt` với car_id đúng

### Cách chạy hiện tại

```bash
# Test ngắn với vehicle text fallback
python src/inference.py \
  --output output/test_vehicle_text.mp4 \
  --det-conf 0.15 \
  --ocr-conf 0.10 \
  --car-conf 0.25 \
  --max-frames 200

# Full video
python src/inference.py \
  --output output/test_vehicle_full.mp4 \
  --det-conf 0.15 \
  --ocr-conf 0.10 \
  --car-conf 0.25
```

### Git Commit

```
commit 0917cfe
Author: DuyMT
Date: 2026-04-13

Add vehicle text fallback for small/unreadable license plates

- Implement vehicle-text extraction as fallback when plate OCR fails
- Add blacklist patterns for phone numbers and non-plate text
- Add scoring system for vehicle text candidates with temporal voting
- Extract text from multiple vehicle regions (windshield, body, rear)
- Display vehicle text with [VEHICLE] label when used as fallback
- Add thresholds: plate_h<28px, plate_w<70px, conf<0.20 trigger fallback
- Fix FSRCNN support with opencv-contrib-python
- Add vehicle_text_history tracking per car_id
```

### Technical Debt

1. **No vehicle text logging yet:**
   - Cần thêm CSV log riêng cho vehicle text
   - Giống `car_candidates.csv` nhưng cho vehicle text

2. **No performance profiling:**
   - Chưa đo chính xác impact của vehicle text extraction
   - Cần add timing metrics

3. **Hardcoded ROI percentages:**
   - Upper 20%, lower 20%, etc. là hardcoded
   - Có thể cần tune theo loại xe

4. **No vehicle type detection:**
   - Hiện tại apply cho mọi xe
   - Có thể optimize bằng cách chỉ bật cho truck/bus

### Ghi chú quan trọng

1. **Vehicle text là fallback, không phải primary:**
   - Chỉ dùng khi plate OCR yếu
   - Plate OCR vẫn được ưu tiên

2. **Temporal voting rất quan trọng:**
   - Vehicle text cần lặp nhiều frame
   - Giúp lọc false positives

3. **Blacklist cần maintain:**
   - Có thể cần thêm patterns mới
   - Tùy theo loại text gặp trong thực tế

4. **Source attribution giúp debug:**
   - Panel hiển thị [PLATE] hoặc [VEHICLE]
   - Dễ dàng biết kết quả từ đâu

Good luck session 5! 🚀
