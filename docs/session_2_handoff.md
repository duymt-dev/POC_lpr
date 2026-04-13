# Session 2 Handoff

> Cap nhat: 2026-04-12

## Muc tieu du an

Xay dung he thong LPR (License Plate Recognition) cho bien so xe Viet Nam, toi uu de chay tren Raspberry Pi 5, dau vao la video test va dau ra la video co:

- bbox tren xe / bien so
- panel ben phai hien thi bien so dang track duoc
- tracking theo `car_id`
- co gang tai tao bien so tu lich su OCR cua tung xe

## Trang thai hien tai

He thong da co the:

- detect xe bang model COCO (`yolov5n.pt`)
- detect bien so bang model `LP_detector_nano_61.pt`
- OCR ky tu bang model `LP_ocr_nano_62.pt`
- track xe theo `car_id`
- gan plate vao `car_id`
- xuat video co panel ben phai
- log debug crop loi vao `debug_crops/summary.csv`

Nhung he thong **chua** tai tao dung bien so that trong cac crop nho / mo.

## Ground truth da biet

- `car#3 = 29F-027.94`
- `car#5` neu dung thi format cung phai tuong tu, kieu `XXA-123.45`

Day la manh moi quan trong cho session tiep theo.

## File quan trong

- `src/inference.py`
  - file chinh cua pipeline
- `docs/plan.md`
  - ke hoach tong the
- `docs/model_info.md`
  - thong tin model
- `docs/results.md`
  - tong hop ket qua
- `docs/results_additional_tests.md`
  - ket qua cac test confidence / enhancement
- `debug_crops/summary.csv`
  - bang debug moi nhat de doi chieu theo frame va `car_id`

## Models dang dung

### 1. Plate detector
- File: `models/LP_detector_nano_61.pt`
- Dinh dang: legacy YOLOv5 checkpoint
- Chuc nang: detect bbox bien so

### 2. OCR detector
- File: `models/LP_ocr_nano_62.pt`
- Dinh dang: legacy YOLOv5 checkpoint
- Chuc nang: detect tung ky tu rieng le

### 3. Car detector
- File: `vendor/yolov5/yolov5n.pt`
- Dung class COCO:
  - `2` car
  - `3` motorcycle
  - `5` bus
  - `7` truck

### 4. Super-resolution
- File: `models/FSRCNN_x2.pb`
- Dung qua `cv2.dnn_superres`

## Kien truc hien tai

```text
Frame
  -> Car detector (YOLOv5n COCO)
  -> Track car bbox -> car_id
  -> Plate detector (LP_detector)
  -> Match plate bbox vao car_id gan nhat
  -> Crop bien so tu anh goc
  -> Enhance crop (FSRCNN / CLAHE / sharpen / threshold tuy truong hop)
  -> OCR ky tu (LP_ocr)
  -> Tao frame-level candidate
  -> Dua vao lich su cua car_id
  -> Chon best plate cho tung car_id
  -> Ve bbox + panel ben phai
```

## Cac ky thuat da thu

### Da lam

1. YOLOv5 compatibility
- Clone `vendor/yolov5` de load checkpoint cu

2. Enhancement
- FSRCNN x2
- Lanczos upscale
- CLAHE
- sharpen
- detailEnhance
- threshold / non-threshold path

3. OCR candidate generation
- `decode_plate_text()`
- `group_character_rows()`
- `select_frame_plate_candidate()`
- xu ly bien 2 dong co ban

4. Decision logic
- vote exact string
- similarity clustering
- streak-based voting
- best-frame scoring
- per-position vote trong best streak
- province-code constrained decoding
- position-aware correction

5. Tracking
- luc dau track bang plate bbox
- sau do doi sang **car detector rieng** + `car_id`
- ket qua tracking tot hon ro ret

### Da thu nhung khong hieu qua / da bo

1. EasyOCR fallback
- khong tot hon YOLO OCR tren crop nho
- da bo

2. OCR fallback truc tiep tu ROI duoi car bbox
- gay hallucination rat manh
- da bo

## Tinh hinh hien tai theo danh gia cuoi

### Diem manh

- `car_id` tracking on dinh hon nhieu
- debug crop da giam rat manh
- false positive giam
- decision layer bot loan hon

### Diem yeu

- OCR frame-level tren crop nho van qua yeu
- he thong dang co xu huong "on dinh hoa dap an sai"
- chua recover duoc `car#3 = 29F-027.94`

### Lan danh gia cuoi

Build test cuoi da chay:

- `output/test_v14_final_eval.mp4`

Debug moi nhat:

- `debug_crops/summary.csv`
- chi con 1 record low-confidence:
  - `car_id = 3`
  - `frame = 619`
  - `raw_text = 8540`
  - `formatted_text = 8540`

Ket luan cuoi session:

- tracking: **co cai thien that**
- reconstruction: **co on dinh hon**
- do chinh xac plate cuoi: **chua dat**

## Van de ky thuat cot loi con lai

1. Crop plate qua nho
- nhieu crop chi cao ~20-33 px
- OCR char-level rat de nham

2. OCR model hien tai detect tung ky tu
- rat nhay cam voi blur / low-res
- khi ky tu qua nho, bbox ky tu gan nhu khong con y nghia

3. Decision logic da gan het gioi han
- da tune nhieu tang reconstruction
- nhung dau vao frame-level van khong du sach

## Huong de xuat cho session 2

### Uu tien 1: ghi log / dashboard theo car_id chi tiet hon

Can them file debug chi tiet hon, vi du:

- `debug_crops/car_candidates.csv`

Moi dong nen co:

- `car_id`
- `frame`
- `raw_text`
- `row_texts`
- `normalized_candidate`
- `formatted_candidate`
- `plate_score`
- `ocr_conf`
- `crop_w`
- `crop_h`
- `best_raw_now`
- `best_formatted_now`

Muc tieu: xem ro theo tung `car_id`, tung frame, candidate nao dan den quyet dinh cuoi.

### Uu tien 2: model / backend OCR khac cho crop nho

Vi YOLO char-level dang la bottleneck, session 2 nen uu tien:

- fallback OCR backend khac chi cho crop nho
  - PaddleOCR / CRNN / recognizer sequence model
- hoac retrain OCR model voi data bien nho, blur, 2 dong

### Uu tien 3: motorcycle-specific reconstruction

Can rule rieng cho bien xe may 2 dong, thay vi co gang ep ve chuoi 1 dong qua som.

Y tuong:

- dong 1 uu tien 2 so dau (ma tinh)
- dong 2 uu tien series + nhom so
- danh gia candidate theo template 2 dong rieng

### Uu tien 4: xac dinh car#3 bang frame tot nhat

Da biet ground truth `29F-027.94`, nen session 2 co the:

- tim frame nao cua `car#3` co crop lon nhat
- luu rieng cac crop cua `car#3`
- so sanh candidate qua tung frame
- tune reconstruction dua tren 1 xe cu the truoc, roi moi tong quat hoa

## Cach chay hien tai

```bash
python src/inference.py --output output/test_v14_final_eval.mp4 --det-conf 0.15 --ocr-conf 0.10 --car-conf 0.25 --debug-limit 300
```

## Trang thai git

Da tao commit local va commit code/docs thanh cong:

- commit: `c2dc6cc`
- message: `add LPR inference pipeline and tracking docs`

Da **khong** commit:

- `debug_crops/`
- `output/`
- `video test/`
- cac binary / lock file

Push len GitHub chua thanh cong vi van de xac thuc / token.

## Ghi chu quan trong cho session 2

1. Khong quay lai huong ROI fallback tu `car bbox` nua
- da gay hallucination rat nhieu

2. Khong uu tien tiep cac rule vote phuc tap neu khong co them thong tin frame-level
- decision layer da duoc tune kha nhieu

3. Neu muon tiep tuc toi uu, nen uu tien:
- log theo `car_id`
- frame tot nhat cua `car#3`
- OCR backend / model cho crop nho

4. Ground truth da biet:
- `car#3 = 29F-027.94`
- day la anchor cuc ky quan trong de validate moi thay doi tiep theo
