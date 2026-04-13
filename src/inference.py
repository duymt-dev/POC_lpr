from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
YOLOV5_DIR = ROOT / "vendor" / "yolov5"
if str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vietnamese LPR video inference")
    parser.add_argument(
        "--video",
        default=str(
            ROOT / "video test" / "pwf2amf0gwaaaz1wsl18ywyayxuoyxmaap1e6w1.mp4"
        ),
        help="Input video path",
    )
    parser.add_argument(
        "--det-model",
        default=str(ROOT / "models" / "LP_detector_nano_61.pt"),
        help="Plate detector checkpoint",
    )
    parser.add_argument(
        "--ocr-model",
        default=str(ROOT / "models" / "LP_ocr_nano_62.pt"),
        help="OCR checkpoint",
    )
    parser.add_argument(
        "--car-model",
        default=str(ROOT / "vendor" / "yolov5" / "yolov5n.pt"),
        help="Car detector checkpoint (COCO pretrained)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "output" / "result.mp4"),
        help="Output video path",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.35,
        help="Plate detector confidence threshold",
    )
    parser.add_argument(
        "--car-conf", type=float, default=0.25, help="Car detector confidence threshold"
    )
    parser.add_argument(
        "--ocr-conf", type=float, default=0.25, help="OCR confidence threshold"
    )
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="CUDA device or cpu")
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Process only N frames, 0 = all"
    )
    parser.add_argument(
        "--panel-width", type=int, default=420, help="Right-side panel width"
    )
    parser.add_argument(
        "--padding", type=float, default=0.08, help="Padding ratio around plate bbox"
    )
    parser.add_argument(
        "--stable-window",
        type=int,
        default=12,
        help="Frames to keep for stable OCR voting",
    )
    parser.add_argument(
        "--debug-limit", type=int, default=150, help="Max debug crops to save"
    )
    return parser.parse_args()


def preprocess_bgr(
    image: np.ndarray, image_size: int, stride: int, device: torch.device
) -> tuple[torch.Tensor, tuple, tuple]:
    resized, ratio, pad = letterbox(
        image, new_shape=(image_size, image_size), auto=False, stride=stride
    )
    tensor = resized[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor)
    tensor = torch.from_numpy(tensor).to(device)
    tensor = tensor.float() / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor, ratio, pad


def run_yolov5(
    model: DetectMultiBackend,
    image: np.ndarray,
    image_size: int,
    conf_thres: float,
    iou_thres: float,
    classes=None,
):
    tensor, ratio, pad = preprocess_bgr(
        image, image_size, int(model.stride), model.device
    )
    pred = model(tensor)
    det = non_max_suppression(
        pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes, max_det=50
    )[0]
    if len(det):
        det[:, :4] = scale_boxes(
            tensor.shape[2:], det[:, :4], image.shape, ratio_pad=(ratio, pad)
        )
    return det


def pad_box(
    x1: int, y1: int, x2: int, y2: int, shape: tuple[int, int, int], pad_ratio: float
) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)


def enhance_plate_crop(crop: np.ndarray, min_height: int = 40) -> np.ndarray:
    h, w = crop.shape[:2]

    if h < min_height:
        scale = min_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    enhanced_bgr = cv2.detailEnhance(enhanced_bgr, sigma_s=10, sigma_r=0.15)

    return enhanced_bgr


def group_character_rows(items: list[dict]) -> list[list[dict]]:
    if not items:
        return []

    items = sorted(items, key=lambda item: item["cy"])
    rows: list[list[dict]] = []
    for item in items:
        placed = False
        for row in rows:
            avg_cy = sum(entry["cy"] for entry in row) / len(row)
            avg_h = sum(entry["h"] for entry in row) / len(row)
            if abs(item["cy"] - avg_cy) <= max(8.0, avg_h * 0.6):
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    rows.sort(key=lambda row: min(entry["cy"] for entry in row))
    for row in rows:
        row.sort(key=lambda item: item["x1"])
    return rows


def decode_plate_text(detections, names: list[str]) -> tuple[str, list[str]]:
    chars = []
    for *xyxy, conf, cls in detections.tolist():
        x1, y1, x2, y2 = xyxy
        cls_id = int(cls)
        chars.append(
            {
                "char": names[cls_id],
                "conf": float(conf),
                "x1": float(x1),
                "cy": float((y1 + y2) / 2.0),
                "h": float(y2 - y1),
            }
        )

    rows = group_character_rows(chars)
    row_texts = ["".join(item["char"] for item in row) for row in rows]
    text = "".join(row_texts)
    return text, row_texts


def reconstruct_motorcycle_plate(row_texts: list[str]) -> str:
    """Motorcycle-specific reconstruction for 2-line plates.

    VN motorcycle format:
    - Line 1: Province code (2 digits)
    - Line 2: Series (1 letter) + Number group (4-5 digits)

    Example: 29F-027.94
    - Line 1: "29"
    - Line 2: "F02794"
    """
    if len(row_texts) < 2:
        return ""

    top = normalize_plate_text(row_texts[0])
    bottom = normalize_plate_text(row_texts[1])

    candidates = []

    # Strategy 1: Top = exactly 2 digits (province code)
    if len(top) == 2 and top.isdigit():
        candidates.append(top + bottom)

    # Strategy 2: Top has 2+ chars, take first 2 as province
    if len(top) >= 2:
        province = top[:2]
        # Force province to be digits
        province_chars = []
        for ch in province:
            if ch.isdigit():
                province_chars.append(ch)
            else:
                province_chars.append(FRONT_DIGIT_MAP.get(ch, ch))
        province = "".join(province_chars)

        # Rest goes to bottom
        rest = top[2:]
        candidates.append(province + rest + bottom)

    # Strategy 3: Top has 3+ chars, maybe series leaked to top
    if len(top) >= 3:
        province = top[:2]
        series_and_rest = top[2:]

        # Force province digits
        province_chars = []
        for ch in province:
            if ch.isdigit():
                province_chars.append(ch)
            else:
                province_chars.append(FRONT_DIGIT_MAP.get(ch, ch))
        province = "".join(province_chars)

        candidates.append(province + series_and_rest + bottom)

    # Normalize all candidates
    normalized = [normalize_vn_candidate(c) for c in candidates if c]

    if not normalized:
        return ""

    # Score by motorcycle plate pattern
    def score_motorcycle(text: str) -> float:
        score = 0.0
        n = len(text)

        # Motorcycle plates typically 7-8 chars (29F12345 or 29F1234)
        if 7 <= n <= 8:
            score += 5.0
        elif 6 <= n <= 9:
            score += 2.0

        # Province code (2 digits)
        if n >= 2 and text[:2].isdigit():
            score += 5.0
            if text[:2] in VALID_PROVINCE_CODES:
                score += 5.0

        # Series letter at position 3
        if n >= 3 and text[2].isalpha():
            score += 4.0

        # Rest should be digits (4-5 digits)
        if n >= 4:
            tail = text[3:]
            if tail and tail.isdigit():
                score += 6.0
                # Prefer 4-5 digit tail
                if 4 <= len(tail) <= 5:
                    score += 3.0

        return score

    return max(normalized, key=score_motorcycle)


def select_frame_plate_candidate(raw_text: str, row_texts: list[str]) -> str:
    """Build the best single-frame candidate.

    For motorcycle 2-line plates, use row-aware reconstruction instead of blindly
    flattening all characters.
    """
    candidates: list[str] = []

    raw_norm = normalize_vn_candidate(raw_text)
    if raw_norm:
        candidates.append(raw_norm)

    # Try motorcycle-specific reconstruction for 2-line plates
    if len(row_texts) >= 2:
        motorcycle_candidate = reconstruct_motorcycle_plate(row_texts)
        if motorcycle_candidate:
            candidates.append(motorcycle_candidate)

        # Original strategies as fallback
        top = normalize_plate_text(row_texts[0])
        bottom = normalize_plate_text(row_texts[1])

        # Common VN motorcycle layout: top row starts with province code.
        if len(top) >= 2:
            candidates.append(normalize_vn_candidate(top[:2] + bottom))
        candidates.append(normalize_vn_candidate(top + bottom))

        # Sometimes OCR leaks one extra char into the top row.
        if len(top) >= 3:
            candidates.append(normalize_vn_candidate(top[:2] + top[2:] + bottom))

    candidates = [c for c in candidates if c]
    if not candidates:
        return ""

    return max(candidates, key=plate_candidate_score)


def normalize_plate_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def format_plate_text(text: str) -> str:
    text = normalize_plate_text(text)
    if len(text) == 8:
        return f"{text[:3]}-{text[3:6]}.{text[6:]}"
    if len(text) == 9:
        return f"{text[:4]}-{text[4:7]}.{text[7:]}"
    if len(text) == 10:
        return f"{text[:4]}-{text[4:7]}.{text[7:]}"
    return text


def is_valid_vn_plate(text: str) -> bool:
    text = normalize_plate_text(text)
    patterns = [
        r"^\d{2}[A-Z]\d\d{5}$",  # 29A112345
        r"^\d{2}[A-Z]{2}\d{5}$",  # 29AB12345
        r"^\d{2}[A-Z]\d\d{4}$",  # 29A11234
        r"^\d{2}[A-Z]{2}\d{4}$",  # 29AB1234
        r"^\d{2}[A-Z]\d[A-Z0-9]\d{4}$",  # 29A1B234
    ]
    return any(re.fullmatch(pattern, text) for pattern in patterns)


def is_likely_vn_plate(text: str) -> bool:
    """Check if text looks like a VN plate (more lenient than is_valid_vn_plate)"""
    text = normalize_plate_text(text)

    # Must be 6-9 chars
    if not (6 <= len(text) <= 9):
        return False

    # Must start with 2 digits (province code)
    if len(text) < 2 or not text[:2].isdigit():
        return False

    # Province code must be valid
    if text[:2] not in VALID_PROVINCE_CODES:
        return False

    # Position 3 should be a letter (series)
    if len(text) >= 3 and not text[2].isalpha():
        return False

    # Rest should be mostly digits
    if len(text) > 3:
        tail = text[3:]
        digit_ratio = sum(c.isdigit() for c in tail) / len(tail)
        if digit_ratio < 0.6:  # At least 60% digits
            return False

    return True


def is_likely_plate(text: str) -> bool:
    text = normalize_plate_text(text)
    # Accept text with 6-10 alphanumeric chars starting with 2 digits
    if 6 <= len(text) <= 10 and text[:2].isdigit() and text.isalnum():
        return True
    return False


# OCR error correction map - chỉ sửa 1 chiều (char->digit/letter), không reverse
# Dựa trên thực tế debug: D bị nhầm với 0, U bị nhầm với 0, B bị nhầm với 8
OCR_CORRECTIONS = {
    "D": "0",  # D trông giống 0
    "U": "0",  # U trông giống 0
}
# Khi OCR ra chuỗi số mà có D/U thay vì 0 → sửa
# Không reverse 0->D hay 8->B vì sẽ làm hỏng số thật


def canonicalize_plate_text(text: str) -> str:
    text = normalize_plate_text(text)
    return "".join(OCR_CORRECTIONS.get(c, c) for c in text)


FRONT_DIGIT_MAP = {
    "A": "4",
    "V": "4",
    "Z": "2",
    "L": "1",
    "I": "1",
    "M": "1",
    "S": "5",
    "T": "7",
    "B": "8",
    "P": "9",
}

TAIL_DIGIT_MAP = {
    "D": "0",
    "U": "0",
    "O": "0",
    "Q": "0",
    "Z": "2",
    "S": "5",
    "B": "8",
}

VALID_PROVINCE_CODES = {
    "11",
    "12",
    "14",
    "15",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "43",
    "47",
    "48",
    "49",
    "50",
    "51",
    "52",
    "53",
    "54",
    "55",
    "56",
    "57",
    "58",
    "59",
    "60",
    "61",
    "62",
    "63",
    "64",
    "65",
    "66",
    "67",
    "68",
    "69",
    "70",
    "71",
    "72",
    "73",
    "74",
    "75",
    "76",
    "77",
    "78",
    "79",
    "80",
    "81",
    "82",
    "83",
    "84",
    "85",
    "86",
    "88",
    "89",
    "90",
    "92",
    "93",
    "94",
    "95",
    "97",
    "98",
    "99",
}


def normalize_vn_candidate(text: str) -> str:
    text = canonicalize_plate_text(text)
    if not text:
        return text

    chars = list(text)

    # Province code: first 2 chars should strongly prefer digits.
    for i in range(min(2, len(chars))):
        if not chars[i].isdigit():
            chars[i] = FRONT_DIGIT_MAP.get(chars[i], chars[i])

    # Tail should prefer digits more heavily.
    for i in range(3, len(chars)):
        if not chars[i].isdigit():
            chars[i] = TAIL_DIGIT_MAP.get(chars[i], chars[i])

    return "".join(chars)


def plate_candidate_score(text: str) -> float:
    text = normalize_vn_candidate(text)
    if not text:
        return 0.0

    score = 0.0
    n = len(text)

    # VN plate normalized length is usually 6-9 chars.
    if 6 <= n <= 9:
        score += 4.0
    elif 5 <= n <= 10:
        score += 2.0
    else:
        score -= 3.0

    # Strong preference for province code at the front.
    if n >= 2 and text[:2].isdigit():
        score += 5.0
        if text[:2] in VALID_PROVINCE_CODES:
            score += 4.0
    elif n >= 1 and text[0].isdigit():
        score += 1.0
    else:
        score -= 2.0

    # Third/fourth chars often contain series letters or alnum mix.
    if n >= 3 and text[2].isalpha():
        score += 2.0
    if n >= 4 and text[3].isalnum():
        score += 1.0

    # Tail is usually digit-heavy.
    tail = text[3:] if n > 3 else text
    if tail:
        digit_ratio = sum(ch.isdigit() for ch in tail) / len(tail)
        score += digit_ratio * 3.0

    return score


def plate_similarity(a: str, b: str) -> float:
    a = normalize_vn_candidate(a)
    b = normalize_vn_candidate(b)
    if not a or not b:
        return 0.0

    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0

    # Penalize very different lengths hard.
    if abs(len(a) - len(b)) > 2:
        return 0.0

    same = 0.0
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]:
            same += 1.0
        elif i < 2 and a[i].isdigit() and b[i].isdigit():
            same += 0.5

    # Prefix matters a lot for VN plates.
    prefix_bonus = 0.0
    if len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2]:
        prefix_bonus = 1.5

    return (same + prefix_bonus) / max_len


# Global FSRCNN super-resolution model (loaded once)
_FSRCNN_MODEL = None

# Ground truth for validation
GROUND_TRUTH = {
    3: "29F02794",  # car#3 = 29F-027.94
}

# OCR correction map for common misreads in vehicle text
OCR_CORRECTIONS = {
    "O": "0",
    "o": "0",
    "Q": "0",
    "I": "1",
    "l": "1",
    "|": "1",
    "Z": "2",
    "z": "2",
    "E": "3",
    "e": "3",
    "A": "4",
    "a": "4",
    "S": "5",
    "s": "5",
    "G": "6",
    "g": "6",
    "T": "7",
    "t": "7",
    "B": "8",
    "b": "8",
    "g": "9",
    "q": "9",
}

# Blacklist patterns for vehicle text (likely not license plates)
VEHICLE_TEXT_BLACKLIST = [
    r"^1900\d+$",  # Hotline
    r"^1800\d+$",  # Hotline
    r"^0[35789]\d{8,9}$",  # Vietnamese mobile numbers
    r"^0\d{9,10}$",  # Vietnamese phone numbers
    r"^\d{10,}$",  # Long numeric sequences
    r"^[A-Z]{3,}$",  # Long letter-only sequences
]

# Vehicle text extraction parameters
VEHICLE_TEXT_MIN_CONF = 0.25
VEHICLE_TEXT_MIN_HEIGHT = 18
VEHICLE_TEXT_MIN_LENGTH = 4
VEHICLE_TEXT_MAX_LENGTH = 12
VEHICLE_TEXT_STRICT_MIN_FRAMES = 2
VEHICLE_TEXT_NEAR_MATCH_MIN_FRAMES = 4

# Fallback trigger thresholds
PLATE_SMALL_HEIGHT_THRESH = 28
PLATE_SMALL_WIDTH_THRESH = 70
PLATE_LOW_CONF_THRESH = 0.20
PLATE_MIN_FRAMES_FOR_FALLBACK = 3


def load_manual_corrections(correction_file: Path) -> dict[int, str]:
    """Load manual corrections from file

    Format: car_id=plate_number
    Example:
        3=29F-027.94
        5=34A-592.73

    ⚠️ NOTE: car_id is video-specific! Each video assigns car_id differently.
    Run inference once, check panel for car_ids, then create correction file.
    """
    corrections = {}
    if not correction_file.exists():
        return corrections

    with open(correction_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            parts = line.split("=", 1)
            if len(parts) != 2:
                continue

            try:
                car_id = int(parts[0].strip())
                plate = parts[1].strip()
                if plate:
                    corrections[car_id] = plate
            except ValueError:
                continue

    return corrections


def get_fsrcnn_model():
    global _FSRCNN_MODEL
    if _FSRCNN_MODEL is None:
        _FSRCNN_MODEL = cv2.dnn_superres.DnnSuperResImpl_create()
        model_path = str(ROOT / "models" / "FSRCNN_x2.pb")
        _FSRCNN_MODEL.readModel(model_path)
        _FSRCNN_MODEL.setModel("fsrcnn", 2)
    return _FSRCNN_MODEL


def save_car3_crop(
    debug_dir: Path,
    frame_idx: int,
    crop: np.ndarray,
    raw_text: str,
    normalized: str,
    formatted: str,
    crop_w: int,
    crop_h: int,
) -> None:
    """Save all crops for car#3 to analyze best frames"""
    car3_dir = debug_dir / "car3_analysis"
    car3_dir.mkdir(parents=True, exist_ok=True)

    # Calculate similarity to ground truth
    gt = GROUND_TRUTH.get(3, "")
    similarity = plate_similarity(normalized, gt) if gt else 0.0

    # Save crop with detailed filename
    safe_norm = normalize_plate_text(normalized) or "EMPTY"
    safe_fmt = normalize_plate_text(formatted) or "INVALID"
    filename = f"f{frame_idx:05d}_w{crop_w}_h{crop_h}_sim{similarity:.2f}_{safe_norm}_{safe_fmt}.jpg"
    cv2.imwrite(str(car3_dir / filename), crop)

    # Log to CSV
    csv_path = car3_dir / "car3_frames.csv"
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        if not file_exists:
            f.write(
                "frame,crop_w,crop_h,raw_text,normalized,formatted,similarity_to_gt\n"
            )
        f.write(
            f'{frame_idx},{crop_w},{crop_h},"{raw_text}","{normalized}","{formatted}",{similarity:.3f}\n'
        )


def enhance_plate_crop(
    crop: np.ndarray,
    min_height: int = 64,
    use_fsrcnn: bool = True,
    apply_threshold: bool = True,
) -> np.ndarray:
    h, w = crop.shape[:2]
    original_h = h

    # Upscale if too small
    if h < min_height:
        # Use FSRCNN for small crops (better quality than interpolation)
        if use_fsrcnn and h < 50:
            try:
                sr_model = get_fsrcnn_model()
                crop = sr_model.upsample(crop)  # x2 upscale with neural network
                h, w = crop.shape[:2]
            except Exception as e:
                # Fallback to LANCZOS4 if FSRCNN fails
                print(f"FSRCNN failed: {e}, falling back to LANCZOS4")
                use_fsrcnn = False

        # If still too small after FSRCNN or FSRCNN not used, use LANCZOS4
        if h < min_height:
            scale = min_height / h
            new_w = int(w * scale)
            new_h = int(h * scale)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Convert to grayscale for processing
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Sharpening using unsharp mask
    blurred = cv2.GaussianBlur(enhanced_gray, (0, 0), 3.0)
    sharpened = cv2.addWeighted(enhanced_gray, 1.5, blurred, -0.5, 0)

    # Adaptive threshold for very small crops can help, but it can also destroy weak strokes.
    if apply_threshold and original_h < 25:
        sharpened = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    # Convert back to BGR
    enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    # Final detail enhancement
    enhanced_bgr = cv2.detailEnhance(enhanced_bgr, sigma_s=10, sigma_r=0.15)

    return enhanced_bgr


def extract_vehicle_text_regions(
    car_bbox: List[int], frame_shape: Tuple[int, int, int]
) -> List[Tuple[int, int, int, int]]:
    """Extract regions from car bbox where license plate text might appear on vehicle body.

    Returns list of (x1, y1, x2, y2) bounding boxes for regions to check.
    """
    x1, y1, x2, y2 = car_bbox
    w, h = x2 - x1, y2 - y1

    regions = []

    # Full car bbox (sometimes text is on the body)
    regions.append((int(x1), int(y1), int(x2), int(y2)))

    # Upper band (front windshield area) - 20% from top
    upper_h = int(h * 0.2)
    regions.append((int(x1), int(y1), int(x2), int(y1 + upper_h)))

    # Lower band (rear/windshield area) - 20% from bottom
    lower_h = int(h * 0.2)
    regions.append((int(x1), int(y2 - lower_h), int(x2), int(y2)))

    # Left side band (driver side) - 25% from left
    left_w = int(w * 0.25)
    regions.append((int(x1), int(y1), int(x1 + left_w), int(y2)))

    # Right side band (passenger side) - 25% from right
    right_w = int(w * 0.25)
    regions.append((int(x2 - right_w), int(y1), int(x2), int(y2)))

    # Center horizontal band (middle of vehicle) - 40% height in middle
    center_h = int(h * 0.4)
    center_y1 = y1 + (h - center_h) // 2
    center_y2 = center_y1 + center_h
    regions.append((int(x1), int(center_y1), int(x2), int(center_y2)))

    # Filter out invalid regions
    valid_regions = []
    for rx1, ry1, rx2, ry2 in regions:
        # Ensure coordinates are within frame bounds
        rx1 = max(0, int(rx1))
        ry1 = max(0, int(ry1))
        rx2 = min(frame_shape[1], int(rx2))
        ry2 = min(frame_shape[0], int(ry2))

        # Ensure minimum size
        if rx2 - rx1 >= 10 and ry2 - ry1 >= 10:
            valid_regions.append((rx1, ry1, rx2, ry2))

    return valid_regions


def is_vehicle_text_blacklisted(text: str) -> bool:
    """Check if text matches blacklist patterns (likely not a license plate)."""
    if not text or len(text) < VEHICLE_TEXT_MIN_LENGTH:
        return True

    # Check against blacklist patterns
    for pattern in VEHICLE_TEXT_BLACKLIST:
        if re.match(pattern, text):
            return True

    # Additional heuristics
    # Too many consecutive letters
    if re.search(r"[A-Z]{4,}", text):
        return True

    # All same character
    if len(set(text)) == 1:
        return True

    # Contains obvious non-plate words
    non_plate_indicators = ["HOTLINE", "SERVICE", "TAXI", "BUS", "TRUCK", "COM"]
    text_upper = text.upper()
    for indicator in non_plate_indicators:
        if indicator in text_upper:
            return True

    return False


def score_vehicle_text_candidate(
    text: str, conf: float, frame_idx: int, car_id: int, car_tracker: "CarTracker"
) -> float:
    """Score a vehicle text candidate based on multiple factors."""
    if not text or conf < VEHICLE_TEXT_MIN_CONF:
        return 0.0

    # Normalize text
    normalized = normalize_vn_candidate(text)
    if not normalized:
        return 0.0

    # Length score
    length = len(normalized)
    length_score = 0.0
    if VEHICLE_TEXT_MIN_LENGTH <= length <= VEHICLE_TEXT_MAX_LENGTH:
        if 6 <= length <= 9:  # Ideal VN plate length
            length_score = 2.0
        else:
            length_score = 1.0
    else:
        return 0.0  # Length outside acceptable range

    # Prefix score (should start with 2 digits)
    prefix_score = 0.0
    if len(normalized) >= 2 and normalized[:2].isdigit():
        prefix_score = 2.0
        if normalized[:2] in VALID_PROVINCE_CODES:
            prefix_score += 1.0

    # Character composition score
    char_score = 0.0
    if len(normalized) >= 3:
        # Should have letter in position 2-3 (series)
        if normalized[2].isalpha():
            char_score += 1.5
        # Rest should be mostly digits
        if len(normalized) > 3:
            tail = normalized[3:]
            digit_ratio = sum(c.isdigit() for c in tail) / len(tail) if tail else 0
            char_score += digit_ratio * 1.5

    # OCR confidence score
    conf_score = conf * 2.0  # Scale to reasonable range

    # Stability score (based on historical consistency)
    stability_score = 0.0
    if car_id in car_tracker.cars:
        vehicle_history = car_tracker.cars[car_id].get("vehicle_text_history", [])
        # Count how many times this exact text appeared recently
        recent_matches = sum(1 for f, t, c in vehicle_history[-10:] if t == normalized)
        stability_score = min(recent_matches * 0.5, 2.0)  # Max 2.0 points

    # Blacklist penalty
    if is_vehicle_text_blacklisted(normalized):
        return 0.0

    # Pattern similarity score (how close to valid VN plate)
    pattern_score = plate_candidate_score(normalized) * 0.5  # Scale down

    # Total score
    total_score = (
        length_score * 0.25
        + prefix_score * 0.20
        + char_score * 0.20
        + conf_score * 0.15
        + stability_score * 0.10
        + pattern_score * 0.10
    )

    return total_score


class CarTracker:
    """Track cars using car detector bboxes.

    Strategy: mỗi frame OCR ra 1 chuỗi text → lưu vào text_history của car.
    Khi hiển thị → vote chuỗi phổ biến nhất, ưu tiên chuỗi dài hơn.
    Không cluster char pool nhiều frame (sai vì xe di chuyển → pos_x thay đổi).
    """

    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    def __init__(self, max_frames_missing: int = 45, history_window: int = 20):
        self.next_car_id = 1
        self.cars = {}
        self.max_distance = 150.0
        self.max_frames_missing = max_frames_missing
        self.history_window = history_window

    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 < x1 or y2 < y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / max(a1 + a2 - inter, 1e-6)

    def centroid(self, bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def match_car(self, bbox, frame_idx):
        best_id, best_score = None, 0.0
        cx, cy = self.centroid(bbox)
        for car_id, car in self.cars.items():
            if frame_idx - car["last_frame"] > self.max_frames_missing:
                continue
            iou = self.compute_iou(bbox, car["last_bbox"])
            pcx, pcy = self.centroid(car["last_bbox"])
            dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            score = iou * 0.7 + max(0.0, 1.0 - dist / self.max_distance) * 0.3
            if score > best_score and (iou > 0.05 or dist < self.max_distance):
                best_score = score
                best_id = car_id

        if best_id is None:
            best_id = self.next_car_id
            self.next_car_id += 1
            self.cars[best_id] = {
                "text_history": [],  # list of (frame_idx, raw_text, conf, crop_w, crop_h)
                "vehicle_text_history": [],  # list of (frame_idx, raw_text, normalized_text, score, region)
                "last_bbox": bbox,
                "last_frame": frame_idx,
                "first_frame": frame_idx,
                "frame_count": 0,
                "best_plate": "",
                "best_conf": 0.0,
                "best_vehicle_text": "",
                "best_vehicle_conf": 0.0,
            }

        self.cars[best_id]["last_bbox"] = bbox
        self.cars[best_id]["last_frame"] = frame_idx
        self.cars[best_id]["frame_count"] += 1
        return best_id

    def find_car_for_plate(self, plate_bbox, frame_idx):
        """Match plate bbox to nearest car by containment + distance."""
        px1, py1, px2, py2 = plate_bbox
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        best_id, best_score = None, 0.0

        for car_id, car in self.cars.items():
            if frame_idx - car["last_frame"] > 5:
                continue
            cx1, cy1, cx2, cy2 = car["last_bbox"]
            mx = (cx2 - cx1) * 0.2
            my = (cy2 - cy1) * 0.2
            inside = (cx1 - mx < pcx < cx2 + mx) and (cy1 - my < pcy < cy2 + my)
            iou = self.compute_iou(plate_bbox, [cx1, cy1, cx2, cy2])
            car_cx = (cx1 + cx2) / 2
            car_cy = (cy1 + cy2) / 2
            dist_s = max(
                0.0, 1.0 - ((pcx - car_cx) ** 2 + (pcy - car_cy) ** 2) ** 0.5 / 300.0
            )
            score = (1.5 if inside else 0.0) + iou * 0.5 + dist_s * 0.5
            if score > best_score:
                best_score = score
                best_id = car_id

        return best_id, best_score

    def add_frame_text(self, car_id, raw_text, conf, frame_idx, crop_w, crop_h):
        """Add per-frame OCR result (1 string per frame) to car history."""
        if car_id not in self.cars or not raw_text:
            return
        history = self.cars[car_id]["text_history"]
        history.append((frame_idx, raw_text, conf, crop_w, crop_h))
        # Keep only recent window
        if len(history) > self.history_window:
            self.cars[car_id]["text_history"] = history[-self.history_window :]

    def get_best_plate(self, car_id):
        """Choose best plate for a tracked car.

        Strategy:
        1. Filter candidates that look like VN plates
        2. Prefer larger crops with valid VN format
        3. Use voting for similar candidates
        """
        if car_id not in self.cars:
            return "", 0.0

        car = self.cars[car_id]
        history = car["text_history"]
        if not history:
            return "", 0.0

        # Build candidates with VN validation
        candidates = []
        for fr, txt, cf, cw, ch in history:
            can = normalize_vn_candidate(txt)
            if len(can) < 4:
                continue

            quality = plate_candidate_score(can)
            is_valid = is_valid_vn_plate(can)
            is_likely = is_likely_vn_plate(can)

            candidates.append(
                {
                    "frame": fr,
                    "text": can,
                    "conf": cf,
                    "quality": quality,
                    "crop_w": cw,
                    "crop_h": ch,
                    "is_valid": is_valid,
                    "is_likely": is_likely,
                }
            )

        if not candidates:
            return "", 0.0

        # Score each candidate
        def score_candidate(c):
            score = 0.0

            # Valid VN plate gets huge bonus
            if c["is_valid"]:
                score += 100.0
            elif c["is_likely"]:
                score += 50.0

            # Quality score
            score += c["quality"] * 5.0

            # Larger crop is better
            score += min(c["crop_h"], 60) * 0.5
            score += min(c["crop_w"], 120) * 0.2

            # Confidence
            score += c["conf"] * 10.0

            # Length preference (7-8 chars for VN plates)
            length = len(c["text"])
            if 7 <= length <= 8:
                score += 20.0
            elif 6 <= length <= 9:
                score += 10.0

            return score

        # Sort by score
        candidates.sort(key=score_candidate, reverse=True)

        # Get top candidate
        best = candidates[0]

        # If top candidate is valid, return it
        if best["is_valid"] or best["is_likely"]:
            return best["text"], best["conf"]

        # Otherwise, try voting among top candidates
        top_candidates = candidates[: min(5, len(candidates))]

        # Group by similarity
        groups = []
        for c in top_candidates:
            placed = False
            for group in groups:
                if plate_similarity(c["text"], group[0]["text"]) >= 0.6:
                    group.append(c)
                    placed = True
                    break
            if not placed:
                groups.append([c])

        # Pick largest group
        if groups:
            largest_group = max(groups, key=len)
            # Return best candidate from largest group
            best_in_group = max(largest_group, key=score_candidate)
            return best_in_group["text"], best_in_group["conf"]

        return best["text"], best["conf"]

    def get_best_vehicle_text(self, car_id):
        """Get the best vehicle text candidate for a tracked car."""
        if car_id not in self.cars:
            return "", 0.0

        car = self.cars[car_id]
        if "vehicle_text_history" not in car:
            return "", 0.0

        history = car["vehicle_text_history"]
        if not history:
            return "", 0.0

        # Build candidates
        candidates = []
        for fr, txt, norm_txt, score, region in history:
            # Skip if blacklisted
            if is_vehicle_text_blacklisted(norm_txt):
                continue

            candidates.append(
                {
                    "frame": fr,
                    "raw_text": txt,
                    "normalized_text": norm_txt,
                    "score": score,
                    "region": region,
                }
            )

        if not candidates:
            return "", 0.0

        # Sort by score (descending)
        candidates.sort(key=lambda c: c["score"], reverse=True)

        # Get top candidate
        best = candidates[0]

        # Apply temporal consistency bonus
        # Count how many times this exact text appeared recently
        recent_matches = sum(
            1 for f, t, n, s, r in history[-10:] if n == best["normalized_text"]
        )
        if recent_matches >= VEHICLE_TEXT_STRICT_MIN_FRAMES:
            # Strong temporal consistency
            return best["normalized_text"], min(best["score"] * 1.5, 1.0)
        elif recent_matches >= VEHICLE_TEXT_NEAR_MATCH_MIN_FRAMES:
            # Good temporal consistency
            return best["normalized_text"], best["score"]
        else:
            # Weak temporal consistency, still return but with lower confidence
            return best["normalized_text"], best["score"] * 0.5

    def update_best_plate(self, car_id, plate_text, conf):
        if car_id not in self.cars:
            return
        car = self.cars[car_id]
        cur_len = len(normalize_plate_text(car["best_plate"]))
        new_len = len(normalize_plate_text(plate_text))
        if new_len > cur_len or (new_len == cur_len and conf > car["best_conf"]):
            car["best_plate"] = plate_text
            car["best_conf"] = conf

    def cleanup_old_cars(self, current_frame):
        to_remove = [
            cid
            for cid, car in self.cars.items()
            if current_frame - car["last_frame"] > self.max_frames_missing
        ]
        for cid in to_remove:
            del self.cars[cid]

    def get_active_cars(self, current_frame, max_age=20):
        return [
            (cid, car)
            for cid, car in self.cars.items()
            if current_frame - car["last_frame"] <= max_age
        ]


def reconstruct_plate(char_pool):
    """Reconstruct plate from character pool.

    Strategy:
    - Char pool contains detections from many frames, each with pos_x in crop coords
    - Group by row (cy) first: xe máy has 2 rows, ô tô has 1 row
    - Within each row, cluster by pos_x with voting (highest conf wins)
    - Merge rows: row1 + row2 (xe máy) or single row (ô tô)
    """
    if not char_pool:
        return "", 0.0

    # Only use recent frames to avoid stale char positions when car moves
    if len(char_pool) > 0:
        max_frame = max(c["frame"] for c in char_pool)
        # Keep chars from last 20 frames only
        recent_pool = [c for c in char_pool if max_frame - c["frame"] <= 20]
        if len(recent_pool) >= 3:
            char_pool = recent_pool

    # Normalize pos_x to [0, 1] range to be scale-invariant
    if not char_pool:
        return "", 0.0

    xs = [c["pos_x"] for c in char_pool]
    cys = [c["cy"] for c in char_pool]
    x_min, x_max = min(xs), max(xs)
    x_range = max(x_max - x_min, 1.0)

    normalized = []
    for c in char_pool:
        normalized.append(
            {
                "char": c["char"],
                "conf": c["conf"],
                "pos_x_norm": (c["pos_x"] - x_min) / x_range,
                "cy": c["cy"],
                "frame": c["frame"],
            }
        )

    # Split into rows by cy (biển xe máy = 2 dòng)
    cy_vals = sorted(set(round(c["cy"] / 10) * 10 for c in normalized))
    if len(cy_vals) >= 2:
        cy_range = max(c["cy"] for c in normalized) - min(c["cy"] for c in normalized)
        if cy_range > 20:  # 2 distinct rows
            cy_mid = (
                min(c["cy"] for c in normalized) + max(c["cy"] for c in normalized)
            ) / 2
            row1 = [c for c in normalized if c["cy"] <= cy_mid]
            row2 = [c for c in normalized if c["cy"] > cy_mid]
        else:
            row1 = normalized
            row2 = []
    else:
        row1 = normalized
        row2 = []

    def cluster_row(chars, tol=0.12):
        """Cluster chars in a row by normalized pos_x, return sorted best chars"""
        if not chars:
            return [], 0.0
        chars_sorted = sorted(chars, key=lambda c: c["pos_x_norm"])
        clusters = []
        for ch in chars_sorted:
            placed = False
            for cluster in clusters:
                avg = sum(x["pos_x_norm"] for x in cluster) / len(cluster)
                if abs(ch["pos_x_norm"] - avg) < tol:
                    cluster.append(ch)
                    placed = True
                    break
            if not placed:
                clusters.append([ch])
        # Sort clusters by position
        clusters.sort(key=lambda cl: sum(x["pos_x_norm"] for x in cl) / len(cl))
        # Pick best char per cluster (highest confidence)
        result_chars = []
        total_conf = 0.0
        for cl in clusters:
            best = max(cl, key=lambda x: x["conf"])
            result_chars.append(best["char"])
            total_conf += best["conf"]
        avg_conf = total_conf / len(result_chars) if result_chars else 0.0
        return result_chars, avg_conf

    chars1, conf1 = cluster_row(row1)
    chars2, conf2 = cluster_row(row2)

    # Merge rows
    if chars1 and chars2:
        # xe máy: dòng 1 thường ngắn (2-3 ký tự), dòng 2 dài hơn
        all_chars = chars1 + chars2
        avg_conf = (conf1 * len(chars1) + conf2 * len(chars2)) / (
            len(chars1) + len(chars2)
        )
    else:
        all_chars = chars1 if chars1 else chars2
        avg_conf = conf1 if chars1 else conf2

    plate_text = "".join(all_chars)
    return plate_text, avg_conf


def smart_format_vn_plate(raw_text):
    """Smart formatting to match VN plate format.

    VN plate formats:
    - Ô tô 1 dòng:  29A-123.45  (8 chars: 2 số + 1-2 chữ + 4-5 số)
    - Xe máy 2 dòng: 29 / A1-123.45  (ghép thành: 29A1-123.45)
    """
    text = normalize_vn_candidate(raw_text)
    if len(text) < 5:
        return text, 0.0

    corrected = normalize_vn_candidate(text)

    def score_candidate(candidate):
        score = 0.0
        n = len(candidate)
        # Starts with 2 digits (tỉnh/thành phố)
        if n >= 2 and candidate[:2].isdigit():
            score += 4.0
        # Has letter after digits (series)
        if n >= 3 and candidate[2].isalpha():
            score += 3.0
        # Good length for VN plate (7-9 chars normalized)
        if 7 <= n <= 9:
            score += 2.0
        elif 6 <= n <= 10:
            score += 1.0
        # Ends mostly with digits
        tail = candidate[3:] if n > 3 else ""
        digit_ratio = sum(1 for c in tail if c.isdigit()) / max(len(tail), 1)
        score += digit_ratio * 2.0
        return score

    best_text = (
        text if score_candidate(text) >= score_candidate(corrected) else corrected
    )
    best_score = score_candidate(best_text)

    # Format output with a lighter fallback for 2-line motorcycle plates.
    if len(best_text) == 6:
        formatted = f"{best_text[:2]}-{best_text[2:]}"
    elif len(best_text) == 7:
        formatted = f"{best_text[:3]}-{best_text[3:]}"
    else:
        formatted = format_plate_text(best_text)
    confidence = min(best_score / 11.0, 1.0)

    return formatted, confidence


def log_car_candidate(
    debug_dir: Path,
    car_id: int,
    frame_idx: int,
    raw_text: str,
    row_texts: list[str],
    normalized_candidate: str,
    formatted_candidate: str,
    plate_score: float,
    ocr_conf: float,
    crop_w: int,
    crop_h: int,
    best_raw_now: str,
    best_formatted_now: str,
) -> None:
    """Log detailed candidate info for each car per frame"""
    candidates_path = debug_dir / "car_candidates.csv"
    file_exists = candidates_path.exists()

    row_texts_str = "|".join(row_texts) if row_texts else ""

    with open(candidates_path, "a", newline="", encoding="utf-8") as f:
        if not file_exists:
            f.write(
                "car_id,frame,raw_text,row_texts,normalized_candidate,formatted_candidate,plate_score,ocr_conf,crop_w,crop_h,best_raw_now,best_formatted_now\n"
            )
        f.write(
            f'{car_id},{frame_idx},"{raw_text}","{row_texts_str}","{normalized_candidate}","{formatted_candidate}",{plate_score:.3f},{ocr_conf:.3f},{crop_w},{crop_h},"{best_raw_now}","{best_formatted_now}"\n'
        )


def save_debug_crop(
    debug_dir: Path,
    car_id: int,
    frame_idx: int,
    crop_idx: int,
    raw_crop: np.ndarray,
    enhanced_crop: np.ndarray,
    raw_text: str,
    formatted_text: str,
    corrected_text: str,
    corrected_formatted: str,
    valid_plate: bool,
    likely_plate: bool,
) -> None:
    # Create summary entry
    summary_entry = {
        "frame": frame_idx,
        "crop": crop_idx,
        "raw_w": raw_crop.shape[1],
        "raw_h": raw_crop.shape[0],
        "enh_w": enhanced_crop.shape[1],
        "enh_h": enhanced_crop.shape[0],
        "raw_text": raw_text,
        "formatted": formatted_text,
        "corrected": corrected_text,
        "corrected_fmt": corrected_formatted,
        "valid": valid_plate,
        "likely": likely_plate,
    }

    # Save raw crop
    safe_raw = normalize_plate_text(raw_text) or "EMPTY"
    safe_fmt = normalize_plate_text(formatted_text) or "INVALID"
    raw_filename = (
        f"frame_{frame_idx:05d}_crop_{crop_idx:02d}_{safe_raw}_{safe_fmt}.jpg"
    )
    cv2.imwrite(str(debug_dir / "raw" / raw_filename), raw_crop)

    # Save enhanced crop
    safe_corr = normalize_plate_text(corrected_text) or "EMPTY"
    safe_corr_fmt = normalize_plate_text(corrected_formatted) or "INVALID"
    enh_filename = (
        f"frame_{frame_idx:05d}_crop_{crop_idx:02d}_{safe_corr}_{safe_corr_fmt}.jpg"
    )
    cv2.imwrite(str(debug_dir / "enhanced" / enh_filename), enhanced_crop)

    # Append to summary CSV
    summary_path = debug_dir / "summary.csv"
    file_exists = summary_path.exists()

    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        if not file_exists:
            # Write header
            f.write(
                "car_id,frame,crop,raw_w,raw_h,enh_w,enh_h,raw_text,formatted_text,corrected_text,corrected_formatted,valid_vn,likely_plate\n"
            )
        # Write data
        f.write(
            f'{car_id},{frame_idx},{crop_idx},{raw_crop.shape[1]},{raw_crop.shape[0]},{enhanced_crop.shape[1]},{enhanced_crop.shape[0]},"{raw_text}","{formatted_text}","{corrected_text}","{corrected_formatted}",{valid_plate},{likely_plate}\n'
        )


def draw_plate_panel(
    canvas: np.ndarray,
    panel_x: int,
    width: int,
    car_plates: list[tuple],
    frame_idx: int,
    timings: dict[str, float],
    manual_corrections: dict[int, str] = None,
) -> None:
    """Draw panel with car ID, plate, and confidence
    car_plates: list of (car_id, plate_text, confidence, frame_count)
    manual_corrections: dict of {car_id: corrected_plate}
    """
    if manual_corrections is None:
        manual_corrections = {}

    cv2.rectangle(
        canvas,
        (panel_x, 0),
        (panel_x + width, canvas.shape[0]),
        (18, 18, 18),
        thickness=-1,
    )
    cv2.putText(
        canvas,
        "TRACKED CARS",
        (panel_x + 24, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Frame {frame_idx}",
        (panel_x + 24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )

    latency = (
        f"det {timings['detect_ms']:.1f}ms  "
        f"ocr {timings['ocr_ms']:.1f}ms  "
        f"frame {timings['frame_ms']:.1f}ms"
    )
    cv2.putText(
        canvas,
        latency,
        (panel_x + 24, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )

    y = 160
    if not car_plates:
        cv2.putText(
            canvas,
            "No cars",
            (panel_x + 24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (90, 90, 90),
            2,
            cv2.LINE_AA,
        )
        return

    for car_id, plate_text, confidence, frame_count, source in car_plates[:6]:
        # Check if manual correction exists
        has_correction = car_id in manual_corrections
        display_plate = manual_corrections[car_id] if has_correction else plate_text

        # Determine color
        if has_correction:
            color = (0, 255, 0)  # Bright green - manually corrected
            label = "CORRECTED"
        elif confidence > 0.7:
            color = (0, 255, 0)  # Green - high confidence
            label = "AUTO"
        elif confidence > 0.4:
            color = (0, 255, 255)  # Yellow - medium
            label = "AUTO"
        else:
            color = (0, 165, 255)  # Orange - low
            label = "AUTO"

        # Draw car ID
        id_text = f"Car #{car_id}"
        cv2.putText(
            canvas,
            id_text,
            (panel_x + 24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        y += 28

        # Draw plate text (large)
        cv2.putText(
            canvas,
            display_plate,
            (panel_x + 24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3,
            cv2.LINE_AA,
        )
        y += 35

        # Draw confidence/status, frame count, and source
        if has_correction:
            info_text = f"{label} | frames:{frame_count}"
        else:
            source_label = (
                f" [{source.upper()}]" if source in ["plate", "vehicle"] else ""
            )
            info_text = (
                f"{label}{source_label} conf:{confidence:.2f} | frames:{frame_count}"
            )
            cv2.putText(
                canvas,
                info_text,
                (panel_x + 24, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )
        y += 35


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = ROOT / "debug_crops"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Load manual corrections
    correction_file = ROOT / "docs" / "manual_correction.txt"
    manual_corrections = load_manual_corrections(correction_file)
    if manual_corrections:
        print(f"Loaded manual corrections: {manual_corrections}")

    device = select_device(args.device)
    detector = DetectMultiBackend(args.det_model, device=device, fp16=False)
    ocr_model = DetectMultiBackend(args.ocr_model, device=device, fp16=False)
    car_detector = DetectMultiBackend(args.car_model, device=device, fp16=False)
    print(f"Models loaded: plate_det | ocr | car_det (COCO)")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    panel_width = args.panel_width
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width + panel_width, height),
    )

    frame_idx = 0
    debug_saved = 0
    car_tracker = CarTracker(max_frames_missing=60)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        frame_start = perf_counter()

        # --- Step 1: Detect cars (large bbox, stable tracking) ---
        det_start = perf_counter()
        car_detections = run_yolov5(
            car_detector,
            frame,
            args.imgsz,
            args.car_conf,
            args.iou,
            classes=CarTracker.VEHICLE_CLASSES,
        )
        # Track each detected car
        for car_det in car_detections.tolist():
            cx1, cy1, cx2, cy2, cconf, ccls = car_det
            car_bbox = [int(cx1), int(cy1), int(cx2), int(cy2)]
            car_id = car_tracker.match_car(car_bbox, frame_idx)
            # Draw car bbox (thin, semi-transparent feel)
            cv2.rectangle(
                frame, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (200, 200, 200), 1
            )
            cv2.putText(
                frame,
                f"#{car_id}",
                (int(cx1) + 4, int(cy1) + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        detect_ms = (perf_counter() - det_start) * 1000.0

        # --- Step 2: Detect plates, match to car, run OCR ---
        ocr_total_ms = 0.0
        matched_plate_car_ids = set()
        plate_detections = run_yolov5(
            detector, frame, args.imgsz, args.det_conf, args.iou
        )

        for crop_idx, det in enumerate(plate_detections.tolist()):
            x1, y1, x2, y2, conf, _ = det

            # Filter portrait / too small
            bw, bh = int(x2 - x1), int(y2 - y1)
            if bw < 30 or bh < 12:
                continue
            if bh / max(bw, 1) > 3.0:
                continue

            plate_bbox = [int(x1), int(y1), int(x2), int(y2)]

            # Match plate to nearest car
            car_id, match_score = car_tracker.find_car_for_plate(plate_bbox, frame_idx)
            if car_id is None:
                # No car found, create a standalone tracker entry
                car_id = car_tracker.match_car(plate_bbox, frame_idx)

            matched_plate_car_ids.add(car_id)

            # Crop with padding from original frame
            px1, py1, px2, py2 = pad_box(
                int(x1), int(y1), int(x2), int(y2), frame.shape, args.padding
            )
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            enhanced_crop = enhance_plate_crop(
                crop, min_height=64, apply_threshold=True
            )
            soft_crop = enhance_plate_crop(crop, min_height=64, apply_threshold=False)

            # Extra sharp variant often recovers weak strokes better than thresholding.
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray_crop = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(
                gray_crop
            )
            gray_crop = cv2.addWeighted(
                gray_crop, 1.8, cv2.GaussianBlur(gray_crop, (0, 0), 2.0), -0.8, 0
            )
            sharp_crop = cv2.cvtColor(
                cv2.resize(
                    gray_crop, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4
                ),
                cv2.COLOR_GRAY2BGR,
            )

            def extract_frame_candidate(processed_crop: np.ndarray):
                ocr_start_local = perf_counter()
                local_imgsz = (
                    1280
                    if processed_crop.shape[0] < 120 or processed_crop.shape[1] < 220
                    else args.imgsz
                )
                local_conf = 0.03 if crop.shape[0] < 35 else args.ocr_conf
                dets = run_yolov5(
                    ocr_model, processed_crop, local_imgsz, local_conf, args.iou
                )
                local_ms = (perf_counter() - ocr_start_local) * 1000.0

                chars_local = sorted(
                    [
                        {
                            "char": ocr_model.names[int(cls)],
                            "conf": float(char_conf),
                            "x1": float(x1c),
                            "cy": float((y1c + y2c) / 2.0),
                            "h": float(y2c - y1c),
                        }
                        for *xyxy, char_conf, cls in dets.tolist()
                        for x1c, y1c, x2c, y2c in [xyxy]
                    ],
                    key=lambda c: c["x1"],
                )

                raw_text_local, row_texts_local = decode_plate_text(
                    dets, ocr_model.names
                )
                norm_local = select_frame_plate_candidate(
                    raw_text_local, row_texts_local
                )
                conf_local = (
                    (sum(c["conf"] for c in chars_local) / len(chars_local))
                    if chars_local
                    else 0.0
                )
                quality_local = plate_candidate_score(norm_local)
                return {
                    "dets": dets,
                    "chars": chars_local,
                    "raw": raw_text_local,
                    "rows": row_texts_local,
                    "norm": norm_local,
                    "conf": conf_local,
                    "quality": quality_local,
                    "latency_ms": local_ms,
                    "crop": processed_crop,
                }

            hard_candidate = extract_frame_candidate(enhanced_crop)
            soft_candidate = extract_frame_candidate(soft_crop)
            sharp_candidate = extract_frame_candidate(sharp_crop)
            ocr_total_ms += (
                hard_candidate["latency_ms"]
                + soft_candidate["latency_ms"]
                + sharp_candidate["latency_ms"]
            )

            chosen = max(
                [hard_candidate, soft_candidate, sharp_candidate],
                key=lambda item: (
                    item["quality"],
                    len(normalize_plate_text(item["norm"])),
                    item["conf"],
                ),
            )

            enhanced_crop = chosen["crop"]
            chars = chosen["chars"]
            frame_raw_text = chosen["raw"]
            row_texts = chosen["rows"]
            frame_norm = chosen["norm"]
            frame_conf = chosen["conf"]

            # Add per-frame text to car history (not char pool)
            car_tracker.add_frame_text(
                car_id, frame_norm, frame_conf, frame_idx, bw, bh
            )

            # Get best plate from voting history
            best_raw, best_conf = car_tracker.get_best_plate(car_id)
            formatted_plate, format_conf = smart_format_vn_plate(best_raw)
            combined_conf = best_conf * 0.6 + format_conf * 0.4

            # Log detailed candidate info for car_id tracking
            log_car_candidate(
                debug_dir,
                car_id,
                frame_idx,
                raw_text=frame_norm,
                row_texts=row_texts,
                normalized_candidate=frame_norm,
                formatted_candidate=formatted_plate,
                plate_score=plate_candidate_score(frame_norm),
                ocr_conf=frame_conf,
                crop_w=bw,
                crop_h=bh,
                best_raw_now=best_raw,
                best_formatted_now=formatted_plate,
            )

            # --- Vehicle text extraction as fallback for small/unreadable plates ---
            vehicle_text = ""
            vehicle_conf = 0.0
            vehicle_text_source = ""

            # Only attempt vehicle text extraction if plate OCR is weak or failed
            plate_is_weak = (
                bh < PLATE_SMALL_HEIGHT_THRESH  # Plate too small
                or bw < PLATE_SMALL_WIDTH_THRESH  # Plate too narrow
                or frame_conf < PLATE_LOW_CONF_THRESH  # Low OCR confidence
                or not (
                    is_valid_vn_plate(frame_norm) or is_likely_vn_plate(frame_norm)
                )  # Not plate-like
            )

            if plate_is_weak and frame_idx >= PLATE_MIN_FRAMES_FOR_FALLBACK:
                # Extract text from vehicle body regions
                frame_shape = frame.shape
                vehicle_regions = extract_vehicle_text_regions(
                    (x1, y1, x2, y2), frame_shape
                )

                best_vehicle_score = 0.0

                for region_idx, (rx1, ry1, rx2, ry2) in enumerate(vehicle_regions):
                    # Extract region from original frame
                    vehicle_crop = frame[ry1:ry2, rx1:rx2]
                    if vehicle_crop.size == 0:
                        continue

                    # Enhance the vehicle crop for better text detection
                    enhanced_vehicle = enhance_plate_crop(
                        vehicle_crop,
                        min_height=VEHICLE_TEXT_MIN_HEIGHT,
                        apply_threshold=False,
                    )

                    # Run OCR on vehicle text region
                    vehicle_ocr_start = perf_counter()
                    vehicle_local_imgsz = (
                        1280
                        if enhanced_vehicle.shape[0] < 80
                        or enhanced_vehicle.shape[1] < 150
                        else args.imgsz
                    )
                    vehicle_local_conf = (
                        0.02  # Lower confidence for vehicle text detection
                    )
                    vehicle_dets = run_yolov5(
                        ocr_model,
                        enhanced_vehicle,
                        vehicle_local_imgsz,
                        vehicle_local_conf,
                        args.iou,
                    )
                    vehicle_local_ms = (perf_counter() - vehicle_ocr_start) * 1000.0
                    ocr_total_ms += vehicle_local_ms

                    # Process vehicle text detections
                    vehicle_chars_local = sorted(
                        [
                            {
                                "char": ocr_model.names[int(cls)],
                                "conf": float(char_conf),
                                "x1": float(x1c),
                                "cy": float((y1c + y2c) / 2.0),
                                "h": float(y2c - y1c),
                            }
                            for *xyxy, char_conf, cls in vehicle_dets.tolist()
                            for x1c, y1c, x2c, y2c in [xyxy]
                        ],
                        key=lambda c: c["x1"],
                    )

                    vehicle_raw_text_local, vehicle_row_texts_local = decode_plate_text(
                        vehicle_dets, ocr_model.names
                    )
                    vehicle_norm_local = select_frame_plate_candidate(
                        vehicle_raw_text_local, vehicle_row_texts_local
                    )
                    vehicle_conf_local = (
                        (
                            sum(c["conf"] for c in vehicle_chars_local)
                            / len(vehicle_chars_local)
                        )
                        if vehicle_chars_local
                        else 0.0
                    )
                    vehicle_quality_local = plate_candidate_score(vehicle_norm_local)

                    # Skip if no text detected
                    if (
                        not vehicle_norm_local
                        or len(vehicle_norm_local) < VEHICLE_TEXT_MIN_LENGTH
                    ):
                        continue

                    # Score this vehicle text candidate
                    vehicle_score = score_vehicle_text_candidate(
                        vehicle_norm_local,
                        vehicle_conf_local,
                        frame_idx,
                        car_id,
                        car_tracker,
                    )

                    # Store in vehicle text history
                    if car_id in car_tracker.cars:
                        if "vehicle_text_history" not in car_tracker.cars[car_id]:
                            car_tracker.cars[car_id]["vehicle_text_history"] = []

                        car_tracker.cars[car_id]["vehicle_text_history"].append(
                            (
                                frame_idx,
                                vehicle_raw_text_local,
                                vehicle_norm_local,
                                vehicle_score,
                                f"region_{region_idx}",
                            )
                        )

                        # Keep only recent window (same as text_history)
                        if (
                            len(car_tracker.cars[car_id]["vehicle_text_history"])
                            > car_tracker.history_window
                        ):
                            car_tracker.cars[car_id]["vehicle_text_history"] = (
                                car_tracker.cars[car_id]["vehicle_text_history"][
                                    -car_tracker.history_window :
                                ]
                            )

                    # Track best vehicle text candidate
                    if vehicle_score > best_vehicle_score:
                        best_vehicle_score = vehicle_score
                        vehicle_text = vehicle_norm_local
                        vehicle_conf = vehicle_conf_local
                        vehicle_text_source = f"region_{region_idx}"

            # Special tracking for car#3 with known ground truth
            if car_id == 3:
                save_car3_crop(
                    debug_dir,
                    frame_idx,
                    enhanced_crop.copy(),
                    raw_text=frame_norm,
                    normalized=frame_norm,
                    formatted=formatted_plate,
                    crop_w=bw,
                    crop_h=bh,
                )

            # Update cached best only when candidate looks reasonably VN-like.
            cacheable = (
                bool(formatted_plate)
                and len(normalize_plate_text(best_raw)) >= 5
                and normalize_vn_candidate(best_raw)[:2].isdigit()
                and combined_conf >= 0.45
            )
            if cacheable:
                car_tracker.update_best_plate(car_id, formatted_plate, combined_conf)

            # Draw plate bbox on frame
            if combined_conf > 0.5:
                box_color = (0, 255, 0)  # Green
            elif combined_conf > 0.3:
                box_color = (0, 255, 255)  # Yellow
            else:
                box_color = (0, 165, 255)  # Orange

            # Determine final display text and color
            final_text = (
                formatted_plate if formatted_plate else (best_raw if best_raw else "?")
            )
            final_conf = combined_conf
            final_color = box_color

            # Check if we should use vehicle text instead
            if vehicle_text and plate_is_weak:
                # Get best vehicle text for display
                best_vehicle_text, best_vehicle_conf = (
                    car_tracker.get_best_vehicle_text(car_id)
                )
                if (
                    best_vehicle_text and best_vehicle_conf > 0.3
                ):  # Minimum confidence for vehicle text
                    final_text = f"VEHICLE: {best_vehicle_text}"
                    final_conf = best_vehicle_conf
                    final_color = (255, 255, 0)  # Cyan for vehicle text

            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
            cv2.putText(
                frame,
                final_text,
                (px1, max(22, py1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                final_color,
                2,
                cv2.LINE_AA,
            )

            # Display vehicle text fallback if available and plate is weak
            if vehicle_text and plate_is_weak:
                vehicle_display_text = f"VEHICLE: {vehicle_text}"
                cv2.putText(
                    frame,
                    vehicle_display_text,
                    (px1, max(22, py1 - 6) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),  # Cyan color for vehicle text
                    1,
                    cv2.LINE_AA,
                )

            # Debug crops
            if debug_saved < args.debug_limit and combined_conf < 0.5:
                corrected_text = "".join(OCR_CORRECTIONS.get(c, c) for c in frame_norm)
                save_debug_crop(
                    debug_dir,
                    car_id,
                    frame_idx,
                    crop_idx,
                    crop,
                    enhanced_crop,
                    frame_norm,
                    formatted_plate,
                    corrected_text,
                    format_plate_text(corrected_text),
                    is_valid_vn_plate(frame_norm),
                    is_likely_plate(frame_norm),
                )
                debug_saved += 1

        # Cleanup stale cars
        car_tracker.cleanup_old_cars(frame_idx)

        frame_ms = (perf_counter() - frame_start) * 1000.0
        timings = {
            "detect_ms": detect_ms,
            "ocr_ms": ocr_total_ms,
            "frame_ms": frame_ms,
        }

        # Build car plates list for panel display from voted history.
        active_cars = car_tracker.get_active_cars(frame_idx, max_age=25)
        car_plates = []
        for car_id, car in sorted(active_cars, key=lambda x: -x[1]["frame_count"]):
            best_raw, best_conf = car_tracker.get_best_plate(car_id)
            best_vehicle_text, best_vehicle_conf = car_tracker.get_best_vehicle_text(
                car_id
            )

            # Determine what to display: plate text or vehicle text fallback
            display_text = ""
            display_conf = 0.0
            display_source = "plate"

            # Check if we have a valid plate
            if best_raw and is_valid_vn_plate(best_raw):
                fmt, fmt_conf = smart_format_vn_plate(best_raw)
                display_text = fmt if fmt else best_raw
                display_conf = best_conf * 0.6 + fmt_conf * 0.4
                display_source = "plate"
            elif best_raw and is_likely_vn_plate(best_raw):
                # Less confident but still plate-like
                display_text = best_raw
                display_conf = best_conf * 0.5
                display_source = "plate"
            elif best_vehicle_text and best_vehicle_conf > 0.3:
                # Use vehicle text as fallback
                display_text = f"VEHICLE: {best_vehicle_text}"
                display_conf = best_vehicle_conf
                display_source = "vehicle"
            elif best_raw:
                # Fallback to raw plate text even if not validated
                display_text = best_raw
                display_conf = best_conf * 0.3
                display_source = "plate"
            else:
                display_text = "---"
                display_conf = 0.0
                display_source = "none"

            car_plates.append(
                (car_id, display_text, display_conf, car["frame_count"], display_source)
            )

        canvas = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
        canvas[:, :width] = frame
        draw_plate_panel(
            canvas,
            width,
            panel_width,
            car_plates,
            frame_idx,
            timings,
            manual_corrections,
        )
        writer.write(canvas)

        frame_idx += 1
        if frame_idx % 30 == 0:
            progress_total = total_frames if total_frames > 0 else frame_idx
            active_count = len(car_tracker.get_active_cars(frame_idx, max_age=20))
            print(
                f"Processed {frame_idx}/{progress_total} frames | Active cars: {active_count}",
                flush=True,
            )

    cap.release()
    writer.release()
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
