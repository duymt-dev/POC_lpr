from pathlib import Path
import importlib.util


def load_inference_module():
    inference_path = Path(__file__).resolve().parents[1] / "src" / "inference.py"
    spec = importlib.util.spec_from_file_location("inference", inference_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_vehicle_text_regions_returns_large_context_for_small_bbox():
    inference = load_inference_module()

    regions = inference.extract_vehicle_text_regions(
        [100, 100, 158, 129], (720, 1280, 3)
    )

    assert regions, "expected regions to be generated"
    full_region = regions[0]
    width = full_region[2] - full_region[0]
    height = full_region[3] - full_region[1]

    # Small plate-like boxes should be expanded aggressively enough to capture body text.
    assert width >= 150, f"expected wider fallback ROI, got width={width}"
    assert height >= 70, f"expected taller fallback ROI, got height={height}"


def test_car_tracker_keeps_same_id_after_short_gap_and_bbox_shift():
    inference = load_inference_module()

    tracker = inference.CarTracker(max_frames_missing=60)

    first_id = tracker.match_car([700, 300, 820, 380], 770)
    second_id = tracker.match_car([706, 301, 818, 377], 790)
    # Simulate a short gap where detector misses the car, then car reappears smaller and shifted.
    third_id = tracker.match_car([720, 306, 810, 365], 826)

    assert first_id == 1
    assert second_id == first_id
    assert third_id == first_id, (
        f"expected tracker to keep same id across short gap, got {first_id}, {second_id}, {third_id}"
    )


if __name__ == "__main__":
    test_extract_vehicle_text_regions_returns_large_context_for_small_bbox()
    test_car_tracker_keeps_same_id_after_short_gap_and_bbox_shift()
    print("ok")
