"""Extract best frames for specific cars to manually verify plates"""
import cv2
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "yolov5"))

def extract_frames_for_cars(video_path, car_frames, output_dir):
    """Extract specific frames from video
    
    Args:
        video_path: Path to video
        car_frames: dict {car_id: [frame_numbers]}
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    frame_idx = 0
    frames_to_extract = set()
    for car_id, frames in car_frames.items():
        frames_to_extract.update(frames)
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        if frame_idx in frames_to_extract:
            # Find which cars are in this frame
            cars_in_frame = [car_id for car_id, frames in car_frames.items() if frame_idx in frames]
            cars_str = "_".join([f"car{cid}" for cid in cars_in_frame])
            
            filename = f"frame_{frame_idx:05d}_{cars_str}.jpg"
            cv2.imwrite(str(output_dir / filename), frame)
            print(f"Extracted frame {frame_idx} for {cars_str}")
        
        frame_idx += 1
    
    cap.release()
    print(f"Done! Extracted {len(frames_to_extract)} frames to {output_dir}")


if __name__ == "__main__":
    video_path = ROOT / "video test" / "pwf2amf0gwaaaz1wsl18ywyayxuoyxmaap1e6w1.mp4"
    
    # Based on car_candidates.csv analysis
    car_frames = {
        3: [619],  # Only frame for car#3
        5: [770, 776, 779],  # Best frames for car#5 (largest crops, highest conf)
    }
    
    output_dir = ROOT / "debug_crops" / "manual_verify"
    extract_frames_for_cars(video_path, car_frames, output_dir)
