import sys
import os
# add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.video_io import extract_frames, detect_and_crop_faces

video_path = "data/raw/sample_video.mp4"  # your sample video
frames_dir = "data/frames/raw"
crops_dir = "data/frames/faces"

print("Extracting frames...")
n = extract_frames(video_path, frames_dir, fps_extract=1)
print(f"Saved {n} frames to {frames_dir}")

print("Detecting and cropping faces...")
m = detect_and_crop_faces(frames_dir, crops_dir)
print(f"Saved {m} face crops to {crops_dir}")
