import os
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

mtcnn = MTCNN(keep_all=False, device='cpu')  # change to 'cuda' if GPU is ready

def extract_frames(video_path, out_dir, fps_extract=1):
    os.makedirs(out_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")

    video_fps = vidcap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(video_fps / fps_extract)))
    count = 0
    saved = 0
    pbar = tqdm(total=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Extracting frames")
    while True:
        success, frame = vidcap.read()
        if not success:
            break
        if count % step == 0:
            fname = os.path.join(out_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
        count += 1
        pbar.update(1)
    pbar.close()
    vidcap.release()
    return saved

def detect_and_crop_faces(frames_dir, out_dir, min_size=20):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg','.png'))])
    pbar = tqdm(files, desc="Detecting faces")
    saved = 0
    for f in pbar:
        path = os.path.join(frames_dir, f)
        img = Image.open(path).convert('RGB')
        try:
            boxes, probs = mtcnn.detect(img)
        except Exception:
            continue
        if boxes is None:
            continue
        if boxes.ndim == 2:
            box = boxes[0]
        else:
            box = boxes
        x1, y1, x2, y2 = [int(max(0, v)) for v in box]
        crop = img.crop((x1, y1, x2, y2))
        save_path = os.path.join(out_dir, f"crop_{saved:06d}.jpg")
        crop.save(save_path)
        saved += 1
    pbar.close()
    return saved
