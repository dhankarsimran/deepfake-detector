import cv2
import torch
import numpy as np
import tempfile
import requests
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def extract_frames(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    frames = []

    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = max(video_fps // fps, 1)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

        count += 1

    cap.release()
    return frames

def predict_video(model, source, from_url=False):
    if from_url:
        r = requests.get(source)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(r.content)
        video_path = tmp.name
    else:
        video_path = source

    frames = extract_frames(video_path)
    probs = []

    with torch.no_grad():
        for frame in frames:
            img = transform(frame).unsqueeze(0)
            output = model(img)
            prob_fake = torch.softmax(output, dim=1)[0, 0].item()
            probs.append(prob_fake)

    mean_prob = np.mean(probs)
    label = "FAKE" if mean_prob > 0.5 else "REAL"
    confidence = mean_prob if label == "FAKE" else (1 - mean_prob)

    return label, confidence, len(frames)
