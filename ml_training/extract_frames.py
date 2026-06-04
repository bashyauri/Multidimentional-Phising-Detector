import os
import cv2
from pathlib import Path

# Settings
VIDEO_DIRS = [
    ('datasets/faceforensics/original', 'datasets/faceforensics/frames/original'),
    ('datasets/faceforensics/manipulated', 'datasets/faceforensics/frames/manipulated'),
]
FRAME_RATE = 1  # frames per second
EXTENSIONS = {'.mp4', '.avi', '.mov'}


def extract_frames_from_video(video_path, output_dir, frame_rate=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25  # fallback
    frame_interval = int(round(fps / frame_rate))
    frame_count = 0
    saved_count = 0
    success, frame = cap.read()
    while success:
        if frame_count % frame_interval == 0:
            frame_name = f"{video_path.stem}_frame{frame_count}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved_count += 1
        frame_count += 1
        success, frame = cap.read()
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")


def extract_all():
    for video_dir, out_dir in VIDEO_DIRS:
        video_dir = Path(video_dir)
        out_dir = Path(out_dir)
        for video_file in video_dir.iterdir():
            if video_file.suffix.lower() in EXTENSIONS:
                extract_frames_from_video(video_file, out_dir, FRAME_RATE)

if __name__ == "__main__":
    extract_all()
    print("Frame extraction complete.")
