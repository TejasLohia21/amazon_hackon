import os
import cv2
from typing import List

class VideoSampler:
    """Samples frames from a video file using OpenCV."""
    def __init__(self, video_path: str, output_dir: str, fps: float = 0.5):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if fps <= 0:
            raise ValueError("FPS must be a positive number.")

        self.video_path = video_path
        self.output_dir = output_dir
        self.fps = fps
        os.makedirs(self.output_dir, exist_ok=True)

    def sample_frames(self) -> List[str]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / self.fps)

        frame_paths = []
        frame_index = 0
        saved_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                frame_filename = os.path.join(self.output_dir, f"frame_{saved_index:05d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_paths.append(frame_filename)
                saved_index += 1

            frame_index += 1

        cap.release()
        return frame_paths