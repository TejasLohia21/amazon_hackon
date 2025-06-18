import os
from typing import List
import imageio
from moviepy.editor import VideoFileClip

class VideoSampler:
    def __init__(self, video_path: str, output_dir: str, fps: int = 0.1):
        self.video_path = video_path
        self.output_dir = output_dir
        self.fps = fps
        self.frames = []
        self._validate_inputs()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _validate_inputs(self):
        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        if self.fps <= 0:
            raise ValueError("FPS must be a positive integer.")

    def sample_frames(self) -> List[str]:
        clip = VideoFileClip(self.video_path)
        duration = clip.duration
        frame_paths = []
        num_frames = int(duration * self.fps)
        for i in range(num_frames):
            t = i / self.fps
            frame = clip.get_frame(t)
            frame_filename = os.path.join(self.output_dir, f"frame_{i:05d}.jpg")
            imageio.imwrite(frame_filename, frame)
            frame_paths.append(frame_filename)
        self.frames = frame_paths
        return frame_paths

# Usage Example:
# sampler = VideoSampler('input_video.mp4', 'output_frames', fps=0.1)
# frame_files = sampler.sample_frames()
# print(frame_files)
