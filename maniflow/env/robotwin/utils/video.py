import numpy as np
import cv2
import wandb
from typing import Optional, List
import io

class SimulationVideoHandler:
    def __init__(self, fps: int = 10, skip_local: bool = True):
        """
        Initialize video handler for simulation recordings
        
        Args:
            fps: Frames per second for the output video
            skip_local: If True, only upload to wandb without saving locally
        """
        self.fps = fps
        self.skip_local = skip_local
        self.frames: List[np.ndarray] = []
        self.frame_size = None
    
    def add_frame(self, frame: np.ndarray):
        """Add a new frame to the video buffer"""
        if self.frame_size is None:
            self.frame_size = (frame.shape[1], frame.shape[0])
        self.frames.append(frame)
    
    def save_and_upload(self, wandb_run, step: int, local_path: Optional[str] = None):
        """
        Save the video and upload to wandb
        
        Args:
            wandb_run: Active wandb run instance
            step: Current step/episode number for logging
            local_path: Path to save video locally (if skip_local is False)
        """
        if not self.frames:
            return
        
        # Create video buffer
        video_buffer = io.BytesIO()
        
        # Create VideoWriter object writing to memory
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            local_path if not self.skip_local else 'tmp.mp4',
            fourcc,
            self.fps,
            self.frame_size,
            True
        )
        
        # Write frames
        for frame in self.frames:
            out.write(frame)
        
        out.release()
        
        # If not skipping local save, the video is already saved
        if not self.skip_local and local_path:
            print(f"Video saved locally to {local_path}")
        
        # Upload to wandb
        if self.skip_local:
            # Read the temporary file into memory
            with open('tmp.mp4', 'rb') as f:
                video_data = f.read()
            import os
            os.remove('tmp.mp4')  # Clean up temp file
        else:
            with open(local_path, 'rb') as f:
                video_data = f.read()
        
        # Log to wandb
        wandb_run.log({
            "simulation_video": wandb.Video(
                video_data,
                fps=self.fps,
                format="mp4"
            )
        }, step=step)
        
        # Clear the frame buffer
        self.frames = []
        
    def __len__(self):
        return len(self.frames)