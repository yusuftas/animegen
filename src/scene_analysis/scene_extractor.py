"""
Scene Detection and Extraction Module
Identifies and extracts interesting scenes from anime videos
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging

try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    logging.warning("scenedetect not available, using fallback scene detection")

@dataclass
class Scene:
    start_time: float
    end_time: float
    duration: float
    start_frame: int
    end_frame: int
    confidence: float
    metadata: Dict[str, Any]

class SceneExtractor:
    def __init__(self, threshold: float = 30.0, min_scene_length: float = 2.0, max_scene_length: float = 15.0):
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.max_scene_length = max_scene_length
        self.logger = logging.getLogger(__name__)
        
    def extract_scenes(self, video_path: str) -> List[Scene]:
        """Extract scenes from video using scene detection"""
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        if SCENEDETECT_AVAILABLE:
            return self._extract_scenes_scenedetect(video_path)
        else:
            return self._extract_scenes_fallback(video_path)
    
    def _extract_scenes_scenedetect(self, video_path: str) -> List[Scene]:
        """Extract scenes using PySceneDetect library"""
        self.logger.info(f"Extracting scenes from {video_path} using scenedetect")
        
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))
        
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        
        fps = video_manager.get_framerate()
        scenes = []
        
        for i, (start_time, end_time) in enumerate(scene_list):
            duration = (end_time - start_time).total_seconds()
            
            if self.min_scene_length <= duration <= self.max_scene_length:
                scene = Scene(
                    start_time=start_time.total_seconds(),
                    end_time=end_time.total_seconds(),
                    duration=duration,
                    start_frame=int(start_time.total_seconds() * fps),
                    end_frame=int(end_time.total_seconds() * fps),
                    confidence=1.0,
                    metadata={'method': 'scenedetect', 'detector': 'content'}
                )
                scenes.append(scene)
        
        video_manager.release()
        self.logger.info(f"Extracted {len(scenes)} valid scenes")
        return scenes
    
    def _extract_scenes_fallback(self, video_path: str) -> List[Scene]:
        """Fallback scene detection using OpenCV"""
        self.logger.info(f"Extracting scenes from {video_path} using fallback method")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scenes = []
        scene_changes = self._detect_scene_changes_opencv(cap, fps)
        
        for i in range(len(scene_changes) - 1):
            start_frame = scene_changes[i]
            end_frame = scene_changes[i + 1]
            
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time
            
            if self.min_scene_length <= duration <= self.max_scene_length:
                scene = Scene(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    confidence=0.8,
                    metadata={'method': 'opencv_fallback'}
                )
                scenes.append(scene)
        
        cap.release()
        self.logger.info(f"Extracted {len(scenes)} valid scenes using fallback")
        return scenes
    
    def _detect_scene_changes_opencv(self, cap: cv2.VideoCapture, fps: float) -> List[int]:
        """Detect scene changes using frame difference analysis"""
        scene_changes = [0]
        prev_frame = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                diff_score = np.mean(diff)
                
                if diff_score > self.threshold:
                    scene_changes.append(frame_count)
            
            prev_frame = gray
            frame_count += 1
        
        scene_changes.append(frame_count - 1)
        return scene_changes
    
    def extract_scene_frames(self, video_path: str, scene: Scene, num_frames: int = 5) -> List[np.ndarray]:
        """Extract key frames from a specific scene"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        frame_indices = np.linspace(scene.start_frame, scene.end_frame, num_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def get_scene_thumbnail(self, video_path: str, scene: Scene) -> np.ndarray:
        """Get a representative thumbnail for the scene"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        mid_frame = int((scene.start_frame + scene.end_frame) / 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        
        cap.release()
        
        if ret:
            return frame
        else:
            raise ValueError("Could not extract thumbnail frame")