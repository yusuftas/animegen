"""
Scene Detection and Extraction Module
Identifies and extracts interesting scenes from anime videos with robust detection methods
"""

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV (cv2) not available. Fallback scene detection will be disabled.")

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
    def __init__(self, threshold: float = 30.0, min_scene_length: float = 5.0, max_scene_length: float = 15.0):
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.max_scene_length = max_scene_length
        self.logger = logging.getLogger(__name__)
        
    def extract_scenes(self, video_path: str) -> List[Scene]:
        """Extract scenes from video using robust scene detection"""
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if CV2_AVAILABLE:
            return self._extract_scenes_robust(video_path)
        elif SCENEDETECT_AVAILABLE:
            return self._extract_scenes_scenedetect(video_path)
        else:
            self.logger.error("Neither OpenCV nor scenedetect available for scene extraction")
            raise ImportError("Scene extraction requires either OpenCV or scenedetect")
    
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
            # Convert FrameTimecode to seconds
            start_seconds = start_time.get_seconds() if hasattr(start_time, 'get_seconds') else float(start_time)
            end_seconds = end_time.get_seconds() if hasattr(end_time, 'get_seconds') else float(end_time)
            duration = end_seconds - start_seconds
            
            if self.min_scene_length <= duration <= self.max_scene_length:
                scene = Scene(
                    start_time=start_seconds,
                    end_time=end_seconds,
                    duration=duration,
                    start_frame=int(start_seconds * fps),
                    end_frame=int(end_seconds * fps),
                    confidence=1.0,
                    metadata={'method': 'scenedetect', 'detector': 'content'}
                )
                scenes.append(scene)
        
        video_manager.release()
        self.logger.info(f"Extracted {len(scenes)} valid scenes")
        return scenes
    
    def _extract_scenes_robust(self, video_path: str) -> List[Scene]:
        """Robust scene detection with multiple methods and fallbacks"""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV not available for robust scene detection")
            
        self.logger.info(f"Extracting scenes from {video_path} using robust method")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        self.logger.info(f"Video info: {duration:.1f}s, {total_frames} frames, {fps:.1f} FPS")
        self.logger.info(f"Scene validation: {self.min_scene_length}s - {self.max_scene_length}s duration")
        
        scenes = []
        
        try:
            # Method 1: Histogram-based detection
            self.logger.info("Method 1: Histogram-based scene detection...")
            scenes = self._histogram_scene_detection(cap, fps)
            
            if len(scenes) < 5:
                self.logger.warning("Few scenes detected, trying threshold method...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                scenes.extend(self._threshold_scene_detection(cap, fps))
            
            if len(scenes) < 8:
                self.logger.warning("Still few scenes, adding fixed intervals...")
                scenes.extend(self._fixed_interval_scenes(duration))
            
        except Exception as e:
            self.logger.error(f"Scene detection error: {e}")
            self.logger.info("Using fallback fixed interval method...")
            scenes = self._fixed_interval_scenes(duration)
        
        finally:
            cap.release()
        
        # Remove duplicates and sort
        scenes = self._remove_duplicate_scenes(scenes, min_gap=2.0)
        scenes = sorted(scenes, key=lambda x: x.start_time)
        
        self.logger.info(f"Detected {len(scenes)} scenes ({self.min_scene_length}s+ duration)")
        return scenes
    
    def _histogram_scene_detection(self, cap: cv2.VideoCapture, fps: float) -> List[Scene]:
        """Histogram-based scene change detection"""
        scenes = []
        prev_hist = None
        scene_start = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % int(fps * 0.5) == 0:  # Sample every 0.5 seconds
                # Calculate histogram
                hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                if prev_hist is not None:
                    # Calculate histogram difference
                    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    
                    # Scene change threshold
                    if diff < 0.7:  # Lower correlation = scene change
                        current_time = frame_count / fps
                        scene_duration = current_time - scene_start
                        
                        if self.min_scene_length <= scene_duration <= self.max_scene_length:
                            scene = Scene(
                                start_time=scene_start,
                                end_time=current_time,
                                duration=scene_duration,
                                start_frame=int(scene_start * fps),
                                end_frame=int(current_time * fps),
                                confidence=1.0 - diff,
                                metadata={'method': 'histogram', 'correlation': diff}
                            )
                            scenes.append(scene)
                        
                        scene_start = current_time
                
                prev_hist = hist
            
            frame_count += 1
        
        # Add final scene
        final_time = frame_count / fps
        if final_time - scene_start >= self.min_scene_length:
            scene = Scene(
                start_time=scene_start,
                end_time=min(scene_start + self.max_scene_length, final_time),
                duration=min(self.max_scene_length, final_time - scene_start),
                start_frame=int(scene_start * fps),
                end_frame=int(min(scene_start + self.max_scene_length, final_time) * fps),
                confidence=0.8,
                metadata={'method': 'histogram', 'final_scene': True}
            )
            scenes.append(scene)
        
        return scenes
    
    def _threshold_scene_detection(self, cap: cv2.VideoCapture, fps: float) -> List[Scene]:
        """Simple threshold-based scene detection"""
        scenes = []
        prev_frame = None
        scene_start = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % int(fps) == 0:  # Sample every second
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(diff)
                    
                    if mean_diff > self.threshold:  # Scene change threshold
                        current_time = frame_count / fps
                        scene_duration = current_time - scene_start
                        
                        if self.min_scene_length <= scene_duration <= self.max_scene_length:
                            scene = Scene(
                                start_time=scene_start,
                                end_time=current_time,
                                duration=scene_duration,
                                start_frame=int(scene_start * fps),
                                end_frame=int(current_time * fps),
                                confidence=min(mean_diff / 50.0, 1.0),
                                metadata={'method': 'threshold', 'diff_score': mean_diff}
                            )
                            scenes.append(scene)
                        
                        scene_start = current_time
                
                prev_frame = gray
            
            frame_count += 1
        
        return scenes
    
    def _fixed_interval_scenes(self, duration: float) -> List[Scene]:
        """Generate scenes at fixed intervals"""
        scenes = []
        interval = self.max_scene_length * 0.6  # Less overlap for longer scenes
        
        current_time = 0
        while current_time < duration - self.min_scene_length:
            scene_end = min(current_time + self.max_scene_length, duration)
            if scene_end - current_time >= self.min_scene_length:
                scene = Scene(
                    start_time=current_time,
                    end_time=scene_end,
                    duration=scene_end - current_time,
                    start_frame=int(current_time * 30),  # Assume 30 FPS for fixed intervals
                    end_frame=int(scene_end * 30),
                    confidence=0.5,
                    metadata={'method': 'fixed_interval'}
                )
                scenes.append(scene)
            current_time += interval
        
        return scenes
    
    def _remove_duplicate_scenes(self, scenes: List[Scene], min_gap: float = 2.0) -> List[Scene]:
        """Remove overlapping or too-close scenes"""
        if not scenes:
            return scenes
        
        unique_scenes = [scenes[0]]
        
        for scene in scenes[1:]:
            last_end = unique_scenes[-1].end_time
            if scene.start_time - last_end >= min_gap:
                unique_scenes.append(scene)
        
        return unique_scenes
    
    def extract_scene_frames(self, video_path: str, scene: Scene, num_frames: int = 5) -> List[np.ndarray]:
        """Extract key frames from a specific scene"""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV not available for frame extraction")
            
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
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV not available for thumbnail extraction")
            
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