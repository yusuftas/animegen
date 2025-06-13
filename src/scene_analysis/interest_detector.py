"""
Interest Detection Module
Analyzes scenes to determine their potential interest level for anime shorts
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

from .scene_extractor import Scene

@dataclass
class InterestScore:
    total_score: float
    motion_score: float
    face_score: float
    color_variance_score: float
    composition_score: float
    audio_peak_score: float
    metadata: Dict[str, Any]

class InterestDetector:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'motion_threshold': 0.3,
            'face_detection_enabled': True,
            'motion_weight': 0.4,
            'face_weight': 0.2,
            'color_variance_weight': 0.2,
            'composition_weight': 0.1,
            'audio_peak_weight': 0.1
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize face detector if available
        self.face_cascade = None
        if self.config.get('face_detection_enabled', True):
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if Path(cascade_path).exists():
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                else:
                    self.logger.warning("Face cascade file not found, disabling face detection")
            except Exception as e:
                self.logger.warning(f"Could not initialize face detector: {e}")
    
    def calculate_interest_score(self, video_path: str, scene: Scene, audio_peaks: List[float] = None) -> InterestScore:
        """Calculate comprehensive interest score for a scene"""
        
        # Extract frames for analysis
        frames = self._extract_scene_frames(video_path, scene)
        
        if not frames:
            return InterestScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {'error': 'no_frames'})
        
        # Calculate individual scores
        motion_score = self._calculate_motion_score(frames)
        face_score = self._calculate_face_score(frames)
        color_variance_score = self._calculate_color_variance_score(frames)
        composition_score = self._calculate_composition_score(frames)
        audio_peak_score = self._calculate_audio_peak_score(audio_peaks, scene) if audio_peaks else 0.0
        
        # Calculate weighted total score
        weights = self.config
        total_score = (
            motion_score * weights['motion_weight'] +
            face_score * weights['face_weight'] +
            color_variance_score * weights['color_variance_weight'] +
            composition_score * weights['composition_weight'] +
            audio_peak_score * weights['audio_peak_weight']
        )
        
        return InterestScore(
            total_score=total_score,
            motion_score=motion_score,
            face_score=face_score,
            color_variance_score=color_variance_score,
            composition_score=composition_score,
            audio_peak_score=audio_peak_score,
            metadata={
                'frame_count': len(frames),
                'scene_duration': scene.duration
            }
        )
    
    def _extract_scene_frames(self, video_path: str, scene: Scene, max_frames: int = 10) -> List[np.ndarray]:
        """Extract frames from scene for analysis"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        # Sample frames evenly throughout the scene
        frame_indices = np.linspace(scene.start_frame, scene.end_frame, 
                                  min(max_frames, int(scene.duration * 2)), dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def _calculate_motion_score(self, frames: List[np.ndarray]) -> float:
        """Calculate motion intensity score"""
        if len(frames) < 2:
            return 0.0
        
        motion_scores = []
        
        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, 
                cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7),
                None
            )[0]
            
            if flow is not None and len(flow) > 0:
                # Calculate motion magnitude
                motion_magnitude = np.mean(np.sqrt(flow[:, 0]**2 + flow[:, 1]**2))
                motion_scores.append(motion_magnitude)
        
        if not motion_scores:
            return 0.0
        
        avg_motion = np.mean(motion_scores)
        # Normalize to 0-1 range
        return min(avg_motion / 50.0, 1.0)
    
    def _calculate_face_score(self, frames: List[np.ndarray]) -> float:
        """Calculate face/character presence score"""
        if not self.face_cascade:
            return 0.0
        
        face_counts = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            face_counts.append(len(faces))
        
        if not face_counts:
            return 0.0
        
        avg_faces = np.mean(face_counts)
        # Normalize: 0 faces = 0.0, 2+ faces = 1.0
        return min(avg_faces / 2.0, 1.0)
    
    def _calculate_color_variance_score(self, frames: List[np.ndarray]) -> float:
        """Calculate color variance/visual complexity score"""
        variance_scores = []
        
        for frame in frames:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Calculate variance in hue and saturation channels
            h_var = np.var(hsv[:, :, 0])
            s_var = np.var(hsv[:, :, 1])
            
            # Combine variances
            total_var = (h_var + s_var) / 2
            variance_scores.append(total_var)
        
        if not variance_scores:
            return 0.0
        
        avg_variance = np.mean(variance_scores)
        # Normalize to 0-1 range
        return min(avg_variance / 1000.0, 1.0)
    
    def _calculate_composition_score(self, frames: List[np.ndarray]) -> float:
        """Calculate composition quality score using rule of thirds and edge detection"""
        composition_scores = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Rule of thirds analysis
            h, w = gray.shape
            thirds_h = [h//3, 2*h//3]
            thirds_w = [w//3, 2*w//3]
            
            # Check for interesting content at rule of thirds intersections
            roi_scores = []
            for th in thirds_h:
                for tw in thirds_w:
                    roi = gray[max(0, th-20):min(h, th+20), max(0, tw-20):min(w, tw+20)]
                    if roi.size > 0:
                        roi_variance = np.var(roi)
                        roi_scores.append(roi_variance)
            
            rule_of_thirds_score = np.mean(roi_scores) if roi_scores else 0
            
            # Combine scores
            composition_score = (edge_density * 0.6 + min(rule_of_thirds_score / 1000.0, 1.0) * 0.4)
            composition_scores.append(composition_score)
        
        return np.mean(composition_scores) if composition_scores else 0.0
    
    def _calculate_audio_peak_score(self, audio_peaks: List[float], scene: Scene) -> float:
        """Calculate audio peak score for the scene timeframe"""
        if not audio_peaks:
            return 0.0
        
        # Find audio peaks within scene timeframe
        scene_peaks = []
        for peak_time in audio_peaks:
            if scene.start_time <= peak_time <= scene.end_time:
                scene_peaks.append(peak_time)
        
        # Score based on number of peaks relative to scene duration
        peak_density = len(scene_peaks) / scene.duration
        return min(peak_density / 2.0, 1.0)  # Normalize: 2+ peaks per second = 1.0
    
    def rank_scenes_by_interest(self, scores: List[Tuple[Scene, InterestScore]]) -> List[Tuple[Scene, InterestScore]]:
        """Rank scenes by their interest scores"""
        return sorted(scores, key=lambda x: x[1].total_score, reverse=True)
    
    def filter_scenes_by_threshold(self, scores: List[Tuple[Scene, InterestScore]], 
                                 threshold: float = 0.5) -> List[Tuple[Scene, InterestScore]]:
        """Filter scenes that meet minimum interest threshold"""
        return [(scene, score) for scene, score in scores if score.total_score >= threshold]