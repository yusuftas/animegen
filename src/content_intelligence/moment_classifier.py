"""
Moment Classification System
Classifies anime scenes into different types for targeted commentary generation
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import pickle
from pathlib import Path
import logging

from ..scene_analysis.scene_extractor import Scene
from ..scene_analysis.interest_detector import InterestScore
from ..utils.logger import LoggerMixin

class MomentType(Enum):
    ACTION_SEQUENCE = "action_sequence"
    EMOTIONAL_MOMENT = "emotional_moment"
    COMEDY_SCENE = "comedy_scene"
    PLOT_REVELATION = "plot_revelation"
    CHARACTER_INTRODUCTION = "character_introduction"
    TRANSFORMATION_SCENE = "transformation_scene"
    DRAMATIC_PAUSE = "dramatic_pause"
    DIALOGUE_HEAVY = "dialogue_heavy"
    FLASHBACK = "flashback"
    OPENING_ENDING = "opening_ending"
    FIGHT_SCENE = "fight_scene"
    ROMANTIC_MOMENT = "romantic_moment"
    WORLD_BUILDING = "world_building"
    SLICE_OF_LIFE = "slice_of_life"
    UNKNOWN = "unknown"

@dataclass
class MomentClassification:
    primary_type: MomentType
    confidence: float
    secondary_types: List[Tuple[MomentType, float]]
    features: Dict[str, float]
    metadata: Dict[str, Any]

class MomentClassifier(LoggerMixin):
    """Classifies anime moments into different scene types"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.feature_weights = self._get_default_weights()
        self.classification_rules = self._build_classification_rules()
        
        if model_path and Path(model_path).exists():
            self._load_model()
        
        self.logger.info("Moment Classifier initialized")
    
    def _get_default_weights(self) -> Dict[str, Dict[str, float]]:
        """Default feature weights for each moment type"""
        return {
            MomentType.ACTION_SEQUENCE.value: {
                'motion_intensity': 0.4,
                'color_variance': 0.3,
                'audio_activity': 0.2,
                'edge_density': 0.1
            },
            MomentType.EMOTIONAL_MOMENT.value: {
                'face_presence': 0.4,
                'motion_intensity': -0.2,  # Usually low motion
                'color_warmth': 0.2,
                'audio_activity': 0.2
            },
            MomentType.COMEDY_SCENE.value: {
                'face_presence': 0.3,
                'color_variance': 0.2,
                'motion_intensity': 0.2,
                'audio_activity': 0.3
            },
            MomentType.FIGHT_SCENE.value: {
                'motion_intensity': 0.5,
                'edge_density': 0.3,
                'color_variance': 0.1,
                'audio_activity': 0.1
            },
            MomentType.DRAMATIC_PAUSE.value: {
                'motion_intensity': -0.4,  # Very low motion
                'face_presence': 0.3,
                'color_temperature': 0.1,
                'audio_activity': -0.2
            },
            MomentType.TRANSFORMATION_SCENE.value: {
                'color_variance': 0.4,
                'motion_intensity': 0.3,
                'brightness_change': 0.2,
                'audio_activity': 0.1
            }
        }
    
    def _build_classification_rules(self) -> Dict[str, Dict[str, Any]]:
        """Build rule-based classification criteria"""
        return {
            MomentType.ACTION_SEQUENCE.value: {
                'motion_threshold': 0.7,
                'duration_range': (2, 15),
                'audio_activity_min': 0.5
            },
            MomentType.EMOTIONAL_MOMENT.value: {
                'face_presence_min': 0.6,
                'motion_threshold_max': 0.3,
                'duration_range': (3, 20)
            },
            MomentType.FIGHT_SCENE.value: {
                'motion_threshold': 0.8,
                'edge_density_min': 0.6,
                'duration_range': (5, 30)
            },
            MomentType.DRAMATIC_PAUSE.value: {
                'motion_threshold_max': 0.2,
                'duration_range': (1, 8),
                'audio_activity_max': 0.3
            },
            MomentType.COMEDY_SCENE.value: {
                'face_presence_min': 0.4,
                'audio_activity_min': 0.4,
                'duration_range': (2, 12)
            }
        }
    
    def classify_moment(self, video_path: str, scene: Scene, 
                       interest_score: InterestScore, 
                       context: Dict[str, Any] = None) -> MomentClassification:
        """Classify a moment/scene into its type"""
        
        # Extract features from the scene
        features = self._extract_scene_features(video_path, scene, interest_score)
        
        # Add context features if available
        if context:
            features.update(self._extract_context_features(context))
        
        # Apply rule-based classification
        rule_scores = self._apply_classification_rules(features, scene)
        
        # Apply ML model if available
        ml_scores = {}
        if self.model:
            ml_scores = self._apply_ml_model(features)
        
        # Combine scores
        final_scores = self._combine_scores(rule_scores, ml_scores)
        
        # Determine primary and secondary classifications
        sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_type = MomentType(sorted_scores[0][0])
        confidence = sorted_scores[0][1]
        
        secondary_types = [
            (MomentType(moment_type), score) 
            for moment_type, score in sorted_scores[1:4]
            if score > 0.3
        ]
        
        classification = MomentClassification(
            primary_type=primary_type,
            confidence=confidence,
            secondary_types=secondary_types,
            features=features,
            metadata={
                'scene_duration': scene.duration,
                'rule_scores': rule_scores,
                'ml_scores': ml_scores
            }
        )
        
        return classification
    
    def _extract_scene_features(self, video_path: str, scene: Scene, 
                               interest_score: InterestScore) -> Dict[str, float]:
        """Extract features from scene for classification"""
        features = {
            'motion_intensity': interest_score.motion_score,
            'face_presence': interest_score.face_score,
            'color_variance': interest_score.color_variance_score,
            'composition_quality': interest_score.composition_score,
            'audio_activity': interest_score.audio_peak_score,
            'scene_duration': scene.duration
        }
        
        # Extract additional visual features
        additional_features = self._extract_visual_features(video_path, scene)
        features.update(additional_features)
        
        return features
    
    def _extract_visual_features(self, video_path: str, scene: Scene) -> Dict[str, float]:
        """Extract additional visual features from scene"""
        features = {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return features
            
            # Sample frames from the scene
            frame_indices = np.linspace(scene.start_frame, scene.end_frame, 5, dtype=int)
            frames = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            
            cap.release()
            
            if not frames:
                return features
            
            # Calculate additional features
            features['edge_density'] = self._calculate_edge_density(frames)
            features['brightness_variance'] = self._calculate_brightness_variance(frames)
            features['color_temperature'] = self._calculate_color_temperature(frames)
            features['contrast_level'] = self._calculate_contrast_level(frames)
            features['brightness_change'] = self._calculate_brightness_change(frames)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract visual features: {e}")
        
        return features
    
    def _calculate_edge_density(self, frames: List[np.ndarray]) -> float:
        """Calculate edge density (complexity) in frames"""
        edge_densities = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            edge_densities.append(density)
        
        return np.mean(edge_densities)
    
    def _calculate_brightness_variance(self, frames: List[np.ndarray]) -> float:
        """Calculate variance in brightness across frames"""
        brightness_values = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightness_values.append(brightness)
        
        return np.var(brightness_values) / 255.0  # Normalize
    
    def _calculate_color_temperature(self, frames: List[np.ndarray]) -> float:
        """Calculate average color temperature (warm vs cool)"""
        temperatures = []
        
        for frame in frames:
            # Simple heuristic: ratio of warm (red/yellow) to cool (blue) colors
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Warm colors (red, orange, yellow)
            warm_mask = ((hsv[:,:,0] < 30) | (hsv[:,:,0] > 150)) & (hsv[:,:,1] > 50)
            
            # Cool colors (blue, cyan)
            cool_mask = (hsv[:,:,0] > 90) & (hsv[:,:,0] < 150) & (hsv[:,:,1] > 50)
            
            warm_pixels = np.sum(warm_mask)
            cool_pixels = np.sum(cool_mask)
            
            if warm_pixels + cool_pixels > 0:
                temperature = warm_pixels / (warm_pixels + cool_pixels)
            else:
                temperature = 0.5
            
            temperatures.append(temperature)
        
        return np.mean(temperatures)
    
    def _calculate_contrast_level(self, frames: List[np.ndarray]) -> float:
        """Calculate average contrast level"""
        contrasts = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray) / 255.0  # Normalize
            contrasts.append(contrast)
        
        return np.mean(contrasts)
    
    def _calculate_brightness_change(self, frames: List[np.ndarray]) -> float:
        """Calculate change in brightness throughout scene"""
        if len(frames) < 2:
            return 0.0
        
        brightness_values = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightness_values.append(brightness)
        
        # Calculate max change
        max_change = (max(brightness_values) - min(brightness_values)) / 255.0
        return max_change
    
    def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from anime context"""
        features = {}
        
        # Genre-based features
        genres = context.get('genres', [])
        features['is_action_genre'] = 1.0 if any(g.lower() in ['action', 'shounen'] for g in genres) else 0.0
        features['is_romance_genre'] = 1.0 if any(g.lower() in ['romance', 'shoujo'] for g in genres) else 0.0
        features['is_comedy_genre'] = 1.0 if any(g.lower() in ['comedy', 'slice of life'] for g in genres) else 0.0
        
        # Episode timing features
        episode = context.get('episode_number', 1)
        features['is_first_episode'] = 1.0 if episode == 1 else 0.0
        features['is_final_episode'] = 1.0 if episode == context.get('total_episodes', 12) else 0.0
        
        return features
    
    def _apply_classification_rules(self, features: Dict[str, float], 
                                   scene: Scene) -> Dict[str, float]:
        """Apply rule-based classification"""
        scores = {}
        
        for moment_type, rules in self.classification_rules.items():
            score = 0.0
            
            # Motion threshold rules
            if 'motion_threshold' in rules:
                if features.get('motion_intensity', 0) >= rules['motion_threshold']:
                    score += 0.3
            
            if 'motion_threshold_max' in rules:
                if features.get('motion_intensity', 1) <= rules['motion_threshold_max']:
                    score += 0.3
            
            # Face presence rules
            if 'face_presence_min' in rules:
                if features.get('face_presence', 0) >= rules['face_presence_min']:
                    score += 0.2
            
            # Audio activity rules
            if 'audio_activity_min' in rules:
                if features.get('audio_activity', 0) >= rules['audio_activity_min']:
                    score += 0.2
            
            if 'audio_activity_max' in rules:
                if features.get('audio_activity', 1) <= rules['audio_activity_max']:
                    score += 0.2
            
            # Duration rules
            if 'duration_range' in rules:
                min_dur, max_dur = rules['duration_range']
                if min_dur <= scene.duration <= max_dur:
                    score += 0.1
            
            # Edge density rules
            if 'edge_density_min' in rules:
                if features.get('edge_density', 0) >= rules['edge_density_min']:
                    score += 0.2
            
            scores[moment_type] = min(score, 1.0)
        
        return scores
    
    def _apply_ml_model(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply ML model for classification (placeholder)"""
        # This would use a trained model if available
        # For now, return empty scores
        return {}
    
    def _combine_scores(self, rule_scores: Dict[str, float], 
                       ml_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine rule-based and ML scores"""
        combined = {}
        
        all_types = set(rule_scores.keys()) | set(ml_scores.keys())
        
        for moment_type in all_types:
            rule_score = rule_scores.get(moment_type, 0.0)
            ml_score = ml_scores.get(moment_type, 0.0)
            
            # Weight: 70% rules, 30% ML (if available)
            if ml_score > 0:
                combined[moment_type] = 0.7 * rule_score + 0.3 * ml_score
            else:
                combined[moment_type] = rule_score
        
        # Add default unknown classification
        combined[MomentType.UNKNOWN.value] = 0.1
        
        return combined
    
    def batch_classify(self, scenes_data: List[Tuple[str, Scene, InterestScore, Dict[str, Any]]]) -> List[MomentClassification]:
        """Classify multiple scenes in batch"""
        classifications = []
        
        for video_path, scene, interest_score, context in scenes_data:
            try:
                classification = self.classify_moment(video_path, scene, interest_score, context)
                classifications.append(classification)
            except Exception as e:
                self.logger.error(f"Failed to classify scene {scene.start_time}-{scene.end_time}: {e}")
                # Add default classification
                fallback = MomentClassification(
                    primary_type=MomentType.UNKNOWN,
                    confidence=0.0,
                    secondary_types=[],
                    features={},
                    metadata={'error': str(e)}
                )
                classifications.append(fallback)
        
        return classifications
    
    def get_classification_stats(self, classifications: List[MomentClassification]) -> Dict[str, Any]:
        """Get statistics about classifications"""
        type_counts = {}
        confidence_scores = []
        
        for classification in classifications:
            moment_type = classification.primary_type.value
            type_counts[moment_type] = type_counts.get(moment_type, 0) + 1
            confidence_scores.append(classification.confidence)
        
        return {
            'type_distribution': type_counts,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'total_classified': len(classifications),
            'high_confidence_count': sum(1 for c in classifications if c.confidence > 0.7)
        }
    
    def _load_model(self):
        """Load trained ML model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.logger.info(f"Loaded ML model from {self.model_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load ML model: {e}")
    
    def save_model(self, model_path: str):
        """Save trained ML model"""
        if self.model:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            self.logger.info(f"Saved ML model to {model_path}")