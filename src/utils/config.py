"""
Configuration Management Module
Handles loading and managing configuration for the anime shorts generator
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class SceneAnalysisConfig:
    threshold: float = 30.0
    min_scene_length: float = 2.0
    max_scene_length: float = 15.0

@dataclass
class InterestDetectionConfig:
    motion_threshold: float = 0.3
    face_detection_enabled: bool = True
    color_variance_weight: float = 0.2
    motion_weight: float = 0.4
    composition_weight: float = 0.2
    audio_peak_weight: float = 0.2

@dataclass
class AudioConfig:
    sample_rate: int = 44100
    original_audio_volume: float = 0.3
    background_music_volume: float = 0.2
    voiceover_volume: float = 1.0

@dataclass
class VideoConfig:
    output_resolution: list = field(default_factory=lambda: [1080, 1920])
    fps: int = 30
    quality: str = "high"
    format: str = "mp4"

@dataclass
class ScriptGenerationConfig:
    max_script_length: int = 150
    min_script_length: int = 50
    styles: list = field(default_factory=lambda: ["analytical", "trivia_focused", "enthusiastic", "mysterious", "comparison"])

@dataclass
class YouTubeConfig:
    max_title_length: int = 100
    max_description_length: int = 5000
    default_tags: list = field(default_factory=lambda: ["anime", "shorts", "animation", "manga", "otaku"])

@dataclass
class QualityControlConfig:
    min_retention_prediction: float = 0.4
    max_copyright_risk: float = 0.3
    min_engagement_score: float = 0.5

@dataclass
class AppConfig:
    scene_analysis: SceneAnalysisConfig = field(default_factory=SceneAnalysisConfig)
    interest_detection: InterestDetectionConfig = field(default_factory=InterestDetectionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    script_generation: ScriptGenerationConfig = field(default_factory=ScriptGenerationConfig)
    youtube: YouTubeConfig = field(default_factory=YouTubeConfig)
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)

class ConfigManager:
    """Manages application configuration loading and access"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self._config: Optional[AppConfig] = None
    
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations"""
        possible_paths = [
            "config/config.yaml",
            "config.yaml",
            "config/config.yml",
            "config.yml"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        # If no config file found, create default
        default_path = "config/config.yaml"
        Path(default_path).parent.mkdir(exist_ok=True)
        self._create_default_config(default_path)
        return default_path
    
    def _create_default_config(self, path: str) -> None:
        """Create a default configuration file"""
        default_config = {
            'scene_analysis': {
                'threshold': 30.0,
                'min_scene_length': 2.0,
                'max_scene_length': 15.0
            },
            'interest_detection': {
                'motion_threshold': 0.3,
                'face_detection_enabled': True,
                'color_variance_weight': 0.2,
                'motion_weight': 0.4,
                'composition_weight': 0.2,
                'audio_peak_weight': 0.2
            },
            'audio': {
                'sample_rate': 44100,
                'original_audio_volume': 0.3,
                'background_music_volume': 0.2,
                'voiceover_volume': 1.0
            },
            'video': {
                'output_resolution': [1080, 1920],
                'fps': 30,
                'quality': "high",
                'format': "mp4"
            },
            'script_generation': {
                'max_script_length': 150,
                'min_script_length': 50,
                'styles': ["analytical", "trivia_focused", "enthusiastic", "mysterious", "comparison"]
            },
            'youtube': {
                'max_title_length': 100,
                'max_description_length': 5000,
                'default_tags': ["anime", "shorts", "animation", "manga", "otaku"]
            },
            'quality_control': {
                'min_retention_prediction': 0.4,
                'max_copyright_risk': 0.3,
                'min_engagement_score': 0.5
            }
        }
        
        with open(path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    def load_config(self) -> AppConfig:
        """Load configuration from file"""
        if self._config is not None:
            return self._config
        
        try:
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Create config objects
            scene_analysis = SceneAnalysisConfig(**config_dict.get('scene_analysis', {}))
            interest_detection = InterestDetectionConfig(**config_dict.get('interest_detection', {}))
            audio = AudioConfig(**config_dict.get('audio', {}))
            video = VideoConfig(**config_dict.get('video', {}))
            script_generation = ScriptGenerationConfig(**config_dict.get('script_generation', {}))
            youtube = YouTubeConfig(**config_dict.get('youtube', {}))
            quality_control = QualityControlConfig(**config_dict.get('quality_control', {}))
            
            self._config = AppConfig(
                scene_analysis=scene_analysis,
                interest_detection=interest_detection,
                audio=audio,
                video=video,
                script_generation=script_generation,
                youtube=youtube,
                quality_control=quality_control
            )
            
            return self._config
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {self.config_path}: {e}")
    
    def get_config(self) -> AppConfig:
        """Get loaded configuration (loads if not already loaded)"""
        return self.load_config()
    
    def reload_config(self) -> AppConfig:
        """Reload configuration from file"""
        self._config = None
        return self.load_config()

# Global config manager instance
_config_manager = None

def get_config(config_path: Optional[str] = None) -> AppConfig:
    """Get global configuration instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager.get_config()

def reload_config() -> AppConfig:
    """Reload global configuration"""
    global _config_manager
    if _config_manager is not None:
        return _config_manager.reload_config()
    else:
        _config_manager = ConfigManager()
        return _config_manager.get_config()