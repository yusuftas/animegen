"""
Main Automation Pipeline
Orchestrates the entire anime shorts generation process
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from ..utils.config import get_config, AppConfig
from ..utils.logger import LoggerMixin
from ..scene_analysis.scene_extractor import SceneExtractor, Scene
from ..scene_analysis.interest_detector import InterestDetector, InterestScore
from ..scene_analysis.audio_analyzer import AudioAnalyzer, AudioPeak

@dataclass
class ProcessingResult:
    video_path: str
    scene: Scene
    interest_score: InterestScore
    audio_peaks: List[AudioPeak]
    metadata: Dict[str, Any]

class AnimeShortsPipeline(LoggerMixin):
    """Main pipeline for generating anime shorts"""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.scene_extractor = SceneExtractor(
            threshold=self.config.scene_analysis.threshold,
            min_scene_length=self.config.scene_analysis.min_scene_length,
            max_scene_length=self.config.scene_analysis.max_scene_length
        )
        self.interest_detector = InterestDetector(self._get_interest_config())
        self.audio_analyzer = AudioAnalyzer(self.config.audio.sample_rate)
        
        self.logger.info("Anime Shorts Pipeline initialized")
    
    def _get_interest_config(self) -> Dict[str, Any]:
        """Convert interest detection config to dictionary"""
        config = self.config.interest_detection
        return {
            'motion_threshold': config.motion_threshold,
            'face_detection_enabled': config.face_detection_enabled,
            'motion_weight': config.motion_weight,
            'face_weight': 0.2,  # Default face weight
            'color_variance_weight': config.color_variance_weight,
            'composition_weight': config.composition_weight,
            'audio_peak_weight': config.audio_peak_weight
        }
    
    def process_anime_episode(self, video_path: str, anime_name: str, 
                            output_dir: str = "./output", max_shorts: int = 5) -> List[str]:
        """
        Process a single anime episode and generate shorts
        
        Args:
            video_path: Path to the anime video file
            anime_name: Name of the anime
            output_dir: Directory to save generated shorts
            max_shorts: Maximum number of shorts to generate
            
        Returns:
            List of paths to generated short videos
        """
        self.logger.info(f"Processing anime episode: {video_path}")
        
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Extract scenes
            self.logger.info("Step 1: Extracting scenes...")
            scenes = self.scene_extractor.extract_scenes(video_path)
            self.logger.info(f"Extracted {len(scenes)} scenes")
            
            if not scenes:
                self.logger.warning("No suitable scenes found in video")
                return []
            
            # Step 2: Analyze audio
            self.logger.info("Step 2: Analyzing audio...")
            audio_analysis = self._analyze_audio(video_path)
            audio_peaks = audio_analysis.get('peaks', [])
            self.logger.info(f"Found {len(audio_peaks)} audio peaks")
            
            # Step 3: Score scenes for interest
            self.logger.info("Step 3: Scoring scenes for interest...")
            scored_scenes = self._score_scenes(video_path, scenes, audio_peaks)
            self.logger.info(f"Scored {len(scored_scenes)} scenes")
            
            # Step 4: Rank and filter scenes
            self.logger.info("Step 4: Ranking scenes...")
            ranked_scenes = self.interest_detector.rank_scenes_by_interest(scored_scenes)
            
            # Filter by quality threshold
            filtered_scenes = self.interest_detector.filter_scenes_by_threshold(
                ranked_scenes, 
                threshold=self.config.quality_control.min_engagement_score
            )
            
            self.logger.info(f"Selected {len(filtered_scenes)} high-quality scenes")
            
            # Step 5: Generate shorts (placeholder for now)
            shorts_generated = []
            selected_scenes = filtered_scenes[:max_shorts]
            
            for i, (scene, score) in enumerate(selected_scenes, 1):
                self.logger.info(f"Processing scene {i}/{len(selected_scenes)} "
                               f"(score: {score.total_score:.3f})")
                
                # For now, just create a placeholder result
                # In future phases, this will include script generation, video editing, etc.
                short_info = {
                    'scene_id': i,
                    'start_time': scene.start_time,
                    'end_time': scene.end_time,
                    'duration': scene.duration,
                    'interest_score': score.total_score,
                    'anime_name': anime_name,
                    'source_video': video_path
                }
                
                # Create placeholder output filename
                output_filename = f"{anime_name}_short_{i:02d}.json"
                output_file = output_path / output_filename
                
                # Save scene info (placeholder for actual video generation)
                import json
                with open(output_file, 'w') as f:
                    json.dump(short_info, f, indent=2)
                
                shorts_generated.append(str(output_file))
            
            self.logger.info(f"Successfully processed {len(shorts_generated)} shorts")
            return shorts_generated
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            raise
    
    def _analyze_audio(self, video_path: str) -> Dict[str, Any]:
        """Analyze audio track of the video"""
        try:
            # First, try to extract audio from video
            audio_path = None
            temp_audio = False
            
            # Check if video has audio or if separate audio file exists
            audio_candidates = [
                str(Path(video_path).with_suffix('.wav')),
                str(Path(video_path).with_suffix('.mp3')),
                str(Path(video_path).with_suffix('.aac'))
            ]
            
            for candidate in audio_candidates:
                if Path(candidate).exists():
                    audio_path = candidate
                    break
            
            if audio_path is None:
                # Extract audio from video
                try:
                    audio_path = self.audio_analyzer.extract_audio_from_video(video_path)
                    temp_audio = True
                except Exception as e:
                    self.logger.warning(f"Could not extract audio: {e}")
                    return {'peaks': [], 'metadata': {'error': 'audio_extraction_failed'}}
            
            # Analyze audio
            analysis_result = self.audio_analyzer.analyze_audio_file(audio_path)
            
            # Clean up temporary audio file
            if temp_audio and audio_path and Path(audio_path).exists():
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Audio analysis failed: {e}")
            return {'peaks': [], 'metadata': {'error': str(e)}}
    
    def _score_scenes(self, video_path: str, scenes: List[Scene], 
                     audio_peaks: List[AudioPeak]) -> List[Tuple[Scene, InterestScore]]:
        """Score all scenes for interest level"""
        scored_scenes = []
        
        # Convert audio peaks to simple timestamps for compatibility
        peak_times = [peak.timestamp for peak in audio_peaks] if audio_peaks else []
        
        for scene in scenes:
            try:
                score = self.interest_detector.calculate_interest_score(
                    video_path, scene, peak_times
                )
                scored_scenes.append((scene, score))
                
            except Exception as e:
                self.logger.warning(f"Failed to score scene {scene.start_time}-{scene.end_time}: {e}")
                # Create a zero score for failed scenes
                zero_score = InterestScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {'error': str(e)})
                scored_scenes.append((scene, zero_score))
        
        return scored_scenes
    
    def batch_process_episodes(self, video_directory: str, anime_name: str, 
                             output_dir: str = "./output") -> Dict[str, List[str]]:
        """
        Process multiple episodes in a directory
        
        Args:
            video_directory: Directory containing anime episodes
            anime_name: Name of the anime series
            output_dir: Directory to save generated shorts
            
        Returns:
            Dictionary mapping episode paths to generated shorts
        """
        self.logger.info(f"Batch processing episodes in: {video_directory}")
        
        video_dir = Path(video_directory)
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_directory}")
        
        # Find video files
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f"*{ext}"))
            video_files.extend(video_dir.glob(f"*{ext.upper()}"))
        
        if not video_files:
            self.logger.warning(f"No video files found in {video_directory}")
            return {}
        
        self.logger.info(f"Found {len(video_files)} video files to process")
        
        results = {}
        for video_file in sorted(video_files):
            try:
                self.logger.info(f"Processing episode: {video_file.name}")
                shorts = self.process_anime_episode(
                    str(video_file), anime_name, output_dir
                )
                results[str(video_file)] = shorts
                
            except Exception as e:
                self.logger.error(f"Failed to process {video_file}: {e}")
                results[str(video_file)] = []
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline configuration and capabilities"""
        return {
            'config': {
                'scene_analysis': {
                    'threshold': self.config.scene_analysis.threshold,
                    'min_scene_length': self.config.scene_analysis.min_scene_length,
                    'max_scene_length': self.config.scene_analysis.max_scene_length
                },
                'quality_thresholds': {
                    'min_engagement_score': self.config.quality_control.min_engagement_score,
                    'min_retention_prediction': self.config.quality_control.min_retention_prediction
                }
            },
            'capabilities': {
                'scene_detection': True,
                'interest_scoring': True,
                'audio_analysis': True,
                'script_generation': False,  # Not implemented yet
                'video_editing': False,      # Not implemented yet
                'youtube_upload': False      # Not implemented yet
            }
        }