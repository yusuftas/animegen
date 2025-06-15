"""
Audio-Visual Synchronization Engine

Provides beat detection, audio analysis, and synchronization of visual effects
to audio beats and musical elements for enhanced anime video editing.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import logging


# MoviePy 1.x imports (fallback)
from moviepy.editor import VideoFileClip, CompositeVideoClip, AudioFileClip, concatenate_videoclips
from moviepy.video.fx import speedx

# Optional librosa import for advanced audio analysis
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available. Some audio sync features will be limited.")

logger = logging.getLogger(__name__)


class AudioSyncEngine:
    """Engine for synchronizing visual effects to audio elements."""
    
    def __init__(self):
        """Initialize the audio synchronization engine."""
        self.beat_cache = {}
        self.tempo_cache = {}
        self.onset_cache = {}
        
    def extract_beats(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """
        Extract beat timestamps from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (beat_times, tempo)
        """
        try:
            if audio_path in self.beat_cache:
                return self.beat_cache[audio_path], self.tempo_cache[audio_path]
            
            if not LIBROSA_AVAILABLE:
                logger.warning("librosa not available, using simple beat detection")
                return self._simple_beat_detection(audio_path)
            
            # Load audio with librosa
            y, sr = librosa.load(audio_path)
            
            # Extract tempo and beats
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
            
            # Cache results
            self.beat_cache[audio_path] = beat_times
            self.tempo_cache[audio_path] = tempo
            
            logger.info(f"Extracted {len(beat_times)} beats at tempo {tempo:.1f} BPM")
            return beat_times, tempo
            
        except Exception as e:
            logger.error(f"Error extracting beats: {e}")
            return np.array([]), 120.0  # Default empty beats at 120 BPM
    
    def _simple_beat_detection(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """
        Simple beat detection without librosa.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (beat_times, tempo)
        """
        try:
            # Load audio using moviepy
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            
            # Simple assumption: beats every 0.5 seconds (120 BPM)
            tempo = 120.0
            beat_interval = 60.0 / tempo
            beat_times = np.arange(0, duration, beat_interval)
            
            logger.info(f"Using simple beat detection: {len(beat_times)} beats")
            return beat_times, tempo
            
        except Exception as e:
            logger.error(f"Error in simple beat detection: {e}")
            return np.array([]), 120.0
    
    def extract_onsets(self, audio_path: str) -> np.ndarray:
        """
        Extract onset timestamps (sudden changes in audio).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Array of onset timestamps
        """
        try:
            if audio_path in self.onset_cache:
                return self.onset_cache[audio_path]
            
            if not LIBROSA_AVAILABLE:
                logger.warning("librosa not available, onset detection limited")
                # Fallback to beat times
                beat_times, _ = self.extract_beats(audio_path)
                return beat_times
            
            # Load audio
            y, sr = librosa.load(audio_path)
            
            # Detect onsets
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
            
            # Cache results
            self.onset_cache[audio_path] = onset_times
            
            logger.info(f"Extracted {len(onset_times)} onsets")
            return onset_times
            
        except Exception as e:
            logger.error(f"Error extracting onsets: {e}")
            return np.array([])
    
    def analyze_audio_energy(self, audio_path: str, frame_length: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze audio energy levels over time.
        
        Args:
            audio_path: Path to audio file
            frame_length: Length of analysis frames
            
        Returns:
            Tuple of (time_stamps, energy_levels)
        """
        try:
            if not LIBROSA_AVAILABLE:
                logger.warning("librosa not available, energy analysis limited")
                return np.array([]), np.array([])
            
            # Load audio
            y, sr = librosa.load(audio_path)
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=frame_length//4)[0]
            
            # Convert to time stamps
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=frame_length//4)
            
            logger.info(f"Analyzed energy for {len(times)} time frames")
            return times, rms
            
        except Exception as e:
            logger.error(f"Error analyzing audio energy: {e}")
            return np.array([]), np.array([])
    
    def sync_effects_to_beats(self, clip: VideoFileClip, audio_path: str, 
                             effect_func: Callable) -> VideoFileClip:
        """
        Synchronize visual effects to audio beats.
        
        Args:
            clip: Input video clip
            audio_path: Path to audio file for beat detection
            effect_func: Function to apply effect at each beat
            
        Returns:
            Video clip with beat-synchronized effects
        """
        try:
            beat_times, tempo = self.extract_beats(audio_path)
            
            if len(beat_times) == 0:
                logger.warning("No beats detected, returning original clip")
                return clip
            
            # Apply effects at each beat
            effects_clip = clip
            for beat_time in beat_times:
                if beat_time < clip.duration:
                    effects_clip = effect_func(effects_clip, beat_time)
            
            logger.info(f"Applied beat-synchronized effects at {len(beat_times)} timestamps")
            return effects_clip
            
        except Exception as e:
            logger.error(f"Error syncing effects to beats: {e}")
            return clip
    
    def create_beat_flash(self, clip: VideoFileClip, beat_time: float, 
                         intensity: float = 0.5, duration: float = 0.1) -> VideoFileClip:
        """
        Create flash effect synchronized to a beat.
        
        Args:
            clip: Input video clip
            beat_time: Time of beat to sync to
            intensity: Intensity of flash effect
            duration: Duration of flash
            
        Returns:
            Video clip with beat flash effect
        """
        try:
            def flash_func(get_frame, t):
                frame = get_frame(t)
                if abs(t - beat_time) < duration / 2:
                    # Calculate flash intensity based on distance from beat
                    flash_intensity = intensity * (1 - abs(t - beat_time) / (duration / 2))
                    
                    # Create white overlay
                    white_overlay = np.ones_like(frame) * 255
                    
                    # Blend with original frame
                    result = cv2.addWeighted(frame, 1 - flash_intensity, 
                                           white_overlay, flash_intensity, 0)
                    return result
                return frame
            
            return clip.fl(flash_func)
            
        except Exception as e:
            logger.error(f"Error creating beat flash: {e}")
            return clip
    
    def create_beat_zoom(self, clip: VideoFileClip, beat_time: float,
                        zoom_factor: float = 1.2, duration: float = 0.2) -> VideoFileClip:
        """
        Create zoom effect synchronized to a beat.
        
        Args:
            clip: Input video clip
            beat_time: Time of beat to sync to
            zoom_factor: Maximum zoom level
            duration: Duration of zoom effect
            
        Returns:
            Video clip with beat zoom effect
        """
        try:
            def zoom_func(get_frame, t):
                frame = get_frame(t)
                if abs(t - beat_time) < duration / 2:
                    # Calculate zoom intensity
                    intensity = 1 - abs(t - beat_time) / (duration / 2)
                    current_zoom = 1 + (zoom_factor - 1) * intensity
                    
                    h, w = frame.shape[:2]
                    center_x, center_y = w // 2, h // 2
                    
                    # Create zoom transformation
                    M = cv2.getRotationMatrix2D((center_x, center_y), 0, current_zoom)
                    
                    # Apply transformation
                    result = cv2.warpAffine(frame, M, (w, h))
                    return result
                return frame
            
            return clip.fl(zoom_func)
            
        except Exception as e:
            logger.error(f"Error creating beat zoom: {e}")
            return clip
    
    def create_beat_color_pulse(self, clip: VideoFileClip, beat_time: float,
                               color_shift: Tuple[float, float, float] = (1.2, 1.0, 1.0),
                               duration: float = 0.15) -> VideoFileClip:
        """
        Create color pulse effect synchronized to a beat.
        
        Args:
            clip: Input video clip
            beat_time: Time of beat to sync to
            color_shift: RGB multipliers for color shift
            duration: Duration of color pulse
            
        Returns:
            Video clip with beat color pulse
        """
        try:
            def color_pulse_func(get_frame, t):
                frame = get_frame(t)
                if abs(t - beat_time) < duration / 2:
                    # Calculate pulse intensity
                    intensity = 1 - abs(t - beat_time) / (duration / 2)
                    
                    # Apply color shift
                    result = frame.astype(np.float32)
                    for i in range(3):
                        multiplier = 1 + (color_shift[i] - 1) * intensity
                        result[:, :, i] = result[:, :, i] * multiplier
                    
                    result = np.clip(result, 0, 255).astype(np.uint8)
                    return result
                return frame
            
            return clip.fl(color_pulse_func)
            
        except Exception as e:
            logger.error(f"Error creating beat color pulse: {e}")
            return clip
    
    def sync_to_energy_levels(self, clip: VideoFileClip, audio_path: str,
                             effect_intensity_func: Callable) -> VideoFileClip:
        """
        Synchronize effect intensity to audio energy levels.
        
        Args:
            clip: Input video clip
            audio_path: Path to audio file
            effect_intensity_func: Function that takes (frame, time, energy_level)
            
        Returns:
            Video clip with energy-synchronized effects
        """
        try:
            times, energy_levels = self.analyze_audio_energy(audio_path)
            
            if len(times) == 0:
                logger.warning("No energy data available")
                return clip
            
            # Normalize energy levels
            max_energy = np.max(energy_levels) if len(energy_levels) > 0 else 1.0
            normalized_energy = energy_levels / max_energy
            
            def energy_sync_func(get_frame, t):
                frame = get_frame(t)
                
                # Find closest energy level for current time
                closest_idx = np.argmin(np.abs(times - t))
                if closest_idx < len(normalized_energy):
                    energy_level = normalized_energy[closest_idx]
                    return effect_intensity_func(frame, t, energy_level)
                
                return frame
            
            return clip.fl(energy_sync_func)
            
        except Exception as e:
            logger.error(f"Error syncing to energy levels: {e}")
            return clip
    
    def create_rhythm_based_cuts(self, clips: List[VideoFileClip], audio_path: str,
                                cut_on_beats: bool = True) -> VideoFileClip:
        """
        Create cuts between clips based on audio rhythm.
        
        Args:
            clips: List of video clips to cut between
            audio_path: Path to audio file for timing
            cut_on_beats: Whether to cut on beats or onsets
            
        Returns:
            Assembled video with rhythm-based cuts
        """
        try:
            if cut_on_beats:
                cut_times, tempo = self.extract_beats(audio_path)
            else:
                cut_times = self.extract_onsets(audio_path)
                tempo = 120.0  # Default tempo
            
            if len(cut_times) == 0 or len(clips) == 0:
                logger.warning("No cut times or clips available")
                return clips[0] if clips else None
            
            # Calculate duration for each segment
            total_duration = clips[0].duration  # Assume all clips have similar duration
            segments = []
            
            for i, cut_time in enumerate(cut_times):
                if i >= len(clips):
                    break
                
                # Determine segment duration
                if i < len(cut_times) - 1:
                    duration = min(cut_times[i + 1] - cut_time, total_duration - cut_time)
                else:
                    duration = total_duration - cut_time
                
                if duration > 0.1:  # Minimum segment duration
                    segment = clips[i % len(clips)].subclip(0, duration)
                    segments.append(segment)
            
            if segments:
                result = concatenate_videoclips(segments)
                logger.info(f"Created rhythm-based cuts with {len(segments)} segments")
                return result
            else:
                return clips[0]
                
        except Exception as e:
            logger.error(f"Error creating rhythm-based cuts: {e}")
            return clips[0] if clips else None
    
    def apply_multi_beat_effects(self, clip: VideoFileClip, audio_path: str,
                                effects_config: Dict[str, Any]) -> VideoFileClip:
        """
        Apply multiple effects synchronized to different beat patterns.
        
        Args:
            clip: Input video clip
            audio_path: Path to audio file
            effects_config: Configuration for different effects
            
        Returns:
            Video clip with multiple synchronized effects
        """
        try:
            beat_times, tempo = self.extract_beats(audio_path)
            
            if len(beat_times) == 0:
                return clip
            
            result_clip = clip
            
            # Apply different effects based on configuration
            for effect_name, config in effects_config.items():
                interval = config.get('interval', 1)  # Every nth beat
                selected_beats = beat_times[::interval]
                
                if effect_name == 'flash':
                    for beat_time in selected_beats:
                        result_clip = self.create_beat_flash(
                            result_clip, beat_time, 
                            config.get('intensity', 0.5),
                            config.get('duration', 0.1)
                        )
                
                elif effect_name == 'zoom':
                    for beat_time in selected_beats:
                        result_clip = self.create_beat_zoom(
                            result_clip, beat_time,
                            config.get('zoom_factor', 1.2),
                            config.get('duration', 0.2)
                        )
                
                elif effect_name == 'color_pulse':
                    for beat_time in selected_beats:
                        result_clip = self.create_beat_color_pulse(
                            result_clip, beat_time,
                            config.get('color_shift', (1.2, 1.0, 1.0)),
                            config.get('duration', 0.15)
                        )
            
            logger.info(f"Applied {len(effects_config)} beat-synchronized effects")
            return result_clip
            
        except Exception as e:
            logger.error(f"Error applying multi-beat effects: {e}")
            return clip
    
    def get_audio_sync_presets(self) -> Dict[str, Any]:
        """
        Get predefined audio synchronization presets.
        
        Returns:
            Dictionary of preset configurations
        """
        return {
            'high_energy': {
                'flash': {'interval': 1, 'intensity': 0.6, 'duration': 0.1},
                'zoom': {'interval': 2, 'zoom_factor': 1.3, 'duration': 0.2},
                'color_pulse': {'interval': 1, 'color_shift': (1.3, 1.1, 1.0), 'duration': 0.15}
            },
            'dramatic': {
                'flash': {'interval': 4, 'intensity': 0.8, 'duration': 0.2},
                'zoom': {'interval': 2, 'zoom_factor': 1.5, 'duration': 0.3}
            },
            'subtle': {
                'color_pulse': {'interval': 2, 'color_shift': (1.1, 1.05, 1.0), 'duration': 0.2},
                'zoom': {'interval': 4, 'zoom_factor': 1.1, 'duration': 0.15}
            },
            'intense_action': {
                'flash': {'interval': 1, 'intensity': 0.7, 'duration': 0.08},
                'zoom': {'interval': 1, 'zoom_factor': 1.4, 'duration': 0.15},
                'color_pulse': {'interval': 1, 'color_shift': (1.4, 1.0, 1.2), 'duration': 0.12}
            }
        }