"""
Motion and Impact Effects Engine for Anime Video Processing

Provides dynamic motion effects including speed ramping, zoom punches, camera shake,
and other motion-based visual enhancements commonly used in anime AMVs.
"""

import cv2
import numpy as np


# MoviePy 1.x imports (fallback)
from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips, ImageClip
from moviepy.video.fx import speedx

from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MotionEffectsEngine:
    """Engine for applying motion and impact effects to video clips."""
    
    def __init__(self):
        """Initialize the motion effects engine."""
        self.effects_cache = {}
        self.default_settings = {
            'speed_ramp': {
                'smooth_transition': True,
                'preserve_audio': False
            },
            'zoom_punch': {
                'max_zoom': 2.0,
                'default_duration': 0.2,
                'shake_intensity': 5
            },
            'camera_shake': {
                'decay_rate': 3.0,
                'border_mode': cv2.BORDER_REFLECT
            }
        }
    
    def speed_ramp_effect(self, clip: VideoFileClip, speed_points: List[Tuple[float, float]]) -> VideoFileClip:
        """
        Apply dynamic speed ramping to a video clip.
        
        Args:
            clip: Input video clip
            speed_points: List of (time, speed) tuples defining speed changes
            
        Returns:
            Video clip with speed ramping applied
            
        Example:
            speed_points = [(0, 1.0), (2, 0.3), (3, 2.5), (5, 1.0)]
            Creates normal -> slow-mo -> fast -> normal speed progression
        """
        
        try:
            if len(speed_points) < 2:
                logger.warning("Speed ramping requires at least 2 speed points")
                return clip
            
            segments = []
            for i in range(len(speed_points) - 1):
                start_time, start_speed = speed_points[i]
                end_time, end_speed = speed_points[i + 1]
                
                if end_time > clip.duration:
                    end_time = clip.duration
                
                segment = clip.subclip(start_time, end_time)
                
                # Calculate average speed for this segment
                avg_speed = (start_speed + end_speed) / 2
                
                # Apply speed change
                if avg_speed != 1.0:
                    try:
                        # Try MoviePy 2.x approach first
                        segment = segment.speedx(segment, avg_speed)
                    except (TypeError, AttributeError):
                        # Fallback to MoviePy 1.x approach
                        segment = segment.fx(speedx, avg_speed)
                
                segments.append(segment)
            
            # Concatenate all segments
            result = concatenate_videoclips(segments)
            logger.info(f"Applied speed ramping with {len(speed_points)} points")
            return result
            
        except Exception as e:
            logger.error(f"Error applying speed ramp effect: {e}")
            return clip
    
    def zoom_punch_effect(self, clip: VideoFileClip, zoom_time: float, 
                         zoom_factor: float = 1.5, duration: float = 0.2) -> VideoFileClip:
        """
        Apply zoom punch effect at a specific timestamp.
        
        Args:
            clip: Input video clip
            zoom_time: Time to apply zoom effect
            zoom_factor: Maximum zoom level (1.0 = no zoom)
            duration: Duration of the zoom effect
            
        Returns:
            Video clip with zoom punch effect applied
        """
        try:
            def zoom_func(get_frame, t):
                frame = get_frame(t)
                
                # Check if we're in the zoom effect time window
                if abs(t - zoom_time) < duration / 2:
                    # Calculate zoom intensity (peak at zoom_time)
                    intensity = 1 - abs(t - zoom_time) / (duration / 2)
                    current_zoom = 1 + (zoom_factor - 1) * intensity
                    
                    h, w = frame.shape[:2]
                    center_x, center_y = w // 2, h // 2
                    
                    # Add slight camera shake for impact
                    shake_intensity = self.default_settings['zoom_punch']['shake_intensity']
                    shake_x = int(np.random.uniform(-shake_intensity, shake_intensity) * intensity)
                    shake_y = int(np.random.uniform(-shake_intensity, shake_intensity) * intensity)
                    
                    # Create transformation matrix
                    M = cv2.getRotationMatrix2D(
                        (center_x + shake_x, center_y + shake_y), 0, current_zoom
                    )
                    
                    # Apply transformation
                    result = cv2.warpAffine(frame, M, (w, h))
                    return result
                
                return frame
            
            result = clip.fl(zoom_func)
            logger.info(f"Applied zoom punch effect at {zoom_time}s with factor {zoom_factor}")
            return result
            
        except Exception as e:
            logger.error(f"Error applying zoom punch effect: {e}")
            return clip
    
    def camera_shake_effect(self, clip: VideoFileClip, shake_intensity: float = 10, 
                           shake_duration: float = 1.0) -> VideoFileClip:
        """
        Add camera shake effect to simulate impact or excitement.
        
        Args:
            clip: Input video clip
            shake_intensity: Maximum shake displacement in pixels
            shake_duration: Duration of shake effect from start of clip
            
        Returns:
            Video clip with camera shake applied
        """
        try:
            def shake_func(get_frame, t):
                frame = get_frame(t)
                
                if t < shake_duration:
                    # Exponential decay for realistic shake
                    decay_rate = self.default_settings['camera_shake']['decay_rate']
                    current_intensity = shake_intensity * np.exp(-t * decay_rate)
                    
                    # Random shake offsets
                    dx = int(np.random.uniform(-current_intensity, current_intensity))
                    dy = int(np.random.uniform(-current_intensity, current_intensity))
                    
                    # Create translation matrix
                    h, w = frame.shape[:2]
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    
                    # Apply translation with border reflection
                    border_mode = self.default_settings['camera_shake']['border_mode']
                    result = cv2.warpAffine(frame, M, (w, h), borderMode=border_mode)
                    return result
                
                return frame
            
            result = clip.fl(shake_func)
            logger.info(f"Applied camera shake with intensity {shake_intensity} for {shake_duration}s")
            return result
            
        except Exception as e:
            logger.error(f"Error applying camera shake effect: {e}")
            return clip
    
    def motion_blur_effect(self, clip: VideoFileClip, blur_strength: float = 5.0,
                          motion_angle: float = 0.0) -> VideoFileClip:
        """
        Apply motion blur effect to simulate fast movement.
        
        Args:
            clip: Input video clip
            blur_strength: Strength of blur effect
            motion_angle: Angle of motion blur in degrees
            
        Returns:
            Video clip with motion blur applied
        """
        try:
            def blur_func(get_frame, t):
                frame = get_frame(t)
                
                # Create motion blur kernel
                kernel_size = int(blur_strength * 2) + 1
                kernel = np.zeros((kernel_size, kernel_size))
                
                # Create line kernel based on motion angle
                angle_radians = np.radians(motion_angle)
                center = kernel_size // 2
                
                for i in range(kernel_size):
                    x = int(center + (i - center) * np.cos(angle_radians))
                    y = int(center + (i - center) * np.sin(angle_radians))
                    
                    if 0 <= x < kernel_size and 0 <= y < kernel_size:
                        kernel[y, x] = 1
                
                # Normalize kernel
                kernel = kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel
                
                # Apply motion blur
                blurred = cv2.filter2D(frame, -1, kernel)
                return blurred
            
            result = clip.fl_image(blur_func)
            logger.info(f"Applied motion blur with strength {blur_strength} at angle {motion_angle}°")
            return result
            
        except Exception as e:
            logger.error(f"Error applying motion blur effect: {e}")
            return clip
    
    def freeze_frame_effect(self, clip: VideoFileClip, freeze_time: float, 
                           freeze_duration: float = 2.0) -> VideoFileClip:
        """
        Create a freeze frame effect at a specific time.
        
        Args:
            clip: Input video clip
            freeze_time: Time to freeze the frame
            freeze_duration: Duration to hold the frozen frame
            
        Returns:
            Video clip with freeze frame effect
        """
        try:
            if freeze_time >= clip.duration:
                logger.warning("Freeze time is beyond clip duration")
                return clip
            
            # Split clip at freeze point
            before_freeze = clip.subclip(0, freeze_time)
            after_freeze = clip.subclip(freeze_time, clip.duration)
            
            # Create frozen frame
            frozen_frame = clip.get_frame(freeze_time)
            freeze_clip = ImageClip(frozen_frame, duration=freeze_duration)
            
            # Concatenate parts
            result = concatenate_videoclips([before_freeze, freeze_clip, after_freeze])
            logger.info(f"Applied freeze frame at {freeze_time}s for {freeze_duration}s")
            return result
            
        except Exception as e:
            logger.error(f"Error applying freeze frame effect: {e}")
            return clip
    
    def apply_multiple_effects(self, clip: VideoFileClip, effects_config: dict) -> VideoFileClip:
        """
        Apply multiple motion effects in sequence.
        
        Args:
            clip: Input video clip
            effects_config: Dictionary defining effects to apply
            
        Returns:
            Video clip with all effects applied
            
        Example:
            effects_config = {
                'speed_ramp': {'speed_points': [(0, 1.0), (2, 0.5), (4, 1.0)]},
                'zoom_punch': {'zoom_time': 2.5, 'zoom_factor': 1.8},
                'camera_shake': {'shake_intensity': 8, 'shake_duration': 1.5}
            }
        """
        try:
            result_clip = clip
            
            # Apply effects in order
            if 'speed_ramp' in effects_config:
                config = effects_config['speed_ramp']
                result_clip = self.speed_ramp_effect(result_clip, **config)
            
            if 'zoom_punch' in effects_config:
                config = effects_config['zoom_punch']
                result_clip = self.zoom_punch_effect(result_clip, **config)
            
            if 'camera_shake' in effects_config:
                config = effects_config['camera_shake']
                result_clip = self.camera_shake_effect(result_clip, **config)
            
            if 'motion_blur' in effects_config:
                config = effects_config['motion_blur']
                result_clip = self.motion_blur_effect(result_clip, **config)
            
            if 'freeze_frame' in effects_config:
                config = effects_config['freeze_frame']
                result_clip = self.freeze_frame_effect(result_clip, **config)
            
            logger.info(f"Applied {len(effects_config)} motion effects to clip")
            return result_clip
            
        except Exception as e:
            logger.error(f"Error applying multiple effects: {e}")
            return clip
    
    def get_effect_presets(self) -> dict:
        """
        Get predefined effect presets for common anime scenarios.
        
        Returns:
            Dictionary of preset configurations
        """
        return {
            'action_sequence': {
                'speed_ramp': {'speed_points': [(0, 1.0), (1, 0.4), (2, 2.0), (4, 1.0)]},
                'zoom_punch': {'zoom_time': 1.5, 'zoom_factor': 1.8, 'duration': 0.3},
                'camera_shake': {'shake_intensity': 12, 'shake_duration': 0.8}
            },
            'dramatic_moment': {
                'speed_ramp': {'speed_points': [(0, 1.0), (1, 0.2), (3, 1.0)]},
                'freeze_frame': {'freeze_time': 1.5, 'freeze_duration': 1.0}
            },
            'impact_hit': {
                'zoom_punch': {'zoom_time': 0.1, 'zoom_factor': 2.2, 'duration': 0.15},
                'camera_shake': {'shake_intensity': 15, 'shake_duration': 0.5}
            },
            'fast_movement': {
                'motion_blur': {'blur_strength': 8.0, 'motion_angle': 45},
                'speed_ramp': {'speed_points': [(0, 0.8), (1, 3.0), (2, 1.0)]}
            }
        }