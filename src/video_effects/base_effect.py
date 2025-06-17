"""
Base Effect Interface - Unified interface for all video effects

Provides a standard interface that all effects must implement to ensure
consistent behavior across the entire effects library.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from moviepy.editor import VideoFileClip
import logging

logger = logging.getLogger(__name__)


class BaseVideoEffect(ABC):
    """Base class for all video effects with unified interface."""
    
    def __init__(self):
        """Initialize base effect."""
        self.name = self.__class__.__name__
        self.default_params = {}
    
    @abstractmethod
    def apply_to_clip_portion(self, clip: VideoFileClip, start_time: float, 
                             duration: float, **params) -> VideoFileClip:
        """
        Apply effect to a specific portion of a video clip.
        
        Args:
            clip: Input video clip
            start_time: Start time within the clip to apply effect (seconds)
            duration: Duration of the effect (seconds)
            **params: Effect-specific parameters
            
        Returns:
            Modified video clip with effect applied to specified portion
        """
        pass
    
    def validate_params(self, **params) -> Dict[str, Any]:
        """
        Validate and normalize effect parameters.
        
        Args:
            **params: Effect parameters to validate
            
        Returns:
            Dictionary of validated parameters
        """
        validated = self.default_params.copy()
        validated.update(params)
        return validated
    
    def apply_to_subclip(self, full_clip: VideoFileClip, start_time: float, 
                        duration: float, effect_func, **params) -> VideoFileClip:
        """
        Helper method to apply effects to a specific time range within a clip.
        
        Args:
            full_clip: Full video clip
            start_time: Start time for the effect
            duration: Duration of the effect
            effect_func: Function to apply the effect
            **params: Effect parameters
            
        Returns:
            Full clip with effect applied to specified portion
        """
        try:
            # Validate time bounds
            clip_duration = full_clip.duration
            end_time = start_time + duration
            
            if start_time < 0:
                start_time = 0
            if end_time > clip_duration:
                end_time = clip_duration
                duration = end_time - start_time
            
            if duration <= 0:
                logger.warning(f"Invalid duration {duration} for {self.name}")
                return full_clip
            
            # Split clip into parts
            parts = []
            
            # Part before effect
            if start_time > 0:
                parts.append(full_clip.subclip(0, start_time))
            
            # Part with effect applied
            effect_clip = full_clip.subclip(start_time, end_time)
            processed_clip = effect_func(effect_clip, **params)
            parts.append(processed_clip)
            
            # Part after effect
            if end_time < clip_duration:
                parts.append(full_clip.subclip(end_time, clip_duration))
            
            # Concatenate all parts
            if len(parts) == 1:
                return parts[0]
            else:
                from moviepy.editor import concatenate_videoclips
                return concatenate_videoclips(parts)
                
        except Exception as e:
            logger.error(f"Error applying {self.name} to subclip: {e}")
            return full_clip
    
    def apply_frame_effect_to_subclip(self, full_clip: VideoFileClip, start_time: float,
                                     duration: float, frame_effect_func, **params) -> VideoFileClip:
        """
        Helper method to apply frame-level effects to a specific time range.
        
        Args:
            full_clip: Full video clip
            start_time: Start time for the effect
            duration: Duration of the effect
            frame_effect_func: Function that processes individual frames
            **params: Effect parameters
            
        Returns:
            Full clip with frame effect applied to specified portion
        """
        def clip_effect_wrapper(clip, **effect_params):
            """Wrapper to apply frame effect to entire clip."""
            def process_frame(get_frame, t):
                frame = get_frame(t)
                return frame_effect_func(frame, **effect_params)
            
            return clip.fl(process_frame)
        
        return self.apply_to_subclip(full_clip, start_time, duration, 
                                   clip_effect_wrapper, **params)


def unified_effect_interface(effect_func):
    """
    Decorator to wrap existing effects with unified interface.
    
    This decorator allows existing effect functions to be used with the
    new unified interface without requiring complete rewrites.
    """
    def wrapper(clip: VideoFileClip, start_time: float, duration: float, **params):
        """Unified interface wrapper."""
        
        # Create a temporary effect instance for helper methods
        class TempEffect(BaseVideoEffect):
            def apply_to_clip_portion(self, clip, start_time, duration, **params):
                return effect_func(clip, **params)
        
        temp_effect = TempEffect()
        
        # Determine if this is a frame-level or clip-level effect
        # Frame-level effects typically have 'frame' as first parameter
        import inspect
        sig = inspect.signature(effect_func)
        first_param = list(sig.parameters.keys())[0] if sig.parameters else ""
        
        if first_param == 'frame':
            # Frame-level effect
            return temp_effect.apply_frame_effect_to_subclip(
                clip, start_time, duration, effect_func, **params
            )
        else:
            # Clip-level effect
            return temp_effect.apply_to_subclip(
                clip, start_time, duration, effect_func, **params
            )
    
    return wrapper