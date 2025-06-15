"""
Advanced Transition Effects Engine

Provides sophisticated transition effects between video clips including
iris transitions, swipes, morphs, and other anime-style transitions.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import logging
import math

# MoviePy 1.x imports (fallback)
from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.fx import speedx

logger = logging.getLogger(__name__)


class TransitionEngine:
    """Engine for creating advanced transition effects between video clips."""
    
    def __init__(self):
        """Initialize the transition engine."""
        self.transition_templates = {}
        self.load_transition_templates()
    
    def load_transition_templates(self):
        """Load predefined transition templates."""
        self.transition_templates = {
            'iris': self.iris_transition,
            'swipe': self.swipe_transition,
            'dissolve': self.dissolve_transition,
            'zoom': self.zoom_transition,
            'spiral': self.spiral_transition,
            'slice': self.slice_transition,
            'pixelate': self.pixelate_transition,
            'radial': self.radial_transition
        }
    
    def iris_transition(self, clip1: VideoFileClip, clip2: VideoFileClip, 
                       duration: float = 1.0, center: Tuple[int, int] = None) -> CompositeVideoClip:
        """
        Create iris transition between clips (classic anime style).
        
        Args:
            clip1: First video clip
            clip2: Second video clip
            duration: Duration of transition
            center: Center point of iris (None for center)
            
        Returns:
            Composite video with iris transition
        """
        try:
            w, h = clip1.size
            if center is None:
                center = (w // 2, h // 2)
            
            def create_iris_mask(t):
                if t < duration:
                    # Progress from 0 (closed) to 1 (fully open)
                    progress = t / duration
                    
                    # Create circular mask
                    mask = np.zeros((h, w), dtype=np.uint8)
                    max_radius = int(np.sqrt(w**2 + h**2))  # Diagonal distance
                    radius = int(max_radius * progress)
                    
                    if radius > 0:
                        cv2.circle(mask, center, radius, 255, -1)
                    
                    return mask
                else:
                    # Fully open
                    return np.ones((h, w), dtype=np.uint8) * 255
            
            def apply_iris_mask(get_frame, t):
                frame2 = clip2.get_frame(t)
                mask = create_iris_mask(t)
                
                if mask is not None:
                    # Create 3-channel mask
                    mask_3d = cv2.merge([mask, mask, mask]) / 255.0
                    
                    # Get frame from first clip
                    frame1 = clip1.get_frame(min(t, clip1.duration - 0.01))
                    
                    # Blend frames using mask
                    result = frame1 * (1 - mask_3d) + frame2 * mask_3d
                    return result.astype(np.uint8)
                
                return frame2
            
            # Create masked version of clip2
            masked_clip2 = clip2.fl(apply_iris_mask)
            
            # Composite clips
            total_duration = max(clip1.duration, clip2.duration)
            result = CompositeVideoClip([clip1.set_duration(total_duration), 
                                       masked_clip2.set_duration(total_duration)])
            
            logger.info(f"Created iris transition with duration {duration}s")
            return result
            
        except Exception as e:
            logger.error(f"Error creating iris transition: {e}")
            return concatenate_videoclips([clip1, clip2])
    
    def swipe_transition(self, clip1: VideoFileClip, clip2: VideoFileClip,
                        direction: str = "left", duration: float = 0.5) -> CompositeVideoClip:
        """
        Create swipe transition between clips.
        
        Args:
            clip1: First video clip
            clip2: Second video clip
            direction: Direction of swipe ("left", "right", "up", "down")
            duration: Duration of transition
            
        Returns:
            Composite video with swipe transition
        """
        try:
            w, h = clip1.size
            
            def create_swipe_mask(t):
                if t < duration:
                    progress = t / duration
                    mask = np.zeros((h, w), dtype=np.uint8)
                    
                    if direction == "left":
                        x_boundary = int(w * progress)
                        mask[:, :x_boundary] = 255
                    elif direction == "right":
                        x_boundary = int(w * (1 - progress))
                        mask[:, x_boundary:] = 255
                    elif direction == "up":
                        y_boundary = int(h * progress)
                        mask[:y_boundary, :] = 255
                    elif direction == "down":
                        y_boundary = int(h * (1 - progress))
                        mask[y_boundary:, :] = 255
                    
                    return mask
                else:
                    return np.ones((h, w), dtype=np.uint8) * 255
            
            def apply_swipe_mask(get_frame, t):
                frame2 = clip2.get_frame(t)
                mask = create_swipe_mask(t)
                
                if mask is not None:
                    mask_3d = cv2.merge([mask, mask, mask]) / 255.0
                    frame1 = clip1.get_frame(min(t, clip1.duration - 0.01))
                    
                    result = frame1 * (1 - mask_3d) + frame2 * mask_3d
                    return result.astype(np.uint8)
                
                return frame2
            
            masked_clip2 = clip2.fl(apply_swipe_mask)
            
            total_duration = max(clip1.duration, clip2.duration)
            result = CompositeVideoClip([clip1.set_duration(total_duration),
                                       masked_clip2.set_duration(total_duration)])
            
            logger.info(f"Created {direction} swipe transition with duration {duration}s")
            return result
            
        except Exception as e:
            logger.error(f"Error creating swipe transition: {e}")
            return concatenate_videoclips([clip1, clip2])
    
    def dissolve_transition(self, clip1: VideoFileClip, clip2: VideoFileClip,
                           duration: float = 1.0) -> CompositeVideoClip:
        """
        Create dissolve (crossfade) transition between clips.
        
        Args:
            clip1: First video clip
            clip2: Second video clip
            duration: Duration of transition
            
        Returns:
            Composite video with dissolve transition
        """
        try:
            def apply_dissolve(get_frame, t):
                if t < duration:
                    alpha = t / duration
                    frame1 = clip1.get_frame(min(t, clip1.duration - 0.01))
                    frame2 = clip2.get_frame(t)
                    
                    # Blend frames
                    result = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
                    return result
                else:
                    return clip2.get_frame(t)
            
            blended_clip = clip2.fl(apply_dissolve)
            
            total_duration = max(clip1.duration, clip2.duration)
            result = CompositeVideoClip([clip1.set_duration(total_duration),
                                       blended_clip.set_duration(total_duration)])
            
            logger.info(f"Created dissolve transition with duration {duration}s")
            return result
            
        except Exception as e:
            logger.error(f"Error creating dissolve transition: {e}")
            return concatenate_videoclips([clip1, clip2])
    
    def zoom_transition(self, clip1: VideoFileClip, clip2: VideoFileClip,
                       zoom_in: bool = True, duration: float = 1.0) -> CompositeVideoClip:
        """
        Create zoom transition between clips.
        
        Args:
            clip1: First video clip
            clip2: Second video clip
            zoom_in: Whether to zoom in (True) or out (False)
            duration: Duration of transition
            
        Returns:
            Composite video with zoom transition
        """
        try:
            w, h = clip1.size
            
            def apply_zoom_transition(get_frame, t):
                if t < duration:
                    progress = t / duration
                    
                    if zoom_in:
                        # Zoom into clip1, then reveal clip2
                        zoom_factor = 1 + progress * 2  # Zoom from 1x to 3x
                        frame1 = clip1.get_frame(min(t, clip1.duration - 0.01))
                        
                        # Apply zoom to frame1
                        center_x, center_y = w // 2, h // 2
                        M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_factor)
                        zoomed_frame1 = cv2.warpAffine(frame1, M, (w, h))
                        
                        # Blend with clip2 based on zoom level
                        if progress > 0.5:
                            frame2 = clip2.get_frame(t)
                            alpha = (progress - 0.5) * 2  # Fade in clip2 in second half
                            result = cv2.addWeighted(zoomed_frame1, 1 - alpha, frame2, alpha, 0)
                            return result
                        else:
                            return zoomed_frame1
                    else:
                        # Zoom out from clip1 to reveal clip2
                        zoom_factor = 3 - progress * 2  # Zoom from 3x to 1x
                        frame1 = clip1.get_frame(min(t, clip1.duration - 0.01))
                        
                        center_x, center_y = w // 2, h // 2
                        M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_factor)
                        zoomed_frame1 = cv2.warpAffine(frame1, M, (w, h))
                        
                        frame2 = clip2.get_frame(t)
                        alpha = progress
                        result = cv2.addWeighted(zoomed_frame1, 1 - alpha, frame2, alpha, 0)
                        return result
                else:
                    return clip2.get_frame(t)
            
            transition_clip = clip2.fl(apply_zoom_transition)
            
            total_duration = max(clip1.duration, clip2.duration)
            result = CompositeVideoClip([clip1.set_duration(total_duration),
                                       transition_clip.set_duration(total_duration)])
            
            zoom_type = "in" if zoom_in else "out"
            logger.info(f"Created zoom {zoom_type} transition with duration {duration}s")
            return result
            
        except Exception as e:
            logger.error(f"Error creating zoom transition: {e}")
            return concatenate_videoclips([clip1, clip2])
    
    def spiral_transition(self, clip1: VideoFileClip, clip2: VideoFileClip,
                         duration: float = 1.5, clockwise: bool = True) -> CompositeVideoClip:
        """
        Create spiral transition between clips.
        
        Args:
            clip1: First video clip
            clip2: Second video clip
            duration: Duration of transition
            clockwise: Direction of spiral
            
        Returns:
            Composite video with spiral transition
        """
        try:
            w, h = clip1.size
            center_x, center_y = w // 2, h // 2
            
            def create_spiral_mask(t):
                if t < duration:
                    progress = t / duration
                    mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # Create spiral pattern
                    max_radius = int(np.sqrt(w**2 + h**2))
                    total_angle = 4 * np.pi * progress  # 2 full rotations
                    
                    for radius in range(0, max_radius, 2):
                        angle = total_angle * (radius / max_radius)
                        if not clockwise:
                            angle = -angle
                        
                        # Calculate point on spiral
                        x = int(center_x + radius * np.cos(angle))
                        y = int(center_y + radius * np.sin(angle))
                        
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(mask, (x, y), 3, 255, -1)
                    
                    # Fill in the spiral
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    
                    return mask
                else:
                    return np.ones((h, w), dtype=np.uint8) * 255
            
            def apply_spiral_mask(get_frame, t):
                frame2 = clip2.get_frame(t)
                mask = create_spiral_mask(t)
                
                if mask is not None:
                    mask_3d = cv2.merge([mask, mask, mask]) / 255.0
                    frame1 = clip1.get_frame(min(t, clip1.duration - 0.01))
                    
                    result = frame1 * (1 - mask_3d) + frame2 * mask_3d
                    return result.astype(np.uint8)
                
                return frame2
            
            masked_clip2 = clip2.fl(apply_spiral_mask)
            
            total_duration = max(clip1.duration, clip2.duration)
            result = CompositeVideoClip([clip1.set_duration(total_duration),
                                       masked_clip2.set_duration(total_duration)])
            
            direction = "clockwise" if clockwise else "counter-clockwise"
            logger.info(f"Created {direction} spiral transition with duration {duration}s")
            return result
            
        except Exception as e:
            logger.error(f"Error creating spiral transition: {e}")
            return concatenate_videoclips([clip1, clip2])
    
    def slice_transition(self, clip1: VideoFileClip, clip2: VideoFileClip,
                        slice_count: int = 10, duration: float = 0.8) -> CompositeVideoClip:
        """
        Create slice transition with multiple segments.
        
        Args:
            clip1: First video clip
            clip2: Second video clip
            slice_count: Number of slices
            duration: Duration of transition
            
        Returns:
            Composite video with slice transition
        """
        try:
            w, h = clip1.size
            slice_width = w // slice_count
            
            def create_slice_mask(t):
                if t < duration:
                    progress = t / duration
                    mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # Animate slices with different delays
                    for i in range(slice_count):
                        slice_delay = (i / slice_count) * 0.5  # Stagger slice reveals
                        slice_progress = max(0, (progress - slice_delay) / (1 - slice_delay))
                        
                        if slice_progress > 0:
                            x_start = i * slice_width
                            x_end = min((i + 1) * slice_width, w)
                            
                            # Animate slice reveal from top to bottom
                            reveal_height = int(h * slice_progress)
                            mask[:reveal_height, x_start:x_end] = 255
                    
                    return mask
                else:
                    return np.ones((h, w), dtype=np.uint8) * 255
            
            def apply_slice_mask(get_frame, t):
                frame2 = clip2.get_frame(t)
                mask = create_slice_mask(t)
                
                if mask is not None:
                    mask_3d = cv2.merge([mask, mask, mask]) / 255.0
                    frame1 = clip1.get_frame(min(t, clip1.duration - 0.01))
                    
                    result = frame1 * (1 - mask_3d) + frame2 * mask_3d
                    return result.astype(np.uint8)
                
                return frame2
            
            masked_clip2 = clip2.fl(apply_slice_mask)
            
            total_duration = max(clip1.duration, clip2.duration)
            result = CompositeVideoClip([clip1.set_duration(total_duration),
                                       masked_clip2.set_duration(total_duration)])
            
            logger.info(f"Created slice transition with {slice_count} slices, duration {duration}s")
            return result
            
        except Exception as e:
            logger.error(f"Error creating slice transition: {e}")
            return concatenate_videoclips([clip1, clip2])
    
    def pixelate_transition(self, clip1: VideoFileClip, clip2: VideoFileClip,
                           duration: float = 1.0, max_pixel_size: int = 20) -> CompositeVideoClip:
        """
        Create pixelate transition between clips.
        
        Args:
            clip1: First video clip
            clip2: Second video clip
            duration: Duration of transition
            max_pixel_size: Maximum pixel size for effect
            
        Returns:
            Composite video with pixelate transition
        """
        try:
            w, h = clip1.size
            
            def apply_pixelate_transition(get_frame, t):
                if t < duration:
                    progress = t / duration
                    
                    # Pixelate clip1 increasingly, then blend with clip2
                    frame1 = clip1.get_frame(min(t, clip1.duration - 0.01))
                    
                    # Apply pixelation to frame1
                    pixel_size = int(max_pixel_size * progress)
                    if pixel_size > 1:
                        # Downscale
                        small_w = max(1, w // pixel_size)
                        small_h = max(1, h // pixel_size)
                        
                        small_frame = cv2.resize(frame1, (small_w, small_h), 
                                               interpolation=cv2.INTER_LINEAR)
                        
                        # Upscale back to original size
                        pixelated_frame1 = cv2.resize(small_frame, (w, h), 
                                                    interpolation=cv2.INTER_NEAREST)
                    else:
                        pixelated_frame1 = frame1
                    
                    # Blend with clip2 based on progress
                    frame2 = clip2.get_frame(t)
                    alpha = progress
                    result = cv2.addWeighted(pixelated_frame1, 1 - alpha, frame2, alpha, 0)
                    
                    return result
                else:
                    return clip2.get_frame(t)
            
            transition_clip = clip2.fl(apply_pixelate_transition)
            
            total_duration = max(clip1.duration, clip2.duration)
            result = CompositeVideoClip([clip1.set_duration(total_duration),
                                       transition_clip.set_duration(total_duration)])
            
            logger.info(f"Created pixelate transition with max pixel size {max_pixel_size}, duration {duration}s")
            return result
            
        except Exception as e:
            logger.error(f"Error creating pixelate transition: {e}")
            return concatenate_videoclips([clip1, clip2])
    
    def radial_transition(self, clip1: VideoFileClip, clip2: VideoFileClip,
                         segments: int = 8, duration: float = 1.0) -> CompositeVideoClip:
        """
        Create radial wipe transition with multiple segments.
        
        Args:
            clip1: First video clip
            clip2: Second video clip
            segments: Number of radial segments
            duration: Duration of transition
            
        Returns:
            Composite video with radial transition
        """
        try:
            w, h = clip1.size
            center_x, center_y = w // 2, h // 2
            
            def create_radial_mask(t):
                if t < duration:
                    progress = t / duration
                    mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # Calculate how many segments to reveal
                    segments_to_reveal = progress * segments
                    
                    for seg in range(int(segments_to_reveal) + 1):
                        if seg < segments:
                            # Calculate angles for this segment
                            start_angle = (seg / segments) * 2 * np.pi
                            end_angle = ((seg + 1) / segments) * 2 * np.pi
                            
                            # If this is the last partial segment
                            if seg == int(segments_to_reveal):
                                partial_progress = segments_to_reveal - int(segments_to_reveal)
                                end_angle = start_angle + (end_angle - start_angle) * partial_progress
                            
                            # Create points for the segment
                            points = [(center_x, center_y)]
                            max_radius = int(np.sqrt(w**2 + h**2))
                            
                            # Add points along the arc
                            num_points = 20
                            for i in range(num_points + 1):
                                angle = start_angle + (end_angle - start_angle) * (i / num_points)
                                x = int(center_x + max_radius * np.cos(angle))
                                y = int(center_y + max_radius * np.sin(angle))
                                points.append((x, y))
                            
                            # Fill the segment
                            points_array = np.array(points, dtype=np.int32)
                            cv2.fillPoly(mask, [points_array], 255)
                    
                    return mask
                else:
                    return np.ones((h, w), dtype=np.uint8) * 255
            
            def apply_radial_mask(get_frame, t):
                frame2 = clip2.get_frame(t)
                mask = create_radial_mask(t)
                
                if mask is not None:
                    mask_3d = cv2.merge([mask, mask, mask]) / 255.0
                    frame1 = clip1.get_frame(min(t, clip1.duration - 0.01))
                    
                    result = frame1 * (1 - mask_3d) + frame2 * mask_3d
                    return result.astype(np.uint8)
                
                return frame2
            
            masked_clip2 = clip2.fl(apply_radial_mask)
            
            total_duration = max(clip1.duration, clip2.duration)
            result = CompositeVideoClip([clip1.set_duration(total_duration),
                                       masked_clip2.set_duration(total_duration)])
            
            logger.info(f"Created radial transition with {segments} segments, duration {duration}s")
            return result
            
        except Exception as e:
            logger.error(f"Error creating radial transition: {e}")
            return concatenate_videoclips([clip1, clip2])
    
    def create_dynamic_transition(self, clip1: VideoFileClip, clip2: VideoFileClip,
                                 transition_type: str = "random", **kwargs) -> CompositeVideoClip:
        """
        Create a transition using the specified type or random selection.
        
        Args:
            clip1: First video clip
            clip2: Second video clip
            transition_type: Type of transition or "random"
            **kwargs: Additional arguments for the transition
            
        Returns:
            Composite video with the specified transition
        """
        try:
            if transition_type == "random":
                transition_types = list(self.transition_templates.keys())
                transition_type = np.random.choice(transition_types)
            
            if transition_type in self.transition_templates:
                transition_func = self.transition_templates[transition_type]
                return transition_func(clip1, clip2, **kwargs)
            else:
                logger.warning(f"Unknown transition type: {transition_type}, using dissolve")
                return self.dissolve_transition(clip1, clip2, **kwargs)
                
        except Exception as e:
            logger.error(f"Error creating dynamic transition: {e}")
            return concatenate_videoclips([clip1, clip2])
    
    def get_transition_presets(self) -> Dict[str, Any]:
        """
        Get predefined transition presets for different scenarios.
        
        Returns:
            Dictionary of preset configurations
        """
        return {
            'action_scene': {
                'type': 'slice',
                'slice_count': 12,
                'duration': 0.6
            },
            'emotional_moment': {
                'type': 'dissolve',
                'duration': 1.5
            },
            'dramatic_reveal': {
                'type': 'iris',
                'duration': 1.2
            },
            'fast_paced': {
                'type': 'swipe',
                'direction': 'left',
                'duration': 0.3
            },
            'magical_transformation': {
                'type': 'spiral',
                'clockwise': True,
                'duration': 2.0
            },
            'digital_glitch': {
                'type': 'pixelate',
                'max_pixel_size': 25,
                'duration': 0.8
            },
            'power_up': {
                'type': 'radial',
                'segments': 6,
                'duration': 1.0
            }
        }