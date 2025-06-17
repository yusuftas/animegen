"""
Anime-Style Visual Effects Library

Provides anime-specific visual effects including speed lines, impact frames,
energy auras, and other stylistic elements common in anime and manga.
"""

import cv2
import numpy as np


from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips, ImageClip

from typing import List, Tuple, Optional, Union
import logging
import math

from .base_effect import BaseVideoEffect, unified_effect_interface

logger = logging.getLogger(__name__)


class AnimeEffectsLibrary:
    """Library of anime-specific visual effects."""
    
    def __init__(self):
        """Initialize the anime effects library."""
        self.effect_templates = {}
        self.default_colors = {
            'speed_lines': (255, 255, 255),
            'impact_frame': (255, 255, 255),
            'energy_aura': (255, 215, 0),  # Gold
            'action_lines': (255, 100, 100)  # Red-ish
        }
        
    def add_speed_lines(self, frame: np.ndarray, direction: str = "right", 
                       intensity: float = 0.8, color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Add anime-style speed lines to a frame.
        
        Args:
            frame: Input frame as numpy array
            direction: Direction of speed lines ("right", "left", "radial", "diagonal")
            intensity: Intensity of effect (0.0 to 1.0)
            color: RGB color tuple for lines
            
        Returns:
            Frame with speed lines added
        """
        try:
            if color is None:
                color = self.default_colors['speed_lines']
                
            h, w = frame.shape[:2]
            overlay = np.zeros_like(frame)
            
            # Number of lines based on intensity
            num_lines = int(30 * intensity)
            
            for i in range(num_lines):
                if direction == "right":
                    # Horizontal lines moving right
                    y = np.random.randint(0, h)
                    thickness = np.random.randint(1, 4)
                    length = np.random.randint(w//4, w//2)
                    x_start = np.random.randint(0, w - length)
                    
                    cv2.line(overlay, (x_start, y), (x_start + length, y), color, thickness)
                    
                elif direction == "left":
                    # Horizontal lines moving left
                    y = np.random.randint(0, h)
                    thickness = np.random.randint(1, 4)
                    length = np.random.randint(w//4, w//2)
                    x_end = np.random.randint(length, w)
                    
                    cv2.line(overlay, (x_end - length, y), (x_end, y), color, thickness)
                    
                elif direction == "radial":
                    # Radial lines from center
                    center_x, center_y = w//2, h//2
                    angle = np.random.uniform(0, 2*np.pi)
                    length = np.random.randint(50, min(w, h)//3)
                    thickness = np.random.randint(1, 3)
                    
                    end_x = int(center_x + length * np.cos(angle))
                    end_y = int(center_y + length * np.sin(angle))
                    
                    # Ensure line stays within frame
                    end_x = max(0, min(w-1, end_x))
                    end_y = max(0, min(h-1, end_y))
                    
                    cv2.line(overlay, (center_x, center_y), (end_x, end_y), color, thickness)
                    
                elif direction == "diagonal":
                    # Diagonal lines for dynamic movement
                    x1 = np.random.randint(0, w//2)
                    y1 = np.random.randint(0, h)
                    length = np.random.randint(w//6, w//3)
                    thickness = np.random.randint(1, 3)
                    
                    x2 = min(w-1, x1 + length)
                    y2 = max(0, y1 - np.random.randint(0, length//2))
                    
                    cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)
            
            # Blur lines slightly for more natural look
            overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
            
            # Blend with original frame
            alpha = 0.6
            result = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error adding speed lines: {e}")
            return frame
    
    def create_impact_frame(self, frame: np.ndarray, style: str = "manga") -> np.ndarray:
        """
        Create high-contrast impact frame effect.
        
        Args:
            frame: Input frame
            style: Style of impact frame ("manga", "energy", "flash")
            
        Returns:
            Processed impact frame
        """
        try:
            if style == "manga":
                # Convert to high contrast black and white
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply adaptive threshold for manga-like effect
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
                
                # Add dramatic border effect
                bordered = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, 
                                            cv2.BORDER_CONSTANT, value=0)
                
                # Convert back to color and add red accent
                impact_frame = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)
                impact_frame[:, :, 2] = np.maximum(impact_frame[:, :, 2], 100)  # Red channel
                
                return impact_frame
                
            elif style == "energy":
                # High saturation energy effect
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.5, 0, 255)  # Boost saturation
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.4, 0, 255)  # Increase brightness
                
                energy_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                # Add energy glow effect
                glow = cv2.GaussianBlur(energy_frame, (21, 21), 0)
                result = cv2.addWeighted(energy_frame, 0.7, glow, 0.3, 0)
                
                return result
                
            elif style == "flash":
                # White flash with frame outline
                flash_frame = np.ones_like(frame) * 255
                
                # Add frame outline
                h, w = frame.shape[:2]
                cv2.rectangle(flash_frame, (5, 5), (w-6, h-6), (0, 0, 0), 3)
                
                # Blend with original
                result = cv2.addWeighted(frame, 0.3, flash_frame, 0.7, 0)
                
                return result
                
        except Exception as e:
            logger.error(f"Error creating impact frame: {e}")
            return frame
    
    def create_character_glow(self, frame: np.ndarray, glow_color: Tuple[int, int, int] = None,
                             glow_intensity: float = 0.5) -> np.ndarray:
        """
        Create glow effect around characters or objects.
        
        Args:
            frame: Input frame
            glow_color: Color of the glow effect
            glow_intensity: Intensity of the glow
            
        Returns:
            Frame with glow effect
        """
        try:
            if glow_color is None:
                glow_color = self.default_colors['energy_aura']
            
            # Create edge detection for glow boundaries
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Create glow effect
            glow = np.zeros_like(frame)
            
            # Dilate edges to create glow boundary
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            glow_mask = cv2.dilate(edges, kernel, iterations=2)
            glow_mask = cv2.GaussianBlur(glow_mask, (21, 21), 0)
            
            # Apply glow color
            for i in range(3):
                glow[:, :, i] = (glow_mask / 255.0) * glow_color[i]
            
            # Blend with original
            result = cv2.addWeighted(frame, 1.0, glow, glow_intensity, 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating character glow: {e}")
            return frame
    
    def create_energy_aura(self, frame: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """
        Create energy aura effect on a single frame.
        
        Args:
            frame: Input frame
            intensity: Intensity of the aura effect
            
        Returns:
            Frame with energy aura effect
        """
        try:
            # Enhance brightness and saturation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + intensity * 0.5), 0, 255)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + intensity * 0.3), 0, 255)
            
            energy_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Add outer glow for strong intensities
            if intensity > 0.6:
                energy_frame = self.create_character_glow(energy_frame, 
                                                        glow_intensity=intensity * 0.4)
            
            return energy_frame
            
        except Exception as e:
            logger.error(f"Error creating energy aura: {e}")
            return frame
    
    def add_action_lines(self, frame: np.ndarray, center_point: Tuple[int, int] = None,
                        line_count: int = 12, max_length: int = 100) -> np.ndarray:
        """
        Add action lines radiating from a center point.
        
        Args:
            frame: Input frame
            center_point: Center point for action lines (None for frame center)
            line_count: Number of action lines
            max_length: Maximum length of lines
            
        Returns:
            Frame with action lines added
        """
        try:
            h, w = frame.shape[:2]
            if center_point is None:
                center_point = (w//2, h//2)
            
            overlay = np.zeros_like(frame)
            color = self.default_colors['action_lines']
            
            for i in range(line_count):
                angle = (2 * np.pi * i) / line_count
                length = np.random.randint(max_length//2, max_length)
                thickness = np.random.randint(2, 5)
                
                end_x = int(center_point[0] + length * np.cos(angle))
                end_y = int(center_point[1] + length * np.sin(angle))
                
                # Ensure lines stay within frame
                end_x = max(0, min(w-1, end_x))
                end_y = max(0, min(h-1, end_y))
                
                cv2.line(overlay, center_point, (end_x, end_y), color, thickness)
            
            # Blend with original
            result = cv2.addWeighted(frame, 0.8, overlay, 0.6, 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error adding action lines: {e}")
            return frame
    
    def manga_screen_tone(self, frame: np.ndarray, tone_type: str = "dots") -> np.ndarray:
        """
        Apply manga-style screen tone effects.
        
        Args:
            frame: Input frame
            tone_type: Type of screen tone ("dots", "lines", "crosshatch")
            
        Returns:
            Frame with screen tone effect
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            if tone_type == "dots":
                # Create dot pattern
                dots = np.zeros_like(gray)
                for y in range(0, h, 8):
                    for x in range(0, w, 8):
                        if gray[y, x] < 128:  # Dark areas get dots
                            cv2.circle(dots, (x, y), 2, 255, -1)
                
                # Convert back to color
                dots_color = cv2.cvtColor(dots, cv2.COLOR_GRAY2BGR)
                result = cv2.addWeighted(frame, 0.7, dots_color, 0.3, 0)
                
            elif tone_type == "lines":
                # Create line pattern
                lines = np.zeros_like(gray)
                for y in range(0, h, 4):
                    if y % 8 == 0:
                        lines[y, :] = 255
                
                lines_color = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)
                result = cv2.addWeighted(frame, 0.8, lines_color, 0.2, 0)
                
            elif tone_type == "crosshatch":
                # Create crosshatch pattern
                crosshatch = np.zeros_like(gray)
                for y in range(0, h, 6):
                    crosshatch[y, :] = 255
                for x in range(0, w, 6):
                    crosshatch[:, x] = 255
                
                crosshatch_color = cv2.cvtColor(crosshatch, cv2.COLOR_GRAY2BGR)
                result = cv2.addWeighted(frame, 0.8, crosshatch_color, 0.2, 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying manga screen tone: {e}")
            return frame
    
    def get_anime_effect_presets(self) -> dict:
        """
        Get predefined anime effect presets.
        
        Returns:
            Dictionary of preset configurations
        """
        return {
            'power_up_scene': {
                'energy_aura': {'intensity': 1.2, 'pulse_rate': 8.0},
                'action_lines': {'line_count': 16, 'max_length': 120}
            },
            'speed_boost': {
                'speed_lines': {'direction': 'right', 'intensity': 1.0},
                'motion_blur': {'blur_strength': 6.0}
            },
            'dramatic_impact': {
                'impact_frames': {'style': 'manga', 'duration': 0.15},
                'action_lines': {'line_count': 20, 'max_length': 150}
            },
            'energy_attack': {
                'energy_aura': {'intensity': 1.5, 'pulse_rate': 12.0},
                'speed_lines': {'direction': 'radial', 'intensity': 0.9},
                'impact_frames': {'style': 'energy', 'duration': 0.2}
            },
            'emotional_moment': {
                'character_glow': {'glow_intensity': 0.3},
                'manga_tone': {'tone_type': 'dots'}
            }
        }

    # Unified Interface Methods
    def speed_lines_unified(self, clip: VideoFileClip, start_time: float, duration: float, 
                           direction: str = "right", intensity: float = 0.8) -> VideoFileClip:
        """Apply speed lines effect with unified interface."""
        def effect_func(effect_clip, **params):
            def apply_lines(get_frame, t):
                frame = get_frame(t)
                # Apply fade in/out within the effect duration
                local_time = t
                progress = local_time / effect_clip.duration if effect_clip.duration > 0 else 0
                fade_intensity = params['intensity'] * (1.0 - abs(progress - 0.5) * 2)
                fade_intensity = max(0, fade_intensity)
                
                return self.add_speed_lines(frame, params['direction'], fade_intensity)
            
            return effect_clip.fl(apply_lines)
        
        return self._apply_to_subclip(clip, start_time, duration, effect_func, 
                                    direction=direction, intensity=intensity)

    def impact_frames_unified(self, clip: VideoFileClip, start_time: float, duration: float,
                             style: str = "manga") -> VideoFileClip:
        """Apply impact frames effect with unified interface."""
        def effect_func(effect_clip, **params):
            def apply_impact(get_frame, t):
                frame = get_frame(t)
                return self.create_impact_frame(frame, params['style'])
            
            return effect_clip.fl(apply_impact)
        
        return self._apply_to_subclip(clip, start_time, duration, effect_func, style=style)

    def energy_aura_unified(self, clip: VideoFileClip, start_time: float, duration: float,
                           intensity: float = 1.0, pulse_rate: float = 6.0) -> VideoFileClip:
        """Apply energy aura effect with unified interface."""
        def effect_func(effect_clip, **params):
            def apply_aura(get_frame, t):
                frame = get_frame(t)
                # Create pulsing effect
                pulse_intensity = params['intensity'] * (0.8 + 0.2 * math.sin(t * params['pulse_rate']))
                return self.create_energy_aura(frame, pulse_intensity)
            
            return effect_clip.fl(apply_aura)
        
        return self._apply_to_subclip(clip, start_time, duration, effect_func, 
                                    intensity=intensity, pulse_rate=pulse_rate)

    def character_glow_unified(self, clip: VideoFileClip, start_time: float, duration: float,
                              color: Tuple[int, int, int] = (255, 255, 255), 
                              intensity: float = 1.0) -> VideoFileClip:
        """Apply character glow effect with unified interface."""
        def effect_func(effect_clip, **params):
            def apply_glow(get_frame, t):
                frame = get_frame(t)
                return self.create_character_glow(frame, params['color'], params['intensity'])
            
            return effect_clip.fl(apply_glow)
        
        return self._apply_to_subclip(clip, start_time, duration, effect_func, 
                                    color=color, intensity=intensity)

    def action_lines_unified(self, clip: VideoFileClip, start_time: float, duration: float,
                            line_count: int = 12, max_length: int = 100) -> VideoFileClip:
        """Apply action lines effect with unified interface.""" 
        def effect_func(effect_clip, **params):
            def apply_lines(get_frame, t):
                frame = get_frame(t)
                return self.add_action_lines(frame, center_point=None, 
                                           line_count=params['line_count'], 
                                           max_length=params['max_length'])
            
            return effect_clip.fl(apply_lines)
        
        return self._apply_to_subclip(clip, start_time, duration, effect_func,
                                    line_count=line_count, max_length=max_length)

    def manga_tone_unified(self, clip: VideoFileClip, start_time: float, duration: float,
                          tone_type: str = "dots") -> VideoFileClip:
        """Apply manga screen tone effect with unified interface."""
        def effect_func(effect_clip, **params):
            def apply_tone(get_frame, t):
                frame = get_frame(t)
                return self.manga_screen_tone(frame, params['tone_type'])
            
            return effect_clip.fl(apply_tone)
        
        return self._apply_to_subclip(clip, start_time, duration, effect_func,
                                    tone_type=tone_type)

    def _apply_to_subclip(self, full_clip: VideoFileClip, start_time: float, 
                         duration: float, effect_func, **params) -> VideoFileClip:
        """Helper method to apply effects to a specific time range within a clip."""
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
                logger.warning(f"Invalid duration {duration}")
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
                return concatenate_videoclips(parts)
                
        except Exception as e:
            logger.error(f"Error applying effect to subclip: {e}")
            return full_clip