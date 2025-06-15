"""
Text and Typography Effects Engine

Provides advanced text overlay effects including animated text, sound effect text,
character introductions, and dynamic typography for anime content.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import logging


# MoviePy 1.x imports (fallback)
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip
from moviepy.video.fx import speedx

from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})


logger = logging.getLogger(__name__)


class TextEffectsEngine:
    """Engine for creating advanced text and typography effects."""
    
    def __init__(self):
        """Initialize the text effects engine."""
        self.anime_fonts = self.get_anime_fonts()
        self.text_animations = self.get_text_animations()
        self.default_settings = {
            'font_size': 50,
            'stroke_width': 2,
            'default_color': 'white',
            'default_stroke_color': 'black'
        }
    
    def get_anime_fonts(self) -> List[str]:
        """Get list of anime-style fonts."""
        return [
            'Arial-Bold',
            'Impact',
            'Comic Sans MS',
            'Trebuchet MS',
            'Verdana-Bold'
        ]
    
    def get_text_animations(self) -> Dict[str, Callable]:
        """Get available text animation functions."""
        return {
            'slide_in': self._slide_in_animation,
            'typewriter': self._typewriter_animation,
            'bounce': self._bounce_animation,
            'fade_in': self._fade_in_animation,
            'zoom_in': self._zoom_in_animation,
            'glitch': self._glitch_animation
        }
    
    def create_animated_text(self, text: str, duration: float, animation: str = "slide_in",
                           fontsize: int = None, color: str = None, position: Tuple[int, int] = None) -> CompositeVideoClip:
        """
        Create animated text overlay.
        
        Args:
            text: Text content
            duration: Duration of text display
            animation: Animation type
            fontsize: Font size
            color: Text color
            position: Text position (None for default)
            
        Returns:
            Animated text clip
        """
        try:
            if fontsize is None:
                fontsize = self.default_settings['font_size']
            if color is None:
                color = self.default_settings['default_color']
            
            if animation in self.text_animations:
                return self.text_animations[animation](text, duration, fontsize, color, position)
            else:
                logger.warning(f"Unknown animation type: {animation}, using slide_in")
                return self._slide_in_animation(text, duration, fontsize, color, position)
                
        except Exception as e:
            logger.error(f"Error creating animated text: {e}")
            # Return simple text as fallback
            return TextClip(text, fontsize=fontsize, color=color).set_duration(duration)
    
    def _slide_in_animation(self, text: str, duration: float, fontsize: int, 
                           color: str, position: Tuple[int, int]) -> CompositeVideoClip:
        """Create slide-in text animation."""
        def text_position(t):
            if t < 0.5:
                # Slide in from right
                progress = t / 0.5
                x = int(800 * (1 - progress))
                return (x, 100) if position is None else (position[0] + x - 50, position[1])
            else:
                return (50, 100) if position is None else position
        
        text_clip = TextClip(text, fontsize=fontsize, color=color,
                           stroke_color=self.default_settings['default_stroke_color'],
                           stroke_width=self.default_settings['stroke_width'])
        
        return text_clip.set_position(text_position).set_duration(duration)
    
    def _typewriter_animation(self, text: str, duration: float, fontsize: int,
                             color: str, position: Tuple[int, int]) -> CompositeVideoClip:
        """Create typewriter text animation."""
        clips = []
        chars_per_second = max(1, len(text) / duration)
        
        for i in range(1, len(text) + 1):
            partial_text = text[:i]
            start_time = (i - 1) / chars_per_second
            duration_part = 1 / chars_per_second if i < len(text) else duration - start_time
            
            text_part = TextClip(partial_text, fontsize=fontsize, color=color,
                               stroke_color=self.default_settings['default_stroke_color'],
                               stroke_width=self.default_settings['stroke_width'])
            
            if position:
                text_part = text_part.set_position(position)
            
            text_part = text_part.set_start(start_time).set_duration(duration_part)
            clips.append(text_part)
        
        return CompositeVideoClip(clips)
    
    def _bounce_animation(self, text: str, duration: float, fontsize: int,
                         color: str, position: Tuple[int, int]) -> CompositeVideoClip:
        """Create bouncing text animation."""
        def bounce_position(t):
            base_pos = position or (50, 100)
            if t < 0.3:
                # Bounce down
                bounce_offset = int(50 * np.sin(t * 10))
                return (base_pos[0], base_pos[1] + bounce_offset)
            else:
                return base_pos
        
        text_clip = TextClip(text, fontsize=fontsize, color=color,
                           stroke_color=self.default_settings['default_stroke_color'],
                           stroke_width=self.default_settings['stroke_width'])
        
        return text_clip.set_position(bounce_position).set_duration(duration)
    
    def _fade_in_animation(self, text: str, duration: float, fontsize: int,
                          color: str, position: Tuple[int, int]) -> CompositeVideoClip:
        """Create fade-in text animation."""
        text_clip = TextClip(text, fontsize=fontsize, color=color,
                           stroke_color=self.default_settings['default_stroke_color'],
                           stroke_width=self.default_settings['stroke_width'])
        
        if position:
            text_clip = text_clip.set_position(position)
        
        # Fade in over first 0.5 seconds
        fade_duration = min(0.5, duration / 2)
        text_clip = text_clip.crossfadein(fade_duration).set_duration(duration)
        
        return text_clip
    
    def _zoom_in_animation(self, text: str, duration: float, fontsize: int,
                          color: str, position: Tuple[int, int]) -> CompositeVideoClip:
        """Create zoom-in text animation."""
        def scale_func(t):
            if t < 0.3:
                return 0.1 + (0.9 * t / 0.3)  # Scale from 0.1 to 1.0
            else:
                return 1.0
        
        text_clip = TextClip(text, fontsize=fontsize, color=color,
                           stroke_color=self.default_settings['default_stroke_color'],
                           stroke_width=self.default_settings['stroke_width'])
        
        if position:
            text_clip = text_clip.set_position(position)
        
        return text_clip.resize(scale_func).set_duration(duration)
    
    def _glitch_animation(self, text: str, duration: float, fontsize: int,
                         color: str, position: Tuple[int, int]) -> CompositeVideoClip:
        """Create glitch text animation."""
        def glitch_position(t):
            base_pos = position or (50, 100)
            if t < 0.2:  # Glitch for first 0.2 seconds
                glitch_x = int(np.random.uniform(-10, 10))
                glitch_y = int(np.random.uniform(-5, 5))
                return (base_pos[0] + glitch_x, base_pos[1] + glitch_y)
            else:
                return base_pos
        
        # Create multiple colored versions for RGB split effect
        red_clip = TextClip(text, fontsize=fontsize, color='red').set_position(glitch_position)
        green_clip = TextClip(text, fontsize=fontsize, color='green').set_position(glitch_position)
        blue_clip = TextClip(text, fontsize=fontsize, color='blue').set_position(glitch_position)
        
        # Composite with slight offsets during glitch period
        def red_offset(t):
            base_pos = position or (50, 100)
            if t < 0.2:
                return (base_pos[0] - 2, base_pos[1])
            return base_pos
        
        def blue_offset(t):
            base_pos = position or (50, 100)
            if t < 0.2:
                return (base_pos[0] + 2, base_pos[1])
            return base_pos
        
        red_clip = red_clip.set_position(red_offset)
        blue_clip = blue_clip.set_position(blue_offset)
        
        return CompositeVideoClip([red_clip, green_clip, blue_clip], size=(1920, 1080)).set_duration(duration)
    
    def sound_effect_text(self, text: str, position: Tuple[int, int], style: str = "impact") -> CompositeVideoClip:
        """
        Create sound effect text overlay (like 'BOOM!', 'SLASH!', etc.).
        
        Args:
            text: Sound effect text
            position: Position of text
            style: Style of sound effect ("impact", "explosive", "electric")
            
        Returns:
            Sound effect text clip
        """
        try:
            if style == "impact":
                # Large, bold text with scaling animation
                text_clip = TextClip(text, fontsize=80, color='yellow',
                                   font='Arial-Bold', stroke_color='red', stroke_width=3)
                
                def scale_func(t):
                    if t < 0.2:
                        return 1 + (1.5 - 1) * (t / 0.2)  # Scale up
                    elif t < 0.4:
                        return 1.5 - (1.5 - 1) * ((t - 0.2) / 0.2)  # Scale down
                    else:
                        return 1
                
                return text_clip.resize(scale_func).set_position(position).set_duration(0.6)
                
            elif style == "explosive":
                # Multi-colored explosive text
                clips = []
                colors = ['red', 'orange', 'yellow', 'white']
                
                for i, color in enumerate(colors):
                    offset_x = i * 2
                    offset_y = i * 2
                    clip = TextClip(text, fontsize=70, color=color, font='Impact')
                    clip = clip.set_position((position[0] + offset_x, position[1] + offset_y))
                    clips.append(clip)
                
                return CompositeVideoClip(clips).set_duration(0.8)
                
            elif style == "electric":
                # Electric/glitch style text
                def electric_effect(t):
                    if t < 0.3:
                        # Random color changes for electric effect
                        colors = ['cyan', 'white', 'blue', 'purple']
                        return colors[int(t * 20) % len(colors)]
                    return 'cyan'
                
                # Note: This is a simplified version - actual implementation would need
                # frame-by-frame color changes
                text_clip = TextClip(text, fontsize=75, color='cyan',
                                   font='Arial-Bold', stroke_color='blue', stroke_width=2)
                
                return text_clip.set_position(position).set_duration(0.5)
                
        except Exception as e:
            logger.error(f"Error creating sound effect text: {e}")
            # Simple fallback
            return TextClip(text, fontsize=60, color='white').set_position(position).set_duration(0.5)
    
    def character_introduction_text(self, character_name: str, anime_title: str = None,
                                  duration: float = 3.0) -> CompositeVideoClip:
        """
        Create character introduction text overlay.
        
        Args:
            character_name: Name of the character
            anime_title: Title of the anime (optional)
            duration: Duration of display
            
        Returns:
            Character introduction text clip
        """
        try:
            clips = []
            
            # Main character name
            name_clip = TextClip(character_name, fontsize=60, color='white',
                               font='Arial-Bold', stroke_color='black', stroke_width=3)
            name_clip = name_clip.set_position(('center', 'center')).set_duration(duration)
            
            # Fade in animation
            name_clip = name_clip.crossfadein(0.5)
            clips.append(name_clip)
            
            # Anime title (smaller, below name)
            if anime_title:
                title_clip = TextClip(f"from {anime_title}", fontsize=30, color='lightgray',
                                    font='Arial', stroke_color='black', stroke_width=1)
                title_clip = title_clip.set_position(('center', 'center')).set_start(0.5).set_duration(duration - 0.5)
                
                # Position below main name
                def title_position(t):
                    return ('center', 'center')  # Will be offset in composite
                
                clips.append(title_clip)
            
            return CompositeVideoClip(clips, size=(1920, 1080))
            
        except Exception as e:
            logger.error(f"Error creating character introduction: {e}")
            return TextClip(character_name, fontsize=50, color='white').set_duration(duration)
    
    def technique_name_display(self, technique_name: str, character_name: str = None,
                             style: str = "dramatic") -> CompositeVideoClip:
        """
        Create technique/attack name display overlay.
        
        Args:
            technique_name: Name of the technique/attack
            character_name: Character using the technique
            style: Display style ("dramatic", "elegant", "powerful")
            
        Returns:
            Technique name display clip
        """
        try:
            if style == "dramatic":
                # Large text with zoom and glow effect
                main_clip = TextClip(technique_name, fontsize=70, color='gold',
                                   font='Impact', stroke_color='darkred', stroke_width=4)
                
                # Zoom in animation
                def scale_func(t):
                    if t < 0.3:
                        return 0.5 + 0.5 * (t / 0.3)
                    return 1.0
                
                main_clip = main_clip.resize(scale_func).set_position('center').set_duration(2.0)
                
                clips = [main_clip]
                
                # Character name (smaller, above technique)
                if character_name:
                    char_clip = TextClip(character_name, fontsize=40, color='white',
                                       font='Arial-Bold', stroke_color='black', stroke_width=2)
                    char_clip = char_clip.set_position(('center', 200)).set_duration(2.0)
                    clips.append(char_clip)
                
                return CompositeVideoClip(clips, size=(1920, 1080))
                
            elif style == "elegant":
                # Flowing, elegant text
                main_clip = TextClip(technique_name, fontsize=60, color='lightblue',
                                   font='Trebuchet MS', stroke_color='navy', stroke_width=2)
                
                # Fade in with slight movement
                def elegant_position(t):
                    if t < 0.5:
                        y_offset = int(20 * (1 - t / 0.5))
                        return ('center', 300 + y_offset)
                    return ('center', 300)
                
                main_clip = main_clip.set_position(elegant_position).crossfadein(0.5).set_duration(2.5)
                
                return main_clip
                
            elif style == "powerful":
                # Bold, impactful text with shake effect
                main_clip = TextClip(technique_name, fontsize=80, color='red',
                                   font='Impact', stroke_color='black', stroke_width=5)
                
                # Shake effect
                def shake_position(t):
                    if t < 0.3:
                        shake_x = int(np.random.uniform(-5, 5))
                        shake_y = int(np.random.uniform(-3, 3))
                        return ('center', 400 + shake_y)
                    return ('center', 400)
                
                main_clip = main_clip.set_position(shake_position).set_duration(1.5)
                
                return main_clip
                
        except Exception as e:
            logger.error(f"Error creating technique display: {e}")
            return TextClip(technique_name, fontsize=60, color='white').set_duration(2.0)
    
    def create_subtitle_effect(self, text: str, start_time: float, duration: float,
                              style: str = "standard") -> CompositeVideoClip:
        """
        Create styled subtitle text.
        
        Args:
            text: Subtitle text
            start_time: When subtitle appears
            duration: How long subtitle is displayed
            style: Subtitle style ("standard", "anime", "dramatic")
            
        Returns:
            Subtitle text clip
        """
        try:
            if style == "standard":
                subtitle_clip = TextClip(text, fontsize=40, color='white',
                                       font='Arial', stroke_color='black', stroke_width=2)
                
            elif style == "anime":
                subtitle_clip = TextClip(text, fontsize=45, color='yellow',
                                       font='Comic Sans MS', stroke_color='purple', stroke_width=3)
                
            elif style == "dramatic":
                subtitle_clip = TextClip(text, fontsize=50, color='gold',
                                       font='Arial-Bold', stroke_color='darkred', stroke_width=3)
            
            # Position at bottom center
            subtitle_clip = subtitle_clip.set_position(('center', 'bottom')).set_start(start_time).set_duration(duration)
            
            # Add fade in/out
            if duration > 1.0:
                subtitle_clip = subtitle_clip.crossfadein(0.2).crossfadeout(0.2)
            
            return subtitle_clip
            
        except Exception as e:
            logger.error(f"Error creating subtitle: {e}")
            return TextClip(text, fontsize=40, color='white').set_start(start_time).set_duration(duration)
    
    def get_text_effect_presets(self) -> Dict[str, Any]:
        """
        Get predefined text effect presets.
        
        Returns:
            Dictionary of preset configurations
        """
        return {
            'action_scene': {
                'sound_effects': ['SLASH!', 'BOOM!', 'POW!'],
                'style': 'impact',
                'font_size': 80
            },
            'character_focus': {
                'animation': 'zoom_in',
                'font_size': 60,
                'color': 'gold'
            },
            'dramatic_moment': {
                'animation': 'fade_in',
                'technique_style': 'dramatic',
                'font_size': 70
            },
            'comedy_scene': {
                'animation': 'bounce',
                'color': 'yellow',
                'font_size': 55
            },
            'mysterious': {
                'animation': 'glitch',
                'color': 'cyan',
                'font_size': 50
            }
        }