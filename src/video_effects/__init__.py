"""
Video Effects Library for Anime Content Processing

This module provides comprehensive video effects specifically designed for anime content,
including motion effects, anime-style visual effects, color grading, text effects,
and audio-visual synchronization.
"""

from .motion_effects import MotionEffectsEngine
from .anime_effects import AnimeEffectsLibrary
from .color_effects import ColorEffectsEngine
from .text_effects import TextEffectsEngine
from .audio_sync import AudioSyncEngine
from .transitions import TransitionEngine

__all__ = [
    'MotionEffectsEngine',
    'AnimeEffectsLibrary', 
    'ColorEffectsEngine',
    'TextEffectsEngine',
    'AudioSyncEngine',
    'TransitionEngine'
]

__version__ = '1.0.0'