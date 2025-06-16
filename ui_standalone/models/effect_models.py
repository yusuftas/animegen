"""
Effect Models - Data models for individual video effects
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import uuid

class EffectCategory(Enum):
    """Categories of video effects"""
    MOTION = "Motion Effects"
    ANIME = "Anime Effects"
    COLOR = "Color Effects"
    TEXT = "Text Effects"
    AUDIO_SYNC = "Audio Sync"
    TRANSITIONS = "Transitions"

class AnimationType(Enum):
    """Text animation types"""
    SLIDE_IN = "slide_in"
    TYPEWRITER = "typewriter"
    BOUNCE = "bounce"
    FADE_IN = "fade_in"
    ZOOM_IN = "zoom_in"
    GLITCH = "glitch"

class ColorGradingStyle(Enum):
    """Color grading styles"""
    VIBRANT = "vibrant"
    SUNSET = "sunset"
    MOONLIGHT = "moonlight"
    DRAMATIC = "dramatic"

class TransitionType(Enum):
    """Transition types"""
    IRIS = "iris"
    SWIPE = "swipe"
    DISSOLVE = "dissolve"
    ZOOM = "zoom"
    SPIRAL = "spiral"
    SLICE = "slice"
    PIXELATE = "pixelate"
    RADIAL = "radial"

@dataclass
class EffectParameter:
    """Individual effect parameter"""
    name: str
    value: Any
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    param_type: str = "float"  # "float", "int", "bool", "string", "enum"
    options: Optional[List[str]] = None  # For enum/dropdown parameters
    description: str = ""

@dataclass
class BaseEffect:
    """Base class for all video effects"""
    effect_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: EffectCategory = EffectCategory.MOTION
    start_time: float = 0.0  # Start time in seconds
    duration: Optional[float] = None  # Duration in seconds, None = entire clip
    enabled: bool = True
    parameters: Dict[str, EffectParameter] = field(default_factory=dict)
    
    def get_parameter_value(self, param_name: str) -> Any:
        """Get parameter value by name"""
        if param_name in self.parameters:
            return self.parameters[param_name].value
        return None
    
    def set_parameter_value(self, param_name: str, value: Any):
        """Set parameter value by name"""
        if param_name in self.parameters:
            self.parameters[param_name].value = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert effect to dictionary for serialization"""
        return {
            "effect_id": self.effect_id,
            "name": self.name,
            "category": self.category.value,
            "start_time": self.start_time,
            "duration": self.duration,
            "enabled": self.enabled,
            "parameters": {
                name: {
                    "name": param.name,
                    "value": param.value,
                    "min_value": param.min_value,
                    "max_value": param.max_value,
                    "step": param.step,
                    "param_type": param.param_type,
                    "options": param.options,
                    "description": param.description
                }
                for name, param in self.parameters.items()
            }
        }

# Motion Effects
@dataclass
class SpeedRampEffect(BaseEffect):
    """Speed ramping effect for dynamic speed changes"""
    
    def __init__(self):
        super().__init__()
        self.name = "Speed Ramp"
        self.category = EffectCategory.MOTION
        self.parameters = {
            "speed_points": EffectParameter(
                name="Speed Points",
                value="0,1.0;2,0.3;4,2.0;6,1.0",
                param_type="string",
                description="Time,Speed pairs separated by semicolons"
            )
        }

@dataclass
class ZoomPunchEffect(BaseEffect):
    """Zoom punch effect for impact moments"""
    
    def __init__(self):
        super().__init__()
        self.name = "Zoom Punch"
        self.category = EffectCategory.MOTION
        self.parameters = {
            "zoom_factor": EffectParameter(
                name="Zoom Factor",
                value=1.5,
                min_value=1.0,
                max_value=3.0,
                step=0.1,
                param_type="float",
                description="Zoom intensity"
            ),
            "zoom_duration": EffectParameter(
                name="Duration",
                value=0.2,
                min_value=0.1,
                max_value=1.0,
                step=0.1,
                param_type="float",
                description="Effect duration in seconds"
            )
        }

@dataclass
class CameraShakeEffect(BaseEffect):
    """Camera shake effect for impact and excitement"""
    
    def __init__(self):
        super().__init__()
        self.name = "Camera Shake"
        self.category = EffectCategory.MOTION
        self.parameters = {
            "intensity": EffectParameter(
                name="Shake Intensity",
                value=10,
                min_value=1,
                max_value=30,
                step=1,
                param_type="int",
                description="Shake intensity in pixels"
            ),
            "shake_duration": EffectParameter(
                name="Duration",
                value=1.0,
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                param_type="float",
                description="Shake duration in seconds"
            )
        }

# Anime Effects
@dataclass
class SpeedLinesEffect(BaseEffect):
    """Anime-style speed lines effect"""
    
    def __init__(self):
        super().__init__()
        self.name = "Speed Lines"
        self.category = EffectCategory.ANIME
        self.parameters = {
            "direction": EffectParameter(
                name="Direction",
                value="right",
                param_type="enum",
                options=["right", "left", "radial", "diagonal"],
                description="Direction of speed lines"
            ),
            "intensity": EffectParameter(
                name="Intensity",
                value=0.8,
                min_value=0.1,
                max_value=1.0,
                step=0.1,
                param_type="float",
                description="Line intensity"
            ),
            "color": EffectParameter(
                name="Color",
                value="255,255,255",
                param_type="string",
                description="RGB color values"
            )
        }

@dataclass
class ImpactFrameEffect(BaseEffect):
    """High-contrast impact frames"""
    
    def __init__(self):
        super().__init__()
        self.name = "Impact Frame"
        self.category = EffectCategory.ANIME
        self.parameters = {
            "style": EffectParameter(
                name="Style",
                value="energy",
                param_type="enum",
                options=["manga", "energy", "flash"],
                description="Impact frame style"
            ),
            "frame_duration": EffectParameter(
                name="Frame Duration",
                value=0.2,
                min_value=0.1,
                max_value=1.0,
                step=0.1,
                param_type="float",
                description="Duration of impact frame"
            )
        }

@dataclass
class EnergyAuraEffect(BaseEffect):
    """Pulsing energy aura effect"""
    
    def __init__(self):
        super().__init__()
        self.name = "Energy Aura"
        self.category = EffectCategory.ANIME
        self.parameters = {
            "intensity": EffectParameter(
                name="Intensity",
                value=1.0,
                min_value=0.1,
                max_value=2.0,
                step=0.1,
                param_type="float",
                description="Aura intensity"
            ),
            "pulse_rate": EffectParameter(
                name="Pulse Rate",
                value=6.0,
                min_value=1.0,
                max_value=15.0,
                step=0.5,
                param_type="float",
                description="Pulses per second"
            )
        }

# Color Effects
@dataclass
class ColorGradeEffect(BaseEffect):
    """Anime-style color grading"""
    
    def __init__(self):
        super().__init__()
        self.name = "Color Grade"
        self.category = EffectCategory.COLOR
        self.parameters = {
            "style": EffectParameter(
                name="Style",
                value="vibrant",
                param_type="enum",
                options=["vibrant", "sunset", "moonlight", "dramatic"],
                description="Color grading style"
            ),
            "intensity": EffectParameter(
                name="Intensity",
                value=0.9,
                min_value=0.0,
                max_value=2.0,
                step=0.1,
                param_type="float",
                description="Effect intensity"
            )
        }

@dataclass
class ChromaticAberrationEffect(BaseEffect):
    """RGB separation effect"""
    
    def __init__(self):
        super().__init__()
        self.name = "Chromatic Aberration"
        self.category = EffectCategory.COLOR
        self.parameters = {
            "intensity": EffectParameter(
                name="Intensity",
                value=5,
                min_value=1,
                max_value=20,
                step=1,
                param_type="int",
                description="Aberration intensity"
            )
        }

@dataclass
class BloomEffect(BaseEffect):
    """Bloom/glow effect on bright areas"""
    
    def __init__(self):
        super().__init__()
        self.name = "Bloom"
        self.category = EffectCategory.COLOR
        self.parameters = {
            "threshold": EffectParameter(
                name="Threshold",
                value=200,
                min_value=100,
                max_value=255,
                step=5,
                param_type="int",
                description="Brightness threshold"
            ),
            "blur_size": EffectParameter(
                name="Blur Size",
                value=15,
                min_value=5,
                max_value=50,
                step=5,
                param_type="int",
                description="Bloom blur radius"
            )
        }

# Text Effects
@dataclass
class AnimatedTextEffect(BaseEffect):
    """Animated text overlays"""
    
    def __init__(self):
        super().__init__()
        self.name = "Animated Text"
        self.category = EffectCategory.TEXT
        self.parameters = {
            "text": EffectParameter(
                name="Text",
                value="AMAZING!",
                param_type="string",
                description="Text to display"
            ),
            "animation": EffectParameter(
                name="Animation",
                value="zoom_in",
                param_type="enum",
                options=["slide_in", "typewriter", "bounce", "fade_in", "zoom_in", "glitch"],
                description="Text animation type"
            ),
            "fontsize": EffectParameter(
                name="Font Size",
                value=50,
                min_value=12,
                max_value=120,
                step=2,
                param_type="int",
                description="Text font size"
            ),
            "color": EffectParameter(
                name="Color",
                value="255,255,255",
                param_type="string",
                description="Text color (RGB)"
            ),
            "position": EffectParameter(
                name="Position",
                value="center",
                param_type="enum",
                options=["center", "top", "bottom", "left", "right", "custom"],
                description="Text position"
            )
        }

@dataclass
class SoundEffectTextEffect(BaseEffect):
    """Sound effect text like "BOOM!", "SLASH!" """
    
    def __init__(self):
        super().__init__()
        self.name = "Sound FX Text"
        self.category = EffectCategory.TEXT
        self.parameters = {
            "text": EffectParameter(
                name="Text",
                value="BOOM!",
                param_type="string",
                description="Sound effect text"
            ),
            "style": EffectParameter(
                name="Style",
                value="impact",
                param_type="enum",
                options=["impact", "explosive", "electric"],
                description="Sound effect style"
            ),
            "scale_animation": EffectParameter(
                name="Scale Animation",
                value=True,
                param_type="bool",
                description="Enable scale animation"
            )
        }

# Audio Sync Effects
@dataclass
class BeatFlashEffect(BaseEffect):
    """Flash effects synchronized to audio beats"""
    
    def __init__(self):
        super().__init__()
        self.name = "Beat Flash"
        self.category = EffectCategory.AUDIO_SYNC
        self.parameters = {
            "intensity": EffectParameter(
                name="Flash Intensity",
                value=0.5,
                min_value=0.1,
                max_value=1.0,
                step=0.1,
                param_type="float",
                description="Flash intensity"
            ),
            "flash_duration": EffectParameter(
                name="Flash Duration",
                value=0.1,
                min_value=0.05,
                max_value=0.5,
                step=0.05,
                param_type="float",
                description="Duration of each flash"
            )
        }

@dataclass
class BeatZoomEffect(BaseEffect):
    """Zoom effects synchronized to audio beats"""
    
    def __init__(self):
        super().__init__()
        self.name = "Beat Zoom"
        self.category = EffectCategory.AUDIO_SYNC
        self.parameters = {
            "zoom_factor": EffectParameter(
                name="Zoom Factor",
                value=1.1,
                min_value=1.0,
                max_value=1.5,
                step=0.05,
                param_type="float",
                description="Zoom intensity"
            ),
            "zoom_duration": EffectParameter(
                name="Zoom Duration",
                value=0.2,
                min_value=0.1,
                max_value=0.5,
                step=0.05,
                param_type="float",
                description="Duration of each zoom"
            )
        }

# Transition Effects
@dataclass
class IrisTransitionEffect(BaseEffect):
    """Iris/circular transition effect"""
    
    def __init__(self):
        super().__init__()
        self.name = "Iris Transition"
        self.category = EffectCategory.TRANSITIONS
        self.parameters = {
            "transition_duration": EffectParameter(
                name="Duration",
                value=1.0,
                min_value=0.5,
                max_value=3.0,
                step=0.1,
                param_type="float",
                description="Transition duration"
            ),
            "center_x": EffectParameter(
                name="Center X",
                value=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                param_type="float",
                description="Iris center X (0-1)"
            ),
            "center_y": EffectParameter(
                name="Center Y",
                value=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                param_type="float",
                description="Iris center Y (0-1)"
            )
        }

@dataclass
class SwipeTransitionEffect(BaseEffect):
    """Swipe transition effect"""
    
    def __init__(self):
        super().__init__()
        self.name = "Swipe Transition"
        self.category = EffectCategory.TRANSITIONS
        self.parameters = {
            "direction": EffectParameter(
                name="Direction",
                value="left",
                param_type="enum",
                options=["left", "right", "up", "down"],
                description="Swipe direction"
            ),
            "transition_duration": EffectParameter(
                name="Duration",
                value=0.5,
                min_value=0.2,
                max_value=2.0,
                step=0.1,
                param_type="float",
                description="Transition duration"
            )
        }

# Effect Factory
class EffectFactory:
    """Factory class for creating effect instances"""
    
    EFFECT_CLASSES = {
        # Motion Effects
        "speed_ramp": SpeedRampEffect,
        "zoom_punch": ZoomPunchEffect,
        "camera_shake": CameraShakeEffect,
        
        # Anime Effects
        "speed_lines": SpeedLinesEffect,
        "impact_frame": ImpactFrameEffect,
        "energy_aura": EnergyAuraEffect,
        
        # Color Effects
        "color_grade": ColorGradeEffect,
        "chromatic_aberration": ChromaticAberrationEffect,
        "bloom": BloomEffect,
        
        # Text Effects
        "animated_text": AnimatedTextEffect,
        "sound_fx_text": SoundEffectTextEffect,
        
        # Audio Sync Effects
        "beat_flash": BeatFlashEffect,
        "beat_zoom": BeatZoomEffect,
        
        # Transition Effects
        "iris_transition": IrisTransitionEffect,
        "swipe_transition": SwipeTransitionEffect,
    }
    
    @classmethod
    def create_effect(cls, effect_type: str) -> BaseEffect:
        """Create an effect instance by type"""
        if effect_type in cls.EFFECT_CLASSES:
            return cls.EFFECT_CLASSES[effect_type]()
        else:
            raise ValueError(f"Unknown effect type: {effect_type}")
    
    @classmethod
    def get_available_effects(cls) -> Dict[str, List[str]]:
        """Get all available effects grouped by category"""
        effects_by_category = {}
        
        for effect_type, effect_class in cls.EFFECT_CLASSES.items():
            effect_instance = effect_class()
            category = effect_instance.category.value
            
            if category not in effects_by_category:
                effects_by_category[category] = []
            
            effects_by_category[category].append({
                "type": effect_type,
                "name": effect_instance.name,
                "description": f"{effect_instance.name} - {category}"
            })
        
        return effects_by_category