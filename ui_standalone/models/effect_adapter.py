"""
Effect Adapter Layer - Bridges UI models with production video effects engines
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Type
from dataclasses import dataclass, field
import uuid

# Add src to path for importing production engines
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from video_effects.anime_effects import AnimeEffectsLibrary
from video_effects.audio_sync import AudioSyncEngine
from video_effects.color_effects import ColorEffectsEngine
from video_effects.motion_effects import MotionEffectsEngine
from video_effects.text_effects import TextEffectsEngine
from video_effects.transitions import TransitionEngine

from .effect_models import BaseEffect, EffectCategory, EffectParameter

@dataclass
class ProductionEffect(BaseEffect):
    """Enhanced base effect that connects to production engines"""
    engine_class: Optional[Type] = None
    engine_method: Optional[str] = None
    engine_config_key: Optional[str] = None
    
    def __post_init__(self):
        """Initialize engine instance if not already set"""
        if self.engine_class and not hasattr(self, '_engine_instance'):
            self._engine_instance = self.engine_class()
    
    def get_engine_instance(self):
        """Get the production engine instance"""
        if not hasattr(self, '_engine_instance') and self.engine_class:
            self._engine_instance = self.engine_class()
        return getattr(self, '_engine_instance', None)
    
    def to_engine_config(self) -> Dict[str, Any]:
        """Convert UI parameters to engine configuration format"""
        config = {}
        for param_name, param in self.parameters.items():
            # Convert UI parameter names to engine parameter names
            engine_param_name = self._map_ui_to_engine_param(param_name)
            config[engine_param_name] = self._convert_parameter_value(param)
        return config
    
    def _map_ui_to_engine_param(self, ui_param: str) -> str:
        """Map UI parameter names to engine parameter names"""
        # Common mappings
        param_mapping = {
            'intensity': 'intensity',
            'duration': 'duration',
            'zoom_factor': 'zoom_factor',
            'shake_intensity': 'shake_intensity',
            'speed_points': 'speed_points',
            'direction': 'direction',
            'style': 'style',
            'color': 'color',
            'threshold': 'threshold',
            'blur_size': 'blur_size',
            'text': 'text',
            'fontsize': 'fontsize',
            'position': 'position'
        }
        return param_mapping.get(ui_param, ui_param)
    
    def _convert_parameter_value(self, param: EffectParameter) -> Any:
        """Convert UI parameter value to engine-compatible format"""
        value = param.value
        
        # Handle special conversions
        if param.param_type == "string":
            if param.name.lower() == "color" and isinstance(value, str):
                # Convert "255,255,255" to (255, 255, 255)
                if "," in value:
                    try:
                        return tuple(map(int, value.split(",")))
                    except ValueError:
                        return value
            elif param.name.lower() == "speed_points" and isinstance(value, str):
                # Convert "0,1.0;2,0.3;4,2.0" to [(0, 1.0), (2, 0.3), (4, 2.0)]
                try:
                    points = []
                    for point_str in value.split(";"):
                        time_str, speed_str = point_str.split(",")
                        points.append((float(time_str), float(speed_str)))
                    return points
                except ValueError:
                    return value
        
        return value

class EffectEngineRegistry:
    """Registry for production engine instances and their capabilities"""
    
    def __init__(self):
        self.engines = {
            'motion': MotionEffectsEngine(),
            'anime': AnimeEffectsLibrary(),
            'color': ColorEffectsEngine(),
            'text': TextEffectsEngine(),
            'audio_sync': AudioSyncEngine(),
            'transitions': TransitionEngine()
        }
        
        # Build effect registry from production engines
        self._build_effect_registry()
    
    def _build_effect_registry(self):
        """Build comprehensive effect registry from production engines"""
        self.effect_registry = {}
        
        # Motion Effects - use actual method names from MotionEffectsEngine
        motion_effects = {
            'speed_ramp': {'method': 'speed_ramp_effect', 'params': {'speed_points': [(0, 1.0), (2, 1.5), (4, 1.0)]}},
            'zoom_punch': {'method': 'zoom_punch_effect', 'params': {'zoom_time': 2.0, 'zoom_factor': 1.5, 'duration': 0.2}},
            'camera_shake': {'method': 'camera_shake_effect', 'params': {'shake_intensity': 10, 'shake_duration': 1.0}},
            'motion_blur': {'method': 'motion_blur_effect', 'params': {'blur_strength': 5.0, 'motion_angle': 0.0}},
            'freeze_frame': {'method': 'freeze_frame_effect', 'params': {'freeze_time': 1.0, 'freeze_duration': 2.0}}
        }
        
        for effect_name, effect_info in motion_effects.items():
            self.effect_registry[f"motion_{effect_name}"] = {
                'engine': 'motion',
                'method': effect_info['method'],
                'category': EffectCategory.MOTION,
                'preset_config': effect_info['params']
            }
        
        # Anime Effects - use actual method names from AnimeEffectsLibrary
        anime_effects = {
            # Clip-level effects (work with VideoFileClip)
            'speed_lines': {'method': 'speed_lines_clip', 'params': {'direction': 'right', 'start_time': 0, 'duration': 0.5, 'intensity': 0.8}},
            'impact_frame': {'method': 'add_impact_frames', 'params': {'impact_times': [1.0], 'duration': 0.1, 'style': 'manga'}},
            'energy_aura': {'method': 'energy_aura_effect', 'params': {'start_time': 0, 'duration': 2.0, 'intensity': 1.0, 'pulse_rate': 6.0}},
            
            # Frame-level effects (work with numpy arrays)
            'speed_lines_frame': {'method': 'add_speed_lines', 'params': {'direction': 'right', 'intensity': 0.8}, 'frame_level': True},
            'impact_frame_direct': {'method': 'create_impact_frame', 'params': {'style': 'manga'}, 'frame_level': True},
            'character_glow': {'method': 'create_character_glow', 'params': {'color': (255, 255, 255), 'intensity': 1.0}, 'frame_level': True},
            'action_lines': {'method': 'add_action_lines', 'params': {'direction': 'converging', 'intensity': 0.8}, 'frame_level': True}
        }
        
        for effect_name, effect_info in anime_effects.items():
            self.effect_registry[f"anime_{effect_name}"] = {
                'engine': 'anime',
                'method': effect_info['method'],
                'category': EffectCategory.ANIME,
                'preset_config': effect_info['params'],
                'frame_level': effect_info.get('frame_level', False)
            }
        
        # Color Effects - use actual method names from ColorEffectsEngine  
        color_effects = {
            'color_grading': {'method': 'apply_color_grading', 'params': {'style': 'vibrant'}},
            'chromatic_aberration': {'method': 'chromatic_aberration_effect', 'params': {'intensity': 5}, 'frame_level': True},
            'bloom': {'method': 'bloom_effect', 'params': {'threshold': 200, 'blur_size': 15}, 'frame_level': True},
            'vintage_vhs': {'method': 'vintage_vhs_effect', 'params': {}, 'frame_level': True}
        }
        
        for effect_name, effect_info in color_effects.items():
            self.effect_registry[f"color_{effect_name}"] = {
                'engine': 'color',
                'method': effect_info['method'],
                'category': EffectCategory.COLOR,
                'preset_config': effect_info['params'],
                'frame_level': effect_info.get('frame_level', False)
            }
        
        # Text Effects - use actual method names from TextEffectsEngine
        text_effects = {
            'animated_text': {'method': 'create_animated_text', 'params': {'text': 'AMAZING!', 'duration': 2.0, 'animation': 'slide_in'}},
            'sound_fx_text': {'method': 'sound_effect_text', 'params': {'text': 'BOOM!', 'position': 'center', 'style': 'impact'}},
            'character_intro': {'method': 'character_introduction_text', 'params': {'character_name': 'Hero', 'duration': 3.0}},
            'technique_name': {'method': 'technique_name_display', 'params': {'technique_name': 'Special Attack', 'style': 'dramatic'}}
        }
        
        for effect_name, effect_info in text_effects.items():
            self.effect_registry[f"text_{effect_name}"] = {
                'engine': 'text',
                'method': effect_info['method'],
                'category': EffectCategory.TEXT,
                'preset_config': effect_info['params']
            }
        
        # Audio Sync Effects - use actual method names from AudioSyncEngine
        audio_effects = {
            'beat_flash': {'method': 'create_beat_flash', 'params': {'beat_time': 1.0, 'intensity': 0.5, 'duration': 0.1}},
            'beat_zoom': {'method': 'create_beat_zoom', 'params': {'beat_time': 1.0, 'zoom_factor': 1.2, 'duration': 0.2}},
            'beat_color_pulse': {'method': 'create_beat_color_pulse', 'params': {'beat_time': 1.0, 'color_shift': (1.2, 1.0, 1.0), 'duration': 0.15}}
        }
        
        for effect_name, effect_info in audio_effects.items():
            self.effect_registry[f"audio_{effect_name}"] = {
                'engine': 'audio_sync',
                'method': effect_info['method'],
                'category': EffectCategory.AUDIO_SYNC,
                'preset_config': effect_info['params']
            }
        
        # Transition Effects - use actual method names from TransitionEngine
        transition_effects = {
            'iris': {'method': 'iris_transition', 'params': {'duration': 1.0, 'center': None}},
            'swipe': {'method': 'swipe_transition', 'params': {'direction': 'left', 'duration': 0.5}},
            'dissolve': {'method': 'dissolve_transition', 'params': {'duration': 1.0}},
            'zoom': {'method': 'zoom_transition', 'params': {'zoom_in': True, 'duration': 1.0}},
            'spiral': {'method': 'spiral_transition', 'params': {'duration': 1.5, 'clockwise': True}},
            'slice': {'method': 'slice_transition', 'params': {'slice_count': 10, 'duration': 0.8}},
            'pixelate': {'method': 'pixelate_transition', 'params': {'duration': 1.0, 'max_pixel_size': 20}},
            'radial': {'method': 'radial_transition', 'params': {'segments': 8, 'duration': 1.0}}
        }
        
        for effect_name, effect_info in transition_effects.items():
            self.effect_registry[f"transition_{effect_name}"] = {
                'engine': 'transitions',
                'method': effect_info['method'],
                'category': EffectCategory.TRANSITIONS,
                'preset_config': effect_info['params']
            }
    
    def get_available_effects(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all available effects grouped by category"""
        effects_by_category = {}
        
        for effect_id, effect_info in self.effect_registry.items():
            category = effect_info['category'].value
            
            if category not in effects_by_category:
                effects_by_category[category] = []
            
            # Create user-friendly name
            name_parts = effect_id.split('_')[1:]  # Remove engine prefix
            display_name = ' '.join(word.title() for word in name_parts)
            
            effects_by_category[category].append({
                'id': effect_id,
                'name': display_name,
                'engine': effect_info['engine'],
                'method': effect_info['method'],
                'description': f"{display_name} - {category}",
                'preset_config': effect_info.get('preset_config', {})
            })
        
        return effects_by_category
    
    def create_effect_from_registry(self, effect_id: str) -> ProductionEffect:
        """Create a ProductionEffect instance from registry"""
        if effect_id not in self.effect_registry:
            raise ValueError(f"Unknown effect ID: {effect_id}")
        
        effect_info = self.effect_registry[effect_id]
        engine_class = self._get_engine_class(effect_info['engine'])
        
        # Create effect instance
        effect = ProductionEffect(
            effect_id=str(uuid.uuid4()),
            name=effect_id.replace('_', ' ').title(),
            category=effect_info['category'],
            engine_class=engine_class,
            engine_method=effect_info['method'],
            parameters=self._create_parameters_from_preset(effect_info.get('preset_config', {}))
        )
        
        return effect
    
    def _get_engine_class(self, engine_name: str) -> Type:
        """Get engine class by name"""
        engine_classes = {
            'motion': MotionEffectsEngine,
            'anime': AnimeEffectsLibrary,
            'color': ColorEffectsEngine,
            'text': TextEffectsEngine,
            'audio_sync': AudioSyncEngine,
            'transitions': TransitionEngine
        }
        return engine_classes[engine_name]
    
    def _create_parameters_from_preset(self, preset_config: Dict[str, Any]) -> Dict[str, EffectParameter]:
        """Create UI parameters from engine preset configuration"""
        parameters = {}
        
        for param_name, param_value in preset_config.items():
            param_type = self._infer_parameter_type(param_value)
            
            parameters[param_name] = EffectParameter(
                name=param_name.replace('_', ' ').title(),
                value=param_value,
                param_type=param_type,
                description=f"{param_name.replace('_', ' ').title()} parameter"
            )
            
            # Add constraints based on parameter name and type
            if param_type in ['float', 'int']:
                parameters[param_name].min_value = 0.0 if param_type == 'float' else 0
                parameters[param_name].max_value = 10.0 if param_type == 'float' else 100
                parameters[param_name].step = 0.1 if param_type == 'float' else 1
        
        return parameters
    
    def _infer_parameter_type(self, value: Any) -> str:
        """Infer parameter type from value"""
        if isinstance(value, bool):
            return 'bool'
        elif isinstance(value, int):
            return 'int'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, (list, tuple)):
            return 'string'  # Will be converted appropriately
        else:
            return 'string'

# Global registry instance
effect_registry = EffectEngineRegistry()

class ProductionEffectFactory:
    """Factory for creating production-integrated effects"""
    
    @classmethod
    def create_effect(cls, effect_id: str) -> ProductionEffect:
        """Create effect instance using production engines"""
        return effect_registry.create_effect_from_registry(effect_id)
    
    @classmethod
    def get_available_effects(cls) -> Dict[str, List[Dict[str, Any]]]:
        """Get all available effects from production engines"""
        return effect_registry.get_available_effects()
    
    @classmethod
    def apply_effect_to_clip(cls, clip, effect: ProductionEffect):
        """Apply production effect to video clip with unified interface"""
        engine = effect.get_engine_instance()
        if not engine or not effect.engine_method:
            raise ValueError(f"Cannot apply effect {effect.name}: missing engine or method")
        
        # Get the method from the engine
        method = getattr(engine, effect.engine_method, None)
        if not method:
            raise ValueError(f"Method {effect.engine_method} not found in engine")
        
        # Convert UI parameters to engine format
        config = effect.to_engine_config()
        
        # Get effect info from registry for special handling
        effect_key = effect.name.lower().replace(' ', '_')
        effect_info = effect_registry.effect_registry.get(effect_key, {})
        
        # Apply effect with unified interface handling
        try:
            return cls._apply_effect_with_unified_interface(clip, method, config, effect, effect_info)
                
        except Exception as e:
            print(f"Error applying effect {effect.name}: {e}")
            import traceback
            traceback.print_exc()
            return clip  # Return original clip on error
    
    @classmethod
    def _apply_effect_with_unified_interface(cls, clip, method, config, effect: ProductionEffect, effect_info: dict):
        """Apply effect with proper interface handling for clips vs frames"""
        
        # Handle frame-level effects (expects numpy arrays)
        if effect_info.get('frame_level', False):
            return cls._apply_frame_level_effect(clip, method, config)
        
        # Handle special effect patterns by category
        if effect.category == EffectCategory.MOTION:
            return cls._apply_motion_effect(clip, method, config)
        
        elif effect.category == EffectCategory.ANIME:
            return cls._apply_anime_effect(clip, method, config, effect)
        
        elif effect.category == EffectCategory.COLOR:
            return cls._apply_color_effect(clip, method, config)
        
        elif effect.category == EffectCategory.TEXT:
            return cls._apply_text_effect(clip, method, config, effect)
        
        elif effect.category == EffectCategory.AUDIO_SYNC:
            return cls._apply_audio_sync_effect(clip, method, config)
        
        elif effect.category == EffectCategory.TRANSITIONS:
            return cls._apply_transition_effect(clip, method, config)
        
        else:
            # Default: try direct method call with clip
            return method(clip, **config)
    
    @classmethod
    def _apply_frame_level_effect(cls, clip, method, config):
        """Apply effects that work on individual frames (numpy arrays)"""
        def process_frame(get_frame, t):
            # Get frame as numpy array
            frame = get_frame(t)
            # Apply frame-level effect
            processed_frame = method(frame, **config)
            return processed_frame
        
        return clip.fl(process_frame)
    
    @classmethod
    def _apply_motion_effect(cls, clip, method, config):
        """Apply motion effects (work with clips directly)"""
        return method(clip, **config)
    
    @classmethod
    def _apply_anime_effect(cls, clip, method, config, effect):
        """Apply anime effects with special parameter handling"""
        method_name = effect.engine_method
        
        if method_name == 'speed_lines_clip':
            return method(clip, **config)
        elif method_name == 'add_impact_frames':
            # Special handling for impact frames - needs impact_times as list
            impact_times = config.get('impact_times', [1.0])
            duration = config.get('duration', 0.1)
            style = config.get('style', 'manga')
            return method(clip, impact_times, duration, style)
        elif method_name == 'energy_aura_effect':
            # Special handling for energy aura
            start_time = config.get('start_time', 0)
            duration = config.get('duration', 2.0)
            intensity = config.get('intensity', 1.0)
            pulse_rate = config.get('pulse_rate', 6.0)
            return method(clip, start_time, duration, intensity, pulse_rate)
        else:
            return method(clip, **config)
    
    @classmethod
    def _apply_color_effect(cls, clip, method, config):
        """Apply color effects (mostly clip-level)"""
        return method(clip, **config)
    
    @classmethod
    def _apply_text_effect(cls, clip, method, config, effect):
        """Apply text effects with composition"""
        try:
            from moviepy.editor import CompositeVideoClip
            text_clip = method(**config)
            if hasattr(text_clip, 'set_duration'):
                text_clip = text_clip.set_duration(effect.duration or clip.duration)
            if hasattr(text_clip, 'set_start'):
                text_clip = text_clip.set_start(effect.start_time)
            return CompositeVideoClip([clip, text_clip])
        except Exception as e:
            print(f"Text effect composition failed: {e}")
            return clip
    
    @classmethod
    def _apply_audio_sync_effect(cls, clip, method, config):
        """Apply audio sync effects"""
        return method(clip, **config)
    
    @classmethod
    def _apply_transition_effect(cls, clip, method, config):
        """Apply transition effects (need two clips)"""
        # For now, use the same clip as both input and output
        # In a full implementation, this would get the next clip in sequence
        return method(clip, clip, **config)