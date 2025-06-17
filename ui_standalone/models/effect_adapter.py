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
    start_time: float = 0.0
    duration: float = 1.0
    
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
            # Skip timing parameters as they're handled separately
            if param_name in ['start_time', 'duration']:
                continue
                
            # Convert UI parameter names to engine parameter names
            engine_param_name = self._map_ui_to_engine_param(param_name)
            config[engine_param_name] = self._convert_parameter_value(param)
        
        # Update effect timing from parameters
        if 'start_time' in self.parameters:
            self.start_time = self.parameters['start_time'].value
        if 'duration' in self.parameters:
            self.duration = self.parameters['duration'].value
            
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
        
        # Motion Effects - use unified interface methods  
        motion_effects = {
            'speed_ramp': {'method': 'speed_ramp_unified', 'params': {'speed_points': [(0, 1.0), (0.5, 1.5), (1.0, 1.0)]}},
            'zoom_punch': {'method': 'zoom_punch_unified', 'params': {'zoom_factor': 1.5}},
            'camera_shake': {'method': 'camera_shake_unified', 'params': {'shake_intensity': 10}},
            'motion_blur': {'method': 'motion_blur_unified', 'params': {'blur_strength': 5.0, 'motion_angle': 0.0}},
            'freeze_frame': {'method': 'freeze_frame_unified', 'params': {'freeze_duration': 1.0}}
        }
        
        for effect_name, effect_info in motion_effects.items():
            self.effect_registry[f"motion_{effect_name}"] = {
                'engine': 'motion',
                'method': effect_info['method'],
                'category': EffectCategory.MOTION,
                'preset_config': effect_info['params']
            }
        
        # Anime Effects - use unified interface methods
        anime_effects = {
            'speed_lines': {'method': 'speed_lines_unified', 'params': {'direction': 'right', 'intensity': 0.8}},
            'impact_frame': {'method': 'impact_frames_unified', 'params': {'style': 'manga'}},
            'energy_aura': {'method': 'energy_aura_unified', 'params': {'intensity': 1.0, 'pulse_rate': 6.0}},
            'character_glow': {'method': 'character_glow_unified', 'params': {'color': (255, 255, 255), 'intensity': 1.0}},
            'action_lines': {'method': 'action_lines_unified', 'params': {'direction': 'converging', 'intensity': 0.8}}
        }
        
        for effect_name, effect_info in anime_effects.items():
            self.effect_registry[f"anime_{effect_name}"] = {
                'engine': 'anime',
                'method': effect_info['method'],
                'category': EffectCategory.ANIME,
                'preset_config': effect_info['params']
            }
        
        # Color Effects - use unified interface methods
        color_effects = {
            'color_grading': {'method': 'color_grading_unified', 'params': {'style': 'vibrant'}},
            'chromatic_aberration': {'method': 'chromatic_aberration_unified', 'params': {'intensity': 5}},
            'bloom': {'method': 'bloom_unified', 'params': {'threshold': 200, 'blur_size': 15}},
            'vintage_vhs': {'method': 'vintage_vhs_unified', 'params': {}}
        }
        
        for effect_name, effect_info in color_effects.items():
            self.effect_registry[f"color_{effect_name}"] = {
                'engine': 'color',
                'method': effect_info['method'],
                'category': EffectCategory.COLOR,
                'preset_config': effect_info['params']
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
        
        # Add standard timing parameters
        parameters['start_time'] = EffectParameter(
            name='Start Time',
            value=0.0,
            param_type='float',
            description='Start time of the effect (seconds)',
            min_value=0.0,
            max_value=300.0,  # 5 minutes max
            step=0.1
        )
        
        parameters['duration'] = EffectParameter(
            name='Duration', 
            value=1.0,
            param_type='float',
            description='Duration of the effect (seconds)',
            min_value=0.1,
            max_value=60.0,  # 1 minute max
            step=0.1
        )
        
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
        
        # Apply effect using unified interface (all methods now expect clip, start_time, duration, **params)
        try:
            return method(clip, effect.start_time, effect.duration, **config)
                
        except Exception as e:
            print(f"Error applying effect {effect.name}: {e}")
            import traceback
            traceback.print_exc()
            return clip  # Return original clip on error
    
