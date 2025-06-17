"""
Effect Pipeline - Data model for managing the video effects pipeline
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from .effect_models import BaseEffect, EffectCategory
from .effect_adapter import ProductionEffectFactory

@dataclass
class EffectPipeline:
    """Manages the ordered list of effects to apply to video"""
    
    effects: List[BaseEffect] = field(default_factory=list)
    name: str = "Untitled Pipeline"
    description: str = ""
    
    def add_effect(self, effect_type: str, effect_name: str = None, position: Optional[int] = None) -> str:
        """Add an effect to the pipeline"""
        try:
            # Use ProductionEffectFactory for all effects
            effect = ProductionEffectFactory.create_effect(effect_type)
            
            if effect_name:
                effect.name = effect_name
            
            if position is None:
                self.effects.append(effect)
            else:
                self.effects.insert(position, effect)
            
            return effect.effect_id
        except ValueError as e:
            raise ValueError(f"Failed to add effect: {e}")
    
    def remove_effect(self, effect_id: str) -> bool:
        """Remove an effect from the pipeline by ID"""
        for i, effect in enumerate(self.effects):
            if effect.effect_id == effect_id:
                del self.effects[i]
                return True
        return False
    
    def move_effect(self, effect_id: str, new_position: int) -> bool:
        """Move an effect to a new position in the pipeline"""
        effect = self.get_effect(effect_id)
        if not effect:
            return False
        
        # Remove effect from current position
        self.remove_effect(effect_id)
        
        # Insert at new position
        new_position = max(0, min(new_position, len(self.effects)))
        self.effects.insert(new_position, effect)
        return True
    
    def get_effect(self, effect_id: str) -> Optional[BaseEffect]:
        """Get an effect by ID"""
        for effect in self.effects:
            if effect.effect_id == effect_id:
                return effect
        return None
    
    def get_effects_by_category(self, category: EffectCategory) -> List[BaseEffect]:
        """Get all effects of a specific category"""
        return [effect for effect in self.effects if effect.category == category]
    
    def enable_effect(self, effect_id: str):
        """Enable an effect"""
        effect = self.get_effect(effect_id)
        if effect:
            effect.enabled = True
    
    def disable_effect(self, effect_id: str):
        """Disable an effect"""
        effect = self.get_effect(effect_id)
        if effect:
            effect.enabled = False
    
    def clear_pipeline(self):
        """Clear all effects from the pipeline"""
        self.effects.clear()
    
    def get_enabled_effects(self) -> List[BaseEffect]:
        """Get only enabled effects"""
        return [effect for effect in self.effects if effect.enabled]
    
    def duplicate_effect(self, effect_id: str) -> Optional[str]:
        """Duplicate an effect in the pipeline"""
        effect = self.get_effect(effect_id)
        if not effect:
            return None
        
        # Create a new effect of the same type using ProductionEffectFactory
        new_effect = ProductionEffectFactory.create_effect(effect.__class__.__name__.replace('Effect', '').lower())
        
        # Copy parameters
        for param_name, param in effect.parameters.items():
            if param_name in new_effect.parameters:
                new_effect.parameters[param_name].value = param.value
        
        # Copy other properties
        new_effect.name = f"{effect.name} (Copy)"
        new_effect.start_time = effect.start_time
        new_effect.duration = effect.duration
        new_effect.enabled = effect.enabled
        
        # Add to pipeline
        self.effects.append(new_effect)
        return new_effect.effect_id
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline"""
        category_counts = {}
        total_duration = 0.0
        enabled_count = 0
        
        for effect in self.effects:
            # Count by category
            category = effect.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Count enabled effects
            if effect.enabled:
                enabled_count += 1
            
            # Calculate total duration (approximate)
            if effect.duration:
                total_duration = max(total_duration, effect.start_time + effect.duration)
        
        return {
            "total_effects": len(self.effects),
            "enabled_effects": enabled_count,
            "disabled_effects": len(self.effects) - enabled_count,
            "category_counts": category_counts,
            "estimated_duration": total_duration
        }
    
    def validate_pipeline(self) -> List[str]:
        """Validate the pipeline and return list of issues"""
        issues = []
        
        # Check for effects with invalid parameters
        for effect in self.effects:
            for param_name, param in effect.parameters.items():
                if param.min_value is not None and param.value < param.min_value:
                    issues.append(f"{effect.name}: {param_name} is below minimum value")
                
                if param.max_value is not None and param.value > param.max_value:
                    issues.append(f"{effect.name}: {param_name} is above maximum value")
        
        # Check for conflicting effects
        text_effects = self.get_effects_by_category(EffectCategory.TEXT)
        if len(text_effects) > 5:
            issues.append("Too many text effects may cause visual clutter")
        
        # Check timing issues
        overlapping_transitions = []
        for i, effect in enumerate(self.effects):
            if effect.category == EffectCategory.TRANSITIONS:
                for j, other_effect in enumerate(self.effects[i+1:], i+1):
                    if (other_effect.category == EffectCategory.TRANSITIONS and 
                        abs(effect.start_time - other_effect.start_time) < 1.0):
                        overlapping_transitions.append((effect.name, other_effect.name))
        
        if overlapping_transitions:
            issues.append("Overlapping transition effects detected")
        
        return issues
    
    def apply_preset(self, preset_name: str):
        """Apply a predefined effect preset"""
        presets = {
            "action_scene": [
                ("speed_lines", {"direction": "radial", "intensity": 0.9}),
                ("zoom_punch", {"zoom_factor": 1.8}),
                ("camera_shake", {"intensity": 15}),
                ("color_grade", {"style": "dramatic", "intensity": 1.1}),
                ("impact_frame", {"style": "energy"})
            ],
            "power_up": [
                ("energy_aura", {"intensity": 1.5, "pulse_rate": 8}),
                ("bloom", {"threshold": 180, "blur_size": 25}),
                ("animated_text", {"text": "POWER UP!", "animation": "zoom_in"}),
                ("beat_flash", {"intensity": 0.7})
            ],
            "emotional_moment": [
                ("color_grade", {"style": "sunset", "intensity": 0.8}),
                ("bloom", {"threshold": 200, "blur_size": 20}),
                ("animated_text", {"text": "Heartfelt...", "animation": "fade_in"}),
            ],
            "speed_boost": [
                ("speed_lines", {"direction": "right", "intensity": 1.0}),
                ("speed_ramp", {"speed_points": "0,1.0;1,3.0;3,1.0"}),
                ("chromatic_aberration", {"intensity": 8}),
                ("sound_fx_text", {"text": "ZOOM!", "style": "explosive"})
            ],
            "impact_hit": [
                ("zoom_punch", {"zoom_factor": 2.5}),
                ("camera_shake", {"intensity": 25}),
                ("impact_frame", {"style": "manga"}),
                ("sound_fx_text", {"text": "IMPACT!", "style": "impact"})
            ]
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        # Clear current pipeline
        self.clear_pipeline()
        
        # Add preset effects
        for effect_type, params in presets[preset_name]:
            effect_id = self.add_effect(effect_type)
            effect = self.get_effect(effect_id)
            
            # Apply preset parameters
            for param_name, param_value in params.items():
                if param_name in effect.parameters:
                    effect.set_parameter_value(param_name, param_value)
    
    def save_to_file(self, file_path: str):
        """Save pipeline to JSON file"""
        pipeline_data = {
            "name": self.name,
            "description": self.description,
            "effects": [effect.to_dict() for effect in self.effects]
        }
        
        with open(file_path, 'w') as f:
            json.dump(pipeline_data, f, indent=2)
    
    def load_from_file(self, file_path: str):
        """Load pipeline from JSON file"""
        with open(file_path, 'r') as f:
            pipeline_data = json.load(f)
        
        self.name = pipeline_data.get("name", "Loaded Pipeline")
        self.description = pipeline_data.get("description", "")
        
        # Clear current effects
        self.clear_pipeline()
        
        # Load effects
        for effect_data in pipeline_data.get("effects", []):
            # Determine effect type from the data
            effect_type = self._determine_effect_type(effect_data)
            if effect_type:
                effect_id = self.add_effect(effect_type)
                effect = self.get_effect(effect_id)
                
                # Restore properties
                effect.effect_id = effect_data.get("effect_id", effect.effect_id)
                effect.name = effect_data.get("name", effect.name)
                effect.start_time = effect_data.get("start_time", 0.0)
                effect.duration = effect_data.get("duration", None)
                effect.enabled = effect_data.get("enabled", True)
                
                # Restore parameters
                for param_name, param_data in effect_data.get("parameters", {}).items():
                    if param_name in effect.parameters:
                        effect.parameters[param_name].value = param_data.get("value")
    
    def _determine_effect_type(self, effect_data: Dict[str, Any]) -> Optional[str]:
        """Determine effect type from saved data"""
        effect_name = effect_data.get("name", "")
        
        # Map effect names to types
        name_to_type = {
            "Speed Ramp": "speed_ramp",
            "Zoom Punch": "zoom_punch",
            "Camera Shake": "camera_shake",
            "Speed Lines": "speed_lines",
            "Impact Frame": "impact_frame",
            "Energy Aura": "energy_aura",
            "Color Grade": "color_grade",
            "Chromatic Aberration": "chromatic_aberration",
            "Bloom": "bloom",
            "Animated Text": "animated_text",
            "Sound FX Text": "sound_fx_text",
            "Beat Flash": "beat_flash",
            "Beat Zoom": "beat_zoom",
            "Iris Transition": "iris_transition",
            "Swipe Transition": "swipe_transition"
        }
        
        return name_to_type.get(effect_name)
    
    def export_for_processing(self) -> Dict[str, Any]:
        """Export pipeline data for video processing"""
        return {
            "effects": [
                {
                    "type": effect.__class__.__name__.replace('Effect', '').lower(),
                    "name": effect.name,
                    "category": effect.category.value,
                    "start_time": effect.start_time,
                    "duration": effect.duration,
                    "enabled": effect.enabled,
                    "parameters": {
                        name: param.value for name, param in effect.parameters.items()
                    }
                }
                for effect in self.effects if effect.enabled
            ]
        }
    
    def __len__(self) -> int:
        """Return number of effects in pipeline"""
        return len(self.effects)
    
    def __iter__(self):
        """Make pipeline iterable"""
        return iter(self.effects)
    
    def __getitem__(self, index: int) -> BaseEffect:
        """Get effect by index"""
        return self.effects[index]