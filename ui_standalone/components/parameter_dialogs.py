"""
Parameter Configuration Dialogs - Advanced effect parameter editing with real-time preview
"""

import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from typing import Dict, Any, Callable, Optional, List, Tuple
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ui_standalone.models.effect_models import BaseEffect, EffectParameter, EffectCategory
from ui_standalone.models.effect_adapter import ProductionEffect

class BaseParameterDialog:
    """Base class for effect parameter configuration dialogs"""
    
    def __init__(self, parent, effect: ProductionEffect, on_preview_callback: Optional[Callable] = None, 
                 on_apply_callback: Optional[Callable] = None, video_duration: Optional[float] = None):
        self.parent = parent
        self.effect = effect
        self.on_preview_callback = on_preview_callback
        self.on_apply_callback = on_apply_callback
        self.video_duration = video_duration or 60.0  # Default to 60 seconds if not provided
        
        # Store original parameters for reset
        self.original_params = {name: param.value for name, param in effect.parameters.items()}
        
        # UI elements
        self.dialog = None
        self.parameter_widgets = {}
        self.preview_enabled = False
        
        # Dialog result
        self.result = None
        
    def show_dialog(self) -> bool:
        """Show the parameter dialog and return True if applied, False if cancelled"""
        self.create_dialog()
        
        # Center the dialog
        self.center_dialog()
        
        # Make dialog modal
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Wait for dialog to close
        self.parent.wait_window(self.dialog)
        
        return self.result == "apply"
    
    def create_dialog(self):
        """Create the parameter dialog window"""
        self.dialog = ctk.CTkToplevel(self.parent)
        self.dialog.title(f"Configure {self.effect.name}")
        self.dialog.geometry("600x700")
        self.dialog.resizable(True, True)
        
        # Main container
        main_frame = ctk.CTkFrame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        self.create_header(main_frame)
        
        # Parameters section
        params_frame = ctk.CTkScrollableFrame(main_frame)
        params_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.create_parameter_controls(params_frame)
        
        # Preview controls
        preview_frame = ctk.CTkFrame(main_frame)
        preview_frame.pack(fill="x", padx=5, pady=5)
        
        self.create_preview_controls(preview_frame)
        
        # Action buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        self.create_action_buttons(button_frame)
        
        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
    
    def create_header(self, parent):
        """Create dialog header with effect info"""
        header_frame = ctk.CTkFrame(parent)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        # Effect name and category
        title_label = ctk.CTkLabel(
            header_frame,
            text=f"🎬 {self.effect.name}",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(side="left", padx=10, pady=10)
        
        category_label = ctk.CTkLabel(
            header_frame,
            text=self.effect.category.value,
            font=ctk.CTkFont(size=12),
            fg_color=self.get_category_color(),
            corner_radius=15,
            width=120
        )
        category_label.pack(side="right", padx=10, pady=10)
        
        # Description
        desc_label = ctk.CTkLabel(
            header_frame,
            text=f"Configure parameters for {self.effect.name.lower()} effect",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        desc_label.pack(side="left", padx=10)
    
    def get_category_color(self):
        """Get color for effect category"""
        colors = {
            "Motion Effects": "blue",
            "Anime Effects": "purple", 
            "Color Effects": "orange",
            "Text Effects": "green",
            "Audio Sync": "red",
            "Transitions": "cyan"
        }
        return colors.get(self.effect.category.value, "gray")
    
    def create_parameter_controls(self, parent):
        """Create controls for all effect parameters"""
        # Group parameters by type
        timing_params = []
        effect_params = []
        
        for name, param in self.effect.parameters.items():
            if name in ['start_time', 'duration']:
                timing_params.append((name, param))
            else:
                effect_params.append((name, param))
        
        # Timing parameters section
        if timing_params:
            timing_section = ctk.CTkFrame(parent)
            timing_section.pack(fill="x", padx=5, pady=5)
            
            timing_label = ctk.CTkLabel(
                timing_section,
                text="⏱️ Timing Parameters",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            timing_label.pack(anchor="w", padx=10, pady=5)
            
            for name, param in timing_params:
                self.create_parameter_widget(timing_section, name, param)
        
        # Effect parameters section
        if effect_params:
            effects_section = ctk.CTkFrame(parent)
            effects_section.pack(fill="x", padx=5, pady=5)
            
            effects_label = ctk.CTkLabel(
                effects_section,
                text="🎨 Effect Parameters",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            effects_label.pack(anchor="w", padx=10, pady=5)
            
            for name, param in effect_params:
                self.create_parameter_widget(effects_section, name, param)
        
        # Presets section (if available)
        if hasattr(self.effect, 'get_presets') or self.has_presets():
            self.create_presets_section(parent)
    
    def create_parameter_widget(self, parent, param_name: str, param: EffectParameter):
        """Create appropriate widget for parameter type"""
        param_frame = ctk.CTkFrame(parent)
        param_frame.pack(fill="x", padx=10, pady=5)
        
        # Parameter label
        label_frame = ctk.CTkFrame(param_frame)
        label_frame.pack(fill="x", padx=5, pady=2)
        
        param_label = ctk.CTkLabel(
            label_frame,
            text=param.name,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        param_label.pack(side="left", padx=5)
        
        # Parameter description
        if param.description:
            desc_label = ctk.CTkLabel(
                label_frame,
                text=param.description,
                font=ctk.CTkFont(size=10),
                text_color="gray",
                anchor="w"
            )
            desc_label.pack(side="right", padx=5)
        
        # Create widget based on parameter type
        widget_frame = ctk.CTkFrame(param_frame)
        widget_frame.pack(fill="x", padx=5, pady=2)
        
        if param.param_type in ['float', 'int']:
            widget = self.create_numeric_widget(widget_frame, param_name, param)
        elif param.param_type == 'bool':
            widget = self.create_boolean_widget(widget_frame, param_name, param)
        elif param.param_type == 'string':
            widget = self.create_string_widget(widget_frame, param_name, param)
        else:
            widget = self.create_string_widget(widget_frame, param_name, param)
        
        self.parameter_widgets[param_name] = widget
    
    def create_numeric_widget(self, parent, param_name: str, param: EffectParameter):
        """Create slider + entry for numeric parameters"""
        container = ctk.CTkFrame(parent)
        container.pack(fill="x", padx=5, pady=2)
        
        # Current value display
        value_var = tk.DoubleVar(value=float(param.value))
        
        value_label = ctk.CTkLabel(
            container,
            text=f"{param.value}",
            font=ctk.CTkFont(size=11, weight="bold"),
            width=80
        )
        value_label.pack(side="right", padx=5)
        
        # Calculate bounds - special handling for timing parameters
        min_val = getattr(param, 'min_value', 0.0)
        max_val = getattr(param, 'max_value', 10.0 if param.param_type == 'float' else 100.0)
        
        # For start_time parameter, constrain by video duration
        if param_name == 'start_time':
            max_val = max(0.0, self.video_duration - 0.1)  # Leave at least 0.1s for duration
            # Update the parameter's max_value for validation
            param.max_value = max_val
        
        # For duration parameter, adjust max based on start_time and video_duration
        elif param_name == 'duration':
            start_time = self.get_current_start_time()
            max_val = max(0.1, self.video_duration - start_time)
            # Update the parameter's max_value for validation
            param.max_value = max_val
        
        step = getattr(param, 'step', 0.1 if param.param_type == 'float' else 1.0)
        
        def on_slider_change(value):
            if param.param_type == 'int':
                value = int(value)
            value_var.set(value)
            value_label.configure(text=f"{value}")
            self.on_parameter_changed(param_name, value)
        
        slider = ctk.CTkSlider(
            container,
            from_=min_val,
            to=max_val,
            number_of_steps=max(1, int((max_val - min_val) / step)),
            command=on_slider_change
        )
        slider.set(min(float(param.value), max_val))  # Ensure value doesn't exceed new max
        slider.pack(side="left", fill="x", expand=True, padx=5)
        
        return {
            'type': 'numeric',
            'slider': slider,
            'value_var': value_var,
            'value_label': value_label,
            'param': param,
            'min_val': min_val,
            'max_val': max_val,
            'step': step
        }
    
    def create_boolean_widget(self, parent, param_name: str, param: EffectParameter):
        """Create checkbox for boolean parameters"""
        value_var = tk.BooleanVar(value=bool(param.value))
        
        def on_change():
            self.on_parameter_changed(param_name, value_var.get())
        
        checkbox = ctk.CTkCheckBox(
            parent,
            text="Enabled",
            variable=value_var,
            command=on_change
        )
        checkbox.pack(side="left", padx=5, pady=2)
        
        return {
            'type': 'boolean',
            'checkbox': checkbox,
            'value_var': value_var
        }
    
    def create_string_widget(self, parent, param_name: str, param: EffectParameter):
        """Create entry or dropdown for string parameters"""
        # Check if parameter has predefined choices
        choices = getattr(param, 'choices', None)
        
        if choices and isinstance(choices, list):
            # Dropdown for predefined choices
            value_var = tk.StringVar(value=str(param.value))
            
            def on_change(choice):
                self.on_parameter_changed(param_name, choice)
            
            dropdown = ctk.CTkOptionMenu(
                parent,
                values=choices,
                variable=value_var,
                command=on_change
            )
            dropdown.pack(side="left", fill="x", expand=True, padx=5, pady=2)
            
            return {
                'type': 'dropdown',
                'dropdown': dropdown,
                'value_var': value_var
            }
        else:
            # Text entry for free form text
            value_var = tk.StringVar(value=str(param.value))
            
            def on_change(*args):
                self.on_parameter_changed(param_name, value_var.get())
            
            value_var.trace("w", on_change)
            
            entry = ctk.CTkEntry(
                parent,
                textvariable=value_var
            )
            entry.pack(side="left", fill="x", expand=True, padx=5, pady=2)
            
            return {
                'type': 'entry',
                'entry': entry,
                'value_var': value_var
            }
    
    def create_presets_section(self, parent):
        """Create presets section for quick parameter configurations"""
        presets_frame = ctk.CTkFrame(parent)
        presets_frame.pack(fill="x", padx=5, pady=5)
        
        presets_label = ctk.CTkLabel(
            presets_frame,
            text="🎯 Quick Presets",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        presets_label.pack(anchor="w", padx=10, pady=5)
        
        # Get presets based on effect category
        presets = self.get_effect_presets()
        
        if presets:
            preset_buttons_frame = ctk.CTkFrame(presets_frame)
            preset_buttons_frame.pack(fill="x", padx=10, pady=5)
            
            for preset_name, preset_config in presets.items():
                preset_button = ctk.CTkButton(
                    preset_buttons_frame,
                    text=preset_name.replace('_', ' ').title(),
                    command=lambda p=preset_config: self.apply_preset(p),
                    width=120,
                    height=30
                )
                preset_button.pack(side="left", padx=5, pady=2)
    
    def get_effect_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get presets for the current effect type"""
        # This would be extended to include specific presets for each effect
        category = self.effect.category.value.lower().replace(' ', '_')
        
        common_presets = {
            'mild': {'intensity': 0.3, 'duration': 1.0},
            'moderate': {'intensity': 0.7, 'duration': 1.5},
            'intense': {'intensity': 1.0, 'duration': 2.0}
        }
        
        # Category-specific presets
        category_presets = {
            'motion_effects': {
                'subtle_motion': {'zoom_factor': 1.2, 'shake_intensity': 5},
                'dynamic_action': {'zoom_factor': 1.8, 'shake_intensity': 15},
                'extreme_impact': {'zoom_factor': 2.5, 'shake_intensity': 25}
            },
            'anime_effects': {
                'light_energy': {'intensity': 0.5, 'pulse_rate': 3.0},
                'power_surge': {'intensity': 0.8, 'pulse_rate': 8.0},
                'ultimate_power': {'intensity': 1.0, 'pulse_rate': 12.0}
            },
            'color_effects': {
                'natural_look': {'intensity': 0.3},
                'vivid_colors': {'intensity': 0.7}, 
                'hyper_saturated': {'intensity': 1.0}
            }
        }
        
        return category_presets.get(category, common_presets)
    
    def has_presets(self) -> bool:
        """Check if presets are available for this effect"""
        return len(self.get_effect_presets()) > 0
    
    def apply_preset(self, preset_config: Dict[str, Any]):
        """Apply a preset configuration to parameters"""
        for param_name, value in preset_config.items():
            if param_name in self.parameter_widgets:
                widget = self.parameter_widgets[param_name]
                
                if widget['type'] == 'numeric':
                    widget['slider'].set(float(value))
                    widget['value_var'].set(value)
                    widget['value_label'].configure(text=f"{value}")
                elif widget['type'] == 'boolean':
                    widget['value_var'].set(bool(value))
                elif widget['type'] in ['entry', 'dropdown']:
                    widget['value_var'].set(str(value))
                
                # Update the effect parameter
                self.on_parameter_changed(param_name, value)
    
    def create_preview_controls(self, parent):
        """Create preview controls"""
        # Manual preview button
        preview_button = ctk.CTkButton(
            parent,
            text="🔍 Preview",
            command=self.manual_preview,
            width=100
        )
        preview_button.pack(side="left", padx=10, pady=5)
        
        # Reset button
        reset_button = ctk.CTkButton(
            parent,
            text="🔄 Reset",
            command=self.reset_parameters,
            width=100
        )
        reset_button.pack(side="right", padx=10, pady=5)
    
    def create_action_buttons(self, parent):
        """Create action buttons (Apply, Cancel)"""
        # Cancel button
        cancel_button = ctk.CTkButton(
            parent,
            text="❌ Cancel",
            command=self.on_cancel,
            width=100,
            fg_color="red",
            hover_color="darkred"
        )
        cancel_button.pack(side="left", padx=10, pady=10)
        
        # Apply button
        apply_button = ctk.CTkButton(
            parent,
            text="✅ Apply",
            command=self.on_apply,
            width=100,
            fg_color="green",
            hover_color="darkgreen"
        )
        apply_button.pack(side="right", padx=10, pady=10)
    
    def center_dialog(self):
        """Center the dialog on screen"""
        self.dialog.update_idletasks()
        
        # Get screen dimensions
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        
        # Get dialog dimensions
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Calculate position
        x = (screen_width - dialog_width) // 2
        y = (screen_height - dialog_height) // 2
        
        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
    
    def get_current_start_time(self) -> float:
        """Get current start time value"""
        if 'start_time' in self.effect.parameters:
            return float(self.effect.parameters['start_time'].value)
        return 0.0
    
    def get_current_duration(self) -> float:
        """Get current duration value"""
        if 'duration' in self.effect.parameters:
            return float(self.effect.parameters['duration'].value)
        return 1.0
    
    def update_duration_bounds(self):
        """Update duration parameter bounds based on current start time"""
        if 'duration' not in self.parameter_widgets:
            return
            
        duration_widget = self.parameter_widgets['duration']
        if duration_widget['type'] != 'numeric':
            return
            
        # Calculate new max duration
        start_time = self.get_current_start_time()
        new_max_duration = max(0.1, self.video_duration - start_time)
        
        # Update slider bounds
        slider = duration_widget['slider']
        current_value = duration_widget['value_var'].get()
        
        # Adjust current value if it exceeds new max
        if current_value > new_max_duration:
            current_value = new_max_duration
            duration_widget['value_var'].set(current_value)
            duration_widget['value_label'].configure(text=f"{current_value:.1f}")
            # Update the effect parameter
            self.effect.parameters['duration'].value = current_value
        
        # Update slider configuration
        min_val = duration_widget['min_val']
        step = duration_widget['step']
        
        slider.configure(
            from_=min_val,
            to=new_max_duration,
            number_of_steps=max(1, int((new_max_duration - min_val) / step))
        )
        
        # Update parameter max_value for validation
        if 'duration' in self.effect.parameters:
            self.effect.parameters['duration'].max_value = new_max_duration
        
        # Store updated bounds in widget
        duration_widget['max_val'] = new_max_duration
    
    def update_start_time_bounds(self):
        """Update start_time parameter bounds based on current duration"""
        if 'start_time' not in self.parameter_widgets:
            return
            
        start_time_widget = self.parameter_widgets['start_time']
        if start_time_widget['type'] != 'numeric':
            return
            
        # Calculate new max start_time (leave room for minimum duration)
        current_duration = self.get_current_duration()
        new_max_start_time = max(0.0, self.video_duration - current_duration)
        
        # Update slider bounds
        slider = start_time_widget['slider']
        current_value = start_time_widget['value_var'].get()
        
        # Adjust current value if it exceeds new max
        if current_value > new_max_start_time:
            current_value = new_max_start_time
            start_time_widget['value_var'].set(current_value)
            start_time_widget['value_label'].configure(text=f"{current_value:.1f}")
            # Update the effect parameter
            self.effect.parameters['start_time'].value = current_value
        
        # Update slider configuration
        min_val = start_time_widget['min_val']
        step = start_time_widget['step']
        
        slider.configure(
            from_=min_val,
            to=new_max_start_time,
            number_of_steps=max(1, int((new_max_start_time - min_val) / step))
        )
        
        # Update parameter max_value for validation
        if 'start_time' in self.effect.parameters:
            self.effect.parameters['start_time'].max_value = new_max_start_time
        
        # Store updated bounds in widget
        start_time_widget['max_val'] = new_max_start_time
    
    def on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter value change"""
        # Update the effect parameter
        if param_name in self.effect.parameters:
            self.effect.parameters[param_name].value = value
        
        # Update timing bounds when timing parameters change
        if param_name == 'start_time':
            self.update_duration_bounds()
        elif param_name == 'duration':
            self.update_start_time_bounds()
        
        # Note: No automatic preview - only when user clicks Preview or Apply
    
    
    def manual_preview(self):
        """Manually trigger preview"""
        if self.on_preview_callback:
            try:
                self.on_preview_callback(self.effect)
            except Exception as e:
                messagebox.showerror("Preview Error", f"Failed to generate preview: {e}")
    
    def reset_parameters(self):
        """Reset all parameters to original values"""
        for param_name, original_value in self.original_params.items():
            if param_name in self.parameter_widgets:
                widget = self.parameter_widgets[param_name]
                
                if widget['type'] == 'numeric':
                    widget['slider'].set(float(original_value))
                    widget['value_var'].set(original_value)
                    widget['value_label'].configure(text=f"{original_value}")
                elif widget['type'] == 'boolean':
                    widget['value_var'].set(bool(original_value))
                elif widget['type'] in ['entry', 'dropdown']:
                    widget['value_var'].set(str(original_value))
                
                # Update the effect parameter
                self.effect.parameters[param_name].value = original_value
        
        # Note: No automatic preview - user must click Preview button manually
    
    def on_apply(self):
        """Handle apply button click"""
        try:
            # Validate parameters
            if self.validate_parameters():
                if self.on_apply_callback:
                    self.on_apply_callback(self.effect)
                
                self.result = "apply"
                self.dialog.destroy()
            else:
                messagebox.showerror("Invalid Parameters", "Please check your parameter values.")
        except Exception as e:
            messagebox.showerror("Apply Error", f"Failed to apply changes: {e}")
    
    def on_cancel(self):
        """Handle cancel button click"""
        # Restore original parameters
        for param_name, original_value in self.original_params.items():
            if param_name in self.effect.parameters:
                self.effect.parameters[param_name].value = original_value
        
        self.result = "cancel"
        self.dialog.destroy()
    
    def validate_parameters(self) -> bool:
        """Validate all parameter values"""
        for param_name, param in self.effect.parameters.items():
            try:
                # Type validation
                if param.param_type == 'int':
                    int(param.value)
                elif param.param_type == 'float':
                    float(param.value)
                elif param.param_type == 'bool':
                    bool(param.value)
                
                # Range validation
                if hasattr(param, 'min_value') and param.min_value is not None:
                    if float(param.value) < param.min_value:
                        return False
                        
                if hasattr(param, 'max_value') and param.max_value is not None:
                    if float(param.value) > param.max_value:
                        return False
                        
            except (ValueError, TypeError):
                return False
        
        return True


class MotionEffectsDialog(BaseParameterDialog):
    """Specialized dialog for Motion Effects with advanced controls"""
    
    def __init__(self, parent, effect: ProductionEffect, on_preview_callback: Optional[Callable] = None, 
                 on_apply_callback: Optional[Callable] = None, video_duration: Optional[float] = None):
        super().__init__(parent, effect, on_preview_callback, on_apply_callback, video_duration)
    
    def get_effect_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get Motion Effects specific presets"""
        effect_name = self.effect.name.lower()
        
        if 'speed' in effect_name:
            return {
                'smooth_slowmo': {'speed_points': '0,1.0;1,0.3;2,1.0'},
                'dramatic_pause': {'speed_points': '0,1.0;0.5,0.1;1.5,0.1;2,1.0'},
                'action_ramp': {'speed_points': '0,0.5;1,2.0;2,1.0'}
            }
        elif 'zoom' in effect_name:
            return {
                'subtle_punch': {'zoom_factor': 1.2},
                'impact_zoom': {'zoom_factor': 1.8},
                'extreme_zoom': {'zoom_factor': 2.5}
            }
        elif 'shake' in effect_name:
            return {
                'light_tremor': {'shake_intensity': 3},
                'earthquake': {'shake_intensity': 15},
                'explosion': {'shake_intensity': 25}
            }
        elif 'blur' in effect_name:
            return {
                'gentle_motion': {'blur_strength': 2.0, 'motion_angle': 0.0},
                'speed_lines': {'blur_strength': 8.0, 'motion_angle': 45.0},
                'vertical_rush': {'blur_strength': 12.0, 'motion_angle': 90.0}
            }
        else:
            return {
                'mild': {'duration': 1.0},
                'moderate': {'duration': 2.0},
                'extended': {'duration': 3.0}
            }


class AnimeEffectsDialog(BaseParameterDialog):
    """Specialized dialog for Anime Effects with style-specific controls"""
    
    def __init__(self, parent, effect: ProductionEffect, on_preview_callback: Optional[Callable] = None, 
                 on_apply_callback: Optional[Callable] = None, video_duration: Optional[float] = None):
        super().__init__(parent, effect, on_preview_callback, on_apply_callback, video_duration)
    
    def get_effect_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get Anime Effects specific presets"""
        effect_name = self.effect.name.lower()
        
        if 'speed_lines' in effect_name:
            return {
                'subtle_motion': {'direction': 'right', 'intensity': 0.4},
                'power_dash': {'direction': 'right', 'intensity': 0.8},
                'ultimate_speed': {'direction': 'radial', 'intensity': 1.0}
            }
        elif 'impact' in effect_name:
            return {
                'manga_style': {'style': 'manga'},
                'energy_burst': {'style': 'energy'},
                'flash_impact': {'style': 'flash'}
            }
        elif 'aura' in effect_name:
            return {
                'calm_energy': {'intensity': 0.3, 'pulse_rate': 2.0},
                'power_surge': {'intensity': 0.8, 'pulse_rate': 8.0},
                'max_power': {'intensity': 1.0, 'pulse_rate': 15.0}
            }
        elif 'glow' in effect_name:
            return {
                'soft_glow': {'color': '255,255,255', 'intensity': 0.5},
                'golden_aura': {'color': '255,215,0', 'intensity': 0.8},
                'power_glow': {'color': '0,255,255', 'intensity': 1.0}
            }
        else:
            return {
                'subtle': {'intensity': 0.3},
                'moderate': {'intensity': 0.7},
                'intense': {'intensity': 1.0}
            }


class ColorEffectsDialog(BaseParameterDialog):
    """Specialized dialog for Color Effects with color-specific controls"""
    
    def __init__(self, parent, effect: ProductionEffect, on_preview_callback: Optional[Callable] = None, 
                 on_apply_callback: Optional[Callable] = None, video_duration: Optional[float] = None):
        super().__init__(parent, effect, on_preview_callback, on_apply_callback, video_duration)
    
    def get_effect_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get Color Effects specific presets"""
        effect_name = self.effect.name.lower()
        
        if 'grading' in effect_name:
            return {
                'natural_enhance': {'style': 'vibrant', 'intensity': 0.3},
                'cinematic_look': {'style': 'cinematic', 'intensity': 0.7},
                'hyper_stylized': {'style': 'vibrant', 'intensity': 1.0}
            }
        elif 'chromatic' in effect_name:
            return {
                'subtle_shift': {'intensity': 2},
                'noticeable_split': {'intensity': 8},
                'extreme_aberration': {'intensity': 15}
            }
        elif 'bloom' in effect_name:
            return {
                'soft_glow': {'threshold': 220, 'blur_size': 8},
                'bright_bloom': {'threshold': 180, 'blur_size': 20},
                'intense_bloom': {'threshold': 150, 'blur_size': 30}
            }
        else:
            return {
                'mild': {'intensity': 0.3},
                'moderate': {'intensity': 0.7},
                'strong': {'intensity': 1.0}
            }


class TextEffectsDialog(BaseParameterDialog):
    """Specialized dialog for Text Effects with text-specific controls"""
    
    def __init__(self, parent, effect: ProductionEffect, on_preview_callback: Optional[Callable] = None, 
                 on_apply_callback: Optional[Callable] = None, video_duration: Optional[float] = None):
        super().__init__(parent, effect, on_preview_callback, on_apply_callback, video_duration)
    
    def create_parameter_controls(self, parent):
        """Override to add text preview"""
        super().create_parameter_controls(parent)
        
        # Add text preview section
        preview_frame = ctk.CTkFrame(parent)
        preview_frame.pack(fill="x", padx=5, pady=5)
        
        preview_label = ctk.CTkLabel(
            preview_frame,
            text="📝 Text Preview",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        preview_label.pack(anchor="w", padx=10, pady=5)
        
        # Text preview area
        self.text_preview = ctk.CTkLabel(
            preview_frame,
            text=self.get_preview_text(),
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="black",
            text_color="white",
            corner_radius=10,
            height=60
        )
        self.text_preview.pack(padx=10, pady=5, fill="x")
    
    def get_preview_text(self) -> str:
        """Get text for preview"""
        if 'text' in self.effect.parameters:
            return str(self.effect.parameters['text'].value)
        elif 'character_name' in self.effect.parameters:
            return str(self.effect.parameters['character_name'].value)
        elif 'technique_name' in self.effect.parameters:
            return str(self.effect.parameters['technique_name'].value)
        else:
            return "SAMPLE TEXT"
    
    def on_parameter_changed(self, param_name: str, value: Any):
        """Override to update text preview"""
        super().on_parameter_changed(param_name, value)
        
        # Update text preview if text parameter changed (immediate visual feedback)
        if param_name in ['text', 'character_name', 'technique_name'] and hasattr(self, 'text_preview'):
            self.text_preview.configure(text=str(value))
    
    def get_effect_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get Text Effects specific presets"""
        effect_name = self.effect.name.lower()
        
        if 'animated' in effect_name:
            return {
                'slide_entrance': {'animation': 'slide_in', 'duration': 1.5},
                'fade_dramatic': {'animation': 'fade_in', 'duration': 2.0},
                'zoom_impact': {'animation': 'zoom_in', 'duration': 1.0}
            }
        elif 'sound' in effect_name:
            return {
                'impact_text': {'style': 'impact', 'text': 'BOOM!'},
                'explosive_text': {'style': 'explosive', 'text': 'CRASH!'},
                'electric_text': {'style': 'electric', 'text': 'ZAP!'}
            }
        else:
            return {
                'quick_display': {'duration': 1.0},
                'normal_timing': {'duration': 2.0},
                'long_display': {'duration': 3.0}
            }


class AudioSyncDialog(BaseParameterDialog):
    """Specialized dialog for Audio Sync Effects with audio-specific controls"""
    
    def __init__(self, parent, effect: ProductionEffect, on_preview_callback: Optional[Callable] = None, 
                 on_apply_callback: Optional[Callable] = None, video_duration: Optional[float] = None):
        super().__init__(parent, effect, on_preview_callback, on_apply_callback, video_duration)
    
    def get_effect_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get Audio Sync specific presets"""
        effect_name = self.effect.name.lower()
        
        if 'flash' in effect_name:
            return {
                'subtle_flash': {'intensity': 0.3, 'duration': 0.05},
                'bright_flash': {'intensity': 0.7, 'duration': 0.1},
                'strobe_flash': {'intensity': 1.0, 'duration': 0.15}
            }
        elif 'zoom' in effect_name:
            return {
                'gentle_pulse': {'zoom_factor': 1.1, 'duration': 0.2},
                'beat_zoom': {'zoom_factor': 1.3, 'duration': 0.3},
                'heavy_zoom': {'zoom_factor': 1.5, 'duration': 0.4}
            }
        elif 'color' in effect_name:
            return {
                'warm_pulse': {'color_shift': '1.2,1.0,0.8', 'duration': 0.15},
                'cool_pulse': {'color_shift': '0.8,1.0,1.2', 'duration': 0.15},
                'rainbow_pulse': {'color_shift': '1.2,1.2,1.2', 'duration': 0.15}
            }
        else:
            return {
                'quick_sync': {'duration': 0.1},
                'medium_sync': {'duration': 0.2},
                'long_sync': {'duration': 0.3}
            }


class TransitionsDialog(BaseParameterDialog):
    """Specialized dialog for Transition Effects with transition-specific controls"""
    
    def __init__(self, parent, effect: ProductionEffect, on_preview_callback: Optional[Callable] = None, 
                 on_apply_callback: Optional[Callable] = None, video_duration: Optional[float] = None):
        super().__init__(parent, effect, on_preview_callback, on_apply_callback, video_duration)
    
    def get_effect_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get Transitions specific presets"""
        effect_name = self.effect.name.lower()
        
        if 'iris' in effect_name:
            return {
                'quick_reveal': {'duration': 0.5},
                'dramatic_reveal': {'duration': 1.5},
                'slow_reveal': {'duration': 2.5}
            }
        elif 'swipe' in effect_name:
            return {
                'left_swipe': {'direction': 'left', 'duration': 0.8},
                'right_swipe': {'direction': 'right', 'duration': 0.8},
                'up_swipe': {'direction': 'up', 'duration': 1.0}
            }
        elif 'zoom' in effect_name:
            return {
                'zoom_in_fast': {'zoom_in': True, 'duration': 0.5},
                'zoom_out_smooth': {'zoom_in': False, 'duration': 1.5},
                'zoom_dramatic': {'zoom_in': True, 'duration': 2.0}
            }
        else:
            return {
                'quick_transition': {'duration': 0.5},
                'smooth_transition': {'duration': 1.0},
                'slow_transition': {'duration': 2.0}
            }


def show_parameter_dialog(parent, effect: ProductionEffect, 
                         on_preview_callback: Optional[Callable] = None,
                         on_apply_callback: Optional[Callable] = None,
                         video_duration: Optional[float] = None) -> bool:
    """
    Show appropriate parameter configuration dialog for an effect.
    
    Args:
        parent: Parent widget
        effect: ProductionEffect to configure
        on_preview_callback: Callback for preview updates
        on_apply_callback: Callback when parameters are applied
        video_duration: Duration of the video in seconds (for timing constraints)
        
    Returns:
        True if parameters were applied, False if cancelled
    """
    # Choose appropriate dialog based on effect category
    category = effect.category.value.lower().replace(' ', '_')
    
    dialog_classes = {
        'motion_effects': MotionEffectsDialog,
        'anime_effects': AnimeEffectsDialog,
        'color_effects': ColorEffectsDialog,
        'text_effects': TextEffectsDialog,
        'audio_sync': AudioSyncDialog,
        'transitions': TransitionsDialog
    }
    
    dialog_class = dialog_classes.get(category, BaseParameterDialog)
    dialog = dialog_class(parent, effect, on_preview_callback, on_apply_callback, video_duration)
    return dialog.show_dialog()