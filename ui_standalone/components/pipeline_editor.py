"""
Pipeline Editor Panel - Center panel for managing effects pipeline with drag-drop reordering
"""

import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from typing import Callable, Optional, List
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ui_standalone.models.effect_pipeline import EffectPipeline
from ui_standalone.models.effect_models import BaseEffect, EffectFactory
from ui_standalone.models.effect_adapter import ProductionEffectFactory, ProductionEffect

class EffectItem:
    """Represents a single effect in the pipeline editor"""
    
    def __init__(self, parent, effect: BaseEffect, on_configure, on_delete, on_move_up, on_move_down, on_toggle):
        self.parent = parent
        self.effect = effect
        self.on_configure = on_configure
        self.on_delete = on_delete
        self.on_move_up = on_move_up
        self.on_move_down = on_move_down
        self.on_toggle = on_toggle
        
        self.frame = None
        self.selected = False
        
        self.create_item()
    
    def create_item(self):
        """Create the effect item UI"""
        # Main frame
        self.frame = ctk.CTkFrame(self.parent)
        
        # Configure frame colors based on effect state
        self.update_colors()
        
        # Left side - Effect info
        info_frame = ctk.CTkFrame(self.frame)
        info_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Effect title and category
        title_frame = ctk.CTkFrame(info_frame)
        title_frame.pack(fill="x", padx=5, pady=2)
        
        # Effect name
        self.name_label = ctk.CTkLabel(
            title_frame,
            text=f"📌 {self.effect.name}",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        self.name_label.pack(side="left", padx=5)
        
        # Category badge
        category_label = ctk.CTkLabel(
            title_frame,
            text=self.effect.category.value,
            font=ctk.CTkFont(size=10),
            fg_color=self.get_category_color(),
            corner_radius=10,
            width=80
        )
        category_label.pack(side="right", padx=5)
        
        # Timing info
        timing_frame = ctk.CTkFrame(info_frame)
        timing_frame.pack(fill="x", padx=5, pady=2)
        
        timing_text = f"Start: {self.effect.start_time:.1f}s"
        if self.effect.duration:
            timing_text += f", Duration: {self.effect.duration:.1f}s"
        else:
            timing_text += ", Duration: Full clip"
        
        self.timing_label = ctk.CTkLabel(
            timing_frame,
            text=timing_text,
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.timing_label.pack(side="left", padx=5)
        
        # Parameters preview
        if self.effect.parameters:
            params_frame = ctk.CTkFrame(info_frame)
            params_frame.pack(fill="x", padx=5, pady=2)
            
            # Show first few parameters
            param_texts = []
            for name, param in list(self.effect.parameters.items())[:3]:
                param_texts.append(f"{name}: {param.value}")
            
            if len(self.effect.parameters) > 3:
                param_texts.append("...")
            
            params_text = " | ".join(param_texts)
            self.params_label = ctk.CTkLabel(
                params_frame,
                text=params_text,
                font=ctk.CTkFont(size=9),
                text_color="lightgray"
            )
            self.params_label.pack(side="left", padx=5)
        
        # Right side - Control buttons
        controls_frame = ctk.CTkFrame(self.frame)
        controls_frame.pack(side="right", padx=5, pady=5)
        
        # Enable/Disable toggle
        self.toggle_button = ctk.CTkButton(
            controls_frame,
            text="✓" if self.effect.enabled else "✗",
            width=30,
            height=30,
            command=self.toggle_enabled,
            fg_color="green" if self.effect.enabled else "red"
        )
        self.toggle_button.pack(side="top", padx=2, pady=1)
        
        # Configure button
        config_button = ctk.CTkButton(
            controls_frame,
            text="⚙️",
            width=30,
            height=30,
            command=self.configure_effect
        )
        config_button.pack(side="top", padx=2, pady=1)
        
        # Move buttons
        move_frame = ctk.CTkFrame(controls_frame)
        move_frame.pack(side="top", padx=2, pady=1)
        
        self.move_up_button = ctk.CTkButton(
            move_frame,
            text="⬆️",
            width=25,
            height=25,
            command=self.move_up
        )
        self.move_up_button.pack(side="top", pady=1)
        
        self.move_down_button = ctk.CTkButton(
            move_frame,
            text="⬇️",
            width=25,
            height=25,
            command=self.move_down
        )
        self.move_down_button.pack(side="top", pady=1)
        
        # Delete button
        delete_button = ctk.CTkButton(
            controls_frame,
            text="❌",
            width=30,
            height=30,
            command=self.delete_effect,
            fg_color="red",
            hover_color="darkred"
        )
        delete_button.pack(side="top", padx=2, pady=1)
        
        # Bind click events for selection
        self.frame.bind("<Button-1>", self.on_click)
        info_frame.bind("<Button-1>", self.on_click)
        
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
    
    def update_colors(self):
        """Update frame colors based on effect state"""
        if not self.effect.enabled:
            fg_color = ("gray90", "gray20")
        elif self.selected:
            fg_color = ("lightblue", "darkblue")
        else:
            fg_color = ("white", "gray30")
        
        if self.frame:
            self.frame.configure(fg_color=fg_color)
    
    def toggle_enabled(self):
        """Toggle effect enabled state"""
        self.effect.enabled = not self.effect.enabled
        self.toggle_button.configure(
            text="✓" if self.effect.enabled else "✗",
            fg_color="green" if self.effect.enabled else "red"
        )
        self.update_colors()
        self.on_toggle(self.effect.effect_id)
    
    def configure_effect(self):
        """Open effect configuration dialog"""
        self.on_configure(self.effect.effect_id)
    
    def move_up(self):
        """Move effect up in pipeline"""
        self.on_move_up(self.effect.effect_id)
    
    def move_down(self):
        """Move effect down in pipeline"""
        self.on_move_down(self.effect.effect_id)
    
    def delete_effect(self):
        """Delete effect from pipeline"""
        if messagebox.askyesno("Confirm Delete", f"Delete effect '{self.effect.name}'?"):
            self.on_delete(self.effect.effect_id)
    
    def on_click(self, event):
        """Handle item click for selection"""
        if hasattr(self.parent, 'select_item'):
            self.parent.select_item(self)
    
    def set_selected(self, selected: bool):
        """Set selection state"""
        self.selected = selected
        self.update_colors()
    
    def pack(self, **kwargs):
        """Pack the item frame"""
        self.frame.pack(**kwargs)
    
    def destroy(self):
        """Destroy the item"""
        if self.frame:
            self.frame.destroy()

class PipelineEditorPanel:
    """Center panel for editing the effects pipeline"""
    
    def __init__(self, parent, pipeline: EffectPipeline, on_pipeline_changed: Callable[[], None]):
        self.parent = parent
        self.pipeline = pipeline
        self.on_pipeline_changed = on_pipeline_changed
        
        self.panel = None
        self.effect_items = []
        self.selected_item = None
        self.editing_enabled = False
        
    def pack_panel(self, **kwargs):
        """Pack the panel into parent with given options"""
        if self.panel is None:
            self.create_panel()
        self.panel.grid(**kwargs)
    
    def create_panel(self):
        """Create the pipeline editor panel"""
        # Main frame
        self.panel = ctk.CTkFrame(self.parent)
        
        # Header
        header_frame = ctk.CTkFrame(self.panel)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="🎬 EFFECTS PIPELINE",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(side="left", padx=10, pady=5)
        
        # Pipeline info
        self.info_label = ctk.CTkLabel(
            header_frame,
            text="No effects",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.info_label.pack(side="right", padx=10, pady=5)
        
        # Control buttons
        controls_frame = ctk.CTkFrame(self.panel)
        controls_frame.pack(fill="x", padx=5, pady=2)
        
        # Left side controls
        left_controls = ctk.CTkFrame(controls_frame)
        left_controls.pack(side="left", padx=5)
        
        self.add_button = ctk.CTkButton(
            left_controls,
            text="+ ADD EFFECT",
            command=self.show_add_effect_menu,
            width=120,
            state="disabled"
        )
        self.add_button.pack(side="left", padx=2)
        
        self.reorder_button = ctk.CTkButton(
            left_controls,
            text="🔄 REORDER",
            command=self.toggle_reorder_mode,
            width=100,
            state="disabled"
        )
        self.reorder_button.pack(side="left", padx=2)
        
        # Right side controls
        right_controls = ctk.CTkFrame(controls_frame)
        right_controls.pack(side="right", padx=5)
        
        self.save_button = ctk.CTkButton(
            right_controls,
            text="💾 SAVE",
            command=self.save_pipeline,
            width=80
        )
        self.save_button.pack(side="left", padx=2)
        
        self.load_button = ctk.CTkButton(
            right_controls,
            text="📁 LOAD",
            command=self.load_pipeline,
            width=80
        )
        self.load_button.pack(side="left", padx=2)
        
        # Pipeline container
        pipeline_container = ctk.CTkFrame(self.panel)
        pipeline_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Pipeline scroll area
        self.pipeline_scroll = ctk.CTkScrollableFrame(pipeline_container)
        self.pipeline_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Empty state
        self.empty_label = ctk.CTkLabel(
            self.pipeline_scroll,
            text="🎭 No effects in pipeline\n\nLoad a video and add effects to get started!",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.empty_label.pack(expand=True, pady=50)
        
        # Initialize with current pipeline
        self.refresh_pipeline()
    
    def enable_editing(self):
        """Enable editing controls"""
        self.editing_enabled = True
        self.add_button.configure(state="normal")
        self.reorder_button.configure(state="normal")
        
    def disable_editing(self):
        """Disable editing controls"""
        self.editing_enabled = False
        self.add_button.configure(state="disabled")
        self.reorder_button.configure(state="disabled")
    
    def show_add_effect_menu(self):
        """Show menu to add effects (placeholder)"""
        messagebox.showinfo("Add Effect", "Use the Effects Library on the left to add effects to the pipeline.")
    
    def toggle_reorder_mode(self):
        """Toggle drag-and-drop reorder mode"""
        messagebox.showinfo("Reorder Mode", "Drag and drop functionality will be implemented in the next version.")
    
    def save_pipeline(self):
        """Save pipeline to file"""
        from tkinter import filedialog
        
        file_path = filedialog.asksaveasfilename(
            title="Save Pipeline",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.pipeline.save_to_file(file_path)
                messagebox.showinfo("Success", f"Pipeline saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save pipeline: {e}")
    
    def load_pipeline(self):
        """Load pipeline from file"""
        from tkinter import filedialog
        
        file_path = filedialog.askopenfilename(
            title="Load Pipeline",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.pipeline.load_from_file(file_path)
                self.refresh_pipeline()
                self.on_pipeline_changed()
                messagebox.showinfo("Success", f"Pipeline loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load pipeline: {e}")
    
    def refresh_pipeline(self):
        """Refresh the pipeline display"""
        # Clear existing items
        self.clear_items()
        
        # Update info
        self.update_pipeline_info()
        
        if len(self.pipeline.effects) == 0:
            # Show empty state
            self.empty_label.pack(expand=True, pady=50)
        else:
            # Hide empty state
            self.empty_label.pack_forget()
            
            # Create effect items
            for i, effect in enumerate(self.pipeline.effects):
                item = EffectItem(
                    self.pipeline_scroll,
                    effect,
                    on_configure=self.configure_effect,
                    on_delete=self.delete_effect,
                    on_move_up=self.move_effect_up,
                    on_move_down=self.move_effect_down,
                    on_toggle=self.toggle_effect
                )
                item.pack(fill="x", padx=5, pady=2)
                self.effect_items.append(item)
    
    def clear_items(self):
        """Clear all effect items"""
        for item in self.effect_items:
            item.destroy()
        self.effect_items.clear()
        self.selected_item = None
    
    def update_pipeline_info(self):
        """Update pipeline information display"""
        info = self.pipeline.get_pipeline_info()
        
        if info['total_effects'] == 0:
            text = "No effects"
        else:
            text = f"{info['enabled_effects']}/{info['total_effects']} effects enabled"
            
        self.info_label.configure(text=text)
    
    def select_item(self, item: EffectItem):
        """Select an effect item"""
        # Deselect previous
        if self.selected_item:
            self.selected_item.set_selected(False)
        
        # Select new
        self.selected_item = item
        item.set_selected(True)
    
    def configure_effect(self, effect_id: str):
        """Configure effect parameters"""
        effect = self.pipeline.get_effect(effect_id)
        if effect:
            # For now, show a simple info dialog
            # In a full implementation, this would open a parameter dialog
            param_info = []
            for name, param in effect.parameters.items():
                param_info.append(f"{name}: {param.value}")
            
            param_text = "\n".join(param_info) if param_info else "No parameters"
            
            messagebox.showinfo(
                f"Configure {effect.name}",
                f"Effect: {effect.name}\nCategory: {effect.category.value}\n\nParameters:\n{param_text}\n\nFull parameter editing dialog will be implemented next."
            )
    
    def delete_effect(self, effect_id: str):
        """Delete effect from pipeline"""
        if self.pipeline.remove_effect(effect_id):
            self.refresh_pipeline()
            self.on_pipeline_changed()
    
    def move_effect_up(self, effect_id: str):
        """Move effect up in pipeline"""
        effect = self.pipeline.get_effect(effect_id)
        if effect:
            current_index = self.pipeline.effects.index(effect)
            if current_index > 0:
                new_index = current_index - 1
                if self.pipeline.move_effect(effect_id, new_index):
                    self.refresh_pipeline()
                    self.on_pipeline_changed()
    
    def move_effect_down(self, effect_id: str):
        """Move effect down in pipeline"""
        effect = self.pipeline.get_effect(effect_id)
        if effect:
            current_index = self.pipeline.effects.index(effect)
            if current_index < len(self.pipeline.effects) - 1:
                new_index = current_index + 1
                if self.pipeline.move_effect(effect_id, new_index):
                    self.refresh_pipeline()
                    self.on_pipeline_changed()
    
    def toggle_effect(self, effect_id: str):
        """Toggle effect enabled state"""
        self.update_pipeline_info()
        self.on_pipeline_changed()
    
    def delete_selected(self):
        """Delete currently selected effect"""
        if self.selected_item:
            self.delete_effect(self.selected_item.effect.effect_id)
    
    def get_selected_effect(self) -> Optional[BaseEffect]:
        """Get currently selected effect"""
        if self.selected_item:
            return self.selected_item.effect
        return None
    
    def add_effect_to_pipeline(self, effect_type: str, effect_name: str):
        """Add effect to pipeline (called from main app)"""
        effect_id = self.pipeline.add_effect(effect_type, effect_name)
        self.refresh_pipeline()
        return effect_id