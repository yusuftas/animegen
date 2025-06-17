"""
Effects Library Panel - Left panel showing categorized effects
"""

import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from typing import Callable, Optional, Dict, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ui_standalone.models.effect_models import EffectCategory
from ui_standalone.models.effect_adapter import ProductionEffectFactory

class EffectsLibraryPanel:
    """Left panel containing categorized effects library"""
    
    def __init__(self, parent, on_effect_selected: Callable[[str, str], None]):
        self.parent = parent
        self.on_effect_selected = on_effect_selected
        
        # Get available effects from production engines
        self.available_effects = ProductionEffectFactory.get_available_effects()
        
        # Create main panel
        self.panel = None
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.on_search_changed)
        
        # Track expanded categories
        self.expanded_categories = set()
        
        # Effect buttons for search filtering
        self.effect_buttons = []
        
    def pack_panel(self, **kwargs):
        """Pack the panel into parent with given options"""
        if self.panel is None:
            self.create_panel()
        self.panel.grid(**kwargs)
        
    def create_panel(self):
        """Create the effects library panel"""
        # Main frame
        self.panel = ctk.CTkFrame(self.parent)
        
        # Title
        title_frame = ctk.CTkFrame(self.panel)
        title_frame.pack(fill="x", padx=5, pady=5)
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="🎭 EFFECTS LIBRARY",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(pady=5)
        
        # Search box
        search_frame = ctk.CTkFrame(self.panel)
        search_frame.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(search_frame, text="🔍", font=ctk.CTkFont(size=14)).pack(side="left", padx=5)
        
        self.search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="Search effects...",
            textvariable=self.search_var
        )
        self.search_entry.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        
        # Scrollable effects area
        self.scrollable_frame = ctk.CTkScrollableFrame(self.panel)
        self.scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create effect categories
        self.create_effect_categories()
        
        # Preset section
        self.create_preset_section()
        
    def create_effect_categories(self):
        """Create categorized effect sections"""
        category_icons = {
            "Motion Effects": "🏃",
            "Anime Effects": "⚡",
            "Color Effects": "🎨",
            "Text Effects": "📝", 
            "Audio Sync": "🎵",
            "Transitions": "🔄"
        }
        
        self.category_frames = {}
        self.category_buttons = {}
        
        for category, effects in self.available_effects.items():
            # Category header
            category_frame = ctk.CTkFrame(self.scrollable_frame)
            category_frame.pack(fill="x", pady=2)
            
            # Expandable header
            header_button = ctk.CTkButton(
                category_frame,
                text=f"{category_icons.get(category, '🔹')} {category} ({len(effects)})",
                command=lambda cat=category: self.toggle_category(cat),
                height=30,
                font=ctk.CTkFont(size=12, weight="bold"),
                fg_color="transparent",
                text_color=("black", "white"),
                hover_color=("gray90", "gray30")
            )
            header_button.pack(fill="x", padx=2, pady=2)
            
            # Effects container (initially hidden)
            effects_container = ctk.CTkFrame(category_frame)
            
            # Store references
            self.category_frames[category] = effects_container
            self.category_buttons[category] = header_button
            
            # Create effect buttons
            for effect_info in effects:
                self.create_effect_button(effects_container, effect_info, category)
    
    def create_effect_button(self, container, effect_info, category):
        """Create individual effect button"""
        # Determine effect identifier - use 'id' for production effects, 'type' for legacy
        effect_id = effect_info.get('id', effect_info.get('type', ''))
        effect_name = effect_info.get('name', '')
        
        effect_button = ctk.CTkButton(
            container,
            text=f"• {effect_name}",
            command=lambda: self.on_effect_selected(effect_id, effect_name),
            height=25,
            font=ctk.CTkFont(size=11),
            fg_color="transparent",
            text_color=("gray20", "gray80"),
            hover_color=("blue", "blue"),
            anchor="w"
        )
        effect_button.pack(fill="x", padx=15, pady=1)
        
        # Store for search filtering
        self.effect_buttons.append({
            'button': effect_button,
            'info': effect_info,
            'category': category,
            'container': container
        })
        
        # Add tooltip
        description = effect_info.get('description', f"{effect_name} - {category}")
        self.create_tooltip(effect_button, description)
        
    def create_tooltip(self, widget, text):
        """Create tooltip for widget"""
        # Store tooltip reference on the widget itself
        widget.tooltip = None
        
        def show_tooltip(event):
            # Don't create new tooltip if one already exists
            if widget.tooltip is not None:
                return
                
            widget.tooltip = tk.Toplevel()
            widget.tooltip.wm_overrideredirect(True)
            widget.tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = tk.Label(
                widget.tooltip,
                text=text,
                background="lightyellow",
                relief="solid",
                borderwidth=1,
                font=("Arial", "9")
            )
            label.pack()
            
        def hide_tooltip(event):
            # Destroy tooltip if it exists
            if widget.tooltip is not None:
                widget.tooltip.destroy()
                widget.tooltip = None
            
        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)
        
    def toggle_category(self, category):
        """Toggle category expansion"""
        container = self.category_frames[category]
        button = self.category_buttons[category]
        
        if category in self.expanded_categories:
            # Collapse
            container.pack_forget()
            self.expanded_categories.remove(category)
            # Update button text to show collapsed state
            icon = button.cget("text").split()[0]
            text = button.cget("text")
            button.configure(text=text.replace("▼", "▶") if "▼" in text else f"▶ {text[2:]}")
        else:
            # Expand
            container.pack(fill="x", padx=10, pady=2)
            self.expanded_categories.add(category)
            # Update button text to show expanded state
            text = button.cget("text")
            button.configure(text=text.replace("▶", "▼") if "▶" in text else f"▼ {text[2:]}")
    
    def create_preset_section(self):
        """Create preset effects section"""
        # Separator
        separator = ctk.CTkFrame(self.scrollable_frame, height=2)
        separator.pack(fill="x", pady=10)
        
        # Preset header
        preset_header = ctk.CTkLabel(
            self.scrollable_frame,
            text="⚡ QUICK PRESETS",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        preset_header.pack(pady=5)
        
        # Preset buttons
        presets = [
            ("Action Scene", "action_scene", "🥊"),
            ("Power Up", "power_up", "⚡"),
            ("Emotional", "emotional_moment", "💖"),
            ("Speed Boost", "speed_boost", "💨"),
            ("Impact Hit", "impact_hit", "💥")
        ]
        
        preset_frame = ctk.CTkFrame(self.scrollable_frame)
        preset_frame.pack(fill="x", padx=5, pady=5)
        
        for name, preset_key, icon in presets:
            preset_button = ctk.CTkButton(
                preset_frame,
                text=f"{icon} {name}",
                command=lambda key=preset_key: self.apply_preset(key),
                height=35,
                font=ctk.CTkFont(size=11),
                fg_color=("blue", "blue"),
                hover_color=("darkblue", "darkblue")
            )
            preset_button.pack(fill="x", padx=5, pady=2)
    
    def apply_preset(self, preset_key):
        """Apply a preset effect combination"""
        # This will be handled by the main application
        if hasattr(self, 'on_preset_selected'):
            self.on_preset_selected(preset_key)
        else:
            # For now, just show info
            import tkinter.messagebox as messagebox
            messagebox.showinfo("Preset", f"Applying preset: {preset_key}")
    
    def on_search_changed(self, *args):
        """Handle search text changes"""
        search_text = self.search_var.get().lower()
        
        if not search_text:
            # Show all effects
            self.show_all_effects()
            return
        
        # Hide all effect buttons first
        for effect_data in self.effect_buttons:
            effect_data['button'].pack_forget()
        
        # Show matching effects
        matching_effects = []
        for effect_data in self.effect_buttons:
            effect_info = effect_data['info']
            if (search_text in effect_info['name'].lower() or 
                search_text in effect_info['description'].lower()):
                effect_data['button'].pack(fill="x", padx=15, pady=1)
                matching_effects.append(effect_data)
        
        # Auto-expand categories with matches
        categories_with_matches = set()
        for effect_data in matching_effects:
            categories_with_matches.add(effect_data['category'])
        
        for category in categories_with_matches:
            if category not in self.expanded_categories:
                self.toggle_category(category)
    
    def show_all_effects(self):
        """Show all effects (clear search filter)"""
        for effect_data in self.effect_buttons:
            container = effect_data['container']
            category = effect_data['category']
            
            # Only show if category is expanded
            if category in self.expanded_categories:
                effect_data['button'].pack(fill="x", padx=15, pady=1)
            else:
                effect_data['button'].pack_forget()
    
    def expand_all_categories(self):
        """Expand all effect categories"""
        for category in self.available_effects.keys():
            if category not in self.expanded_categories:
                self.toggle_category(category)
    
    def collapse_all_categories(self):
        """Collapse all effect categories"""
        for category in list(self.expanded_categories):
            self.toggle_category(category)
    
    def get_effect_info(self, effect_type: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an effect"""
        for effects in self.available_effects.values():
            for effect_info in effects:
                if effect_info['type'] == effect_type:
                    return effect_info
        return None
    
    def refresh_library(self):
        """Refresh the effects library (useful after updates)"""
        self.available_effects = ProductionEffectFactory.get_available_effects()
        
        # Clear existing content
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Recreate content
        self.effect_buttons.clear()
        self.category_frames.clear()
        self.category_buttons.clear()
        self.expanded_categories.clear()
        
        self.create_effect_categories()
        self.create_preset_section()
    
    def set_preset_callback(self, callback: Callable[[str], None]):
        """Set callback for preset selection"""
        self.on_preset_selected = callback