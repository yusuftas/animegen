#!/usr/bin/env python3
"""
Test script for parameter dialogs to ensure they work correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import customtkinter as ctk
from ui_standalone.models.effect_adapter import ProductionEffectFactory
from ui_standalone.components.parameter_dialogs import show_parameter_dialog

class ParameterDialogTester:
    """Test application for parameter dialogs"""
    
    def __init__(self):
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Parameter Dialog Tester")
        self.root.geometry("800x600")
        
        self.create_ui()
        
        # Get available effects
        self.effects_registry = ProductionEffectFactory.get_available_effects()
        
    def create_ui(self):
        """Create the test UI"""
        # Header
        header = ctk.CTkLabel(
            self.root,
            text="🧪 Parameter Dialog Tester",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        header.pack(pady=20)
        
        # Instructions
        instructions = ctk.CTkLabel(
            self.root,
            text="Select an effect category and then an effect to test its parameter dialog",
            font=ctk.CTkFont(size=12)
        )
        instructions.pack(pady=10)
        
        # Category selection
        category_frame = ctk.CTkFrame(self.root)
        category_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(category_frame, text="Effect Category:", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.category_var = ctk.StringVar(value="Motion Effects")
        self.category_dropdown = ctk.CTkOptionMenu(
            category_frame,
            values=["Motion Effects", "Anime Effects", "Color Effects", "Text Effects", "Audio Sync", "Transitions"],
            variable=self.category_var,
            command=self.on_category_changed
        )
        self.category_dropdown.pack(pady=5)
        
        # Effect selection
        effect_frame = ctk.CTkFrame(self.root)
        effect_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(effect_frame, text="Effect:", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.effect_var = ctk.StringVar()
        self.effect_dropdown = ctk.CTkOptionMenu(
            effect_frame,
            values=[""],
            variable=self.effect_var
        )
        self.effect_dropdown.pack(pady=5)
        
        # Test button
        test_button = ctk.CTkButton(
            self.root,
            text="🔧 Test Parameter Dialog",
            command=self.test_parameter_dialog,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40
        )
        test_button.pack(pady=20)
        
        # Status area
        self.status_frame = ctk.CTkFrame(self.root)
        self.status_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        ctk.CTkLabel(self.status_frame, text="Status:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=10, pady=5)
        
        self.status_text = ctk.CTkTextbox(self.status_frame)
        self.status_text.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Initialize with first category
        self.on_category_changed("Motion Effects")
        
    def on_category_changed(self, category: str):
        """Handle category selection change"""
        try:
            # Get effects for this category
            effects = self.effects_registry.get(category, [])
            effect_names = [effect['name'] for effect in effects]
            
            if effect_names:
                self.effect_dropdown.configure(values=effect_names)
                self.effect_var.set(effect_names[0])
                self.log_status(f"Found {len(effect_names)} effects in {category}")
            else:
                self.effect_dropdown.configure(values=["No effects found"])
                self.effect_var.set("No effects found")
                self.log_status(f"No effects found in {category}")
                
        except Exception as e:
            self.log_status(f"Error loading effects for {category}: {e}")
    
    def test_parameter_dialog(self):
        """Test the parameter dialog for selected effect"""
        try:
            category = self.category_var.get()
            effect_name = self.effect_var.get()
            
            if not effect_name or effect_name == "No effects found":
                self.log_status("No effect selected")
                return
            
            self.log_status(f"Testing parameter dialog for {effect_name} ({category})...")
            
            # Find the effect in registry
            effects = self.effects_registry.get(category, [])
            selected_effect = None
            
            for effect in effects:
                if effect['name'] == effect_name:
                    selected_effect = effect
                    break
            
            if not selected_effect:
                self.log_status(f"Could not find effect {effect_name}")
                return
            
            # Create effect instance
            effect_instance = ProductionEffectFactory.create_effect(selected_effect['id'])
            self.log_status(f"Created effect instance with {len(effect_instance.parameters)} parameters")
            
            # Test with a simulated video duration of 30 seconds
            test_video_duration = 30.0
            self.log_status(f"Testing with video duration: {test_video_duration}s")
            self.log_status("- Start time will be bounded: 0s ≤ start_time ≤ 29.9s")
            self.log_status("- Duration will be bounded: 0.1s ≤ duration ≤ (30s - start_time)")
            self.log_status("- Both parameters will update bounds dynamically")
            self.log_status("- Audio Sync and Transition effects now use correct interfaces")
            
            # Show parameter dialog
            success = show_parameter_dialog(
                parent=self.root,
                effect=effect_instance,
                on_preview_callback=self.on_preview_callback,
                on_apply_callback=self.on_apply_callback,
                video_duration=test_video_duration
            )
            
            if success:
                self.log_status("✅ Parameter dialog completed successfully")
                self.log_effect_parameters(effect_instance)
            else:
                self.log_status("❌ Parameter dialog was cancelled")
            
        except Exception as e:
            self.log_status(f"❌ Error testing parameter dialog: {e}")
            import traceback
            self.log_status(traceback.format_exc())
    
    def on_preview_callback(self, effect):
        """Handle preview callback"""
        self.log_status(f"🔍 Preview callback triggered for {effect.name}")
    
    def on_apply_callback(self, effect):
        """Handle apply callback"""
        self.log_status(f"✅ Apply callback triggered for {effect.name}")
    
    def log_effect_parameters(self, effect):
        """Log current effect parameters"""
        self.log_status(f"Current parameters for {effect.name}:")
        for param_name, param in effect.parameters.items():
            self.log_status(f"  {param_name}: {param.value} ({param.param_type})")
    
    def log_status(self, message: str):
        """Log a status message"""
        current_text = self.status_text.get("1.0", "end-1c")
        if current_text:
            new_text = current_text + "\n" + message
        else:
            new_text = message
        
        self.status_text.delete("1.0", "end")
        self.status_text.insert("1.0", new_text)
        
        # Auto-scroll to bottom
        self.status_text.see("end")
        
        # Update UI
        self.root.update_idletasks()
    
    def run(self):
        """Run the test application"""
        try:
            self.log_status("🚀 Parameter Dialog Tester started")
            self.log_status(f"Found {len(self.effects_registry)} effect categories")
            
            self.root.mainloop()
            
        except Exception as e:
            print(f"Error running tester: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    print("🧪 Starting Parameter Dialog Tester...")
    
    try:
        app = ParameterDialogTester()
        app.run()
    except Exception as e:
        print(f"Failed to start tester: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()