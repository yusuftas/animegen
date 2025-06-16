#!/usr/bin/env python3
"""
Anime Effects Studio - Standalone Video Effects Pipeline UI
Main application window with comprehensive video effects editing capabilities.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import os
import sys
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ui_standalone.components.effects_library import EffectsLibraryPanel
from ui_standalone.components.pipeline_editor import PipelineEditorPanel
from ui_standalone.components.video_preview import VideoPreviewPanel
from ui_standalone.models.effect_pipeline import EffectPipeline
from ui_standalone.utils.video_processor import VideoProcessor

class AnimeEffectsStudio:
    """Main application class for the Anime Effects Studio"""
    
    def __init__(self):
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize main window
        self.root = ctk.CTk()
        self.root.title("Anime Effects Studio - Video Effects Pipeline")
        self.root.geometry("1600x900")
        self.root.minsize(1200, 700)
        
        # Initialize data models
        self.effect_pipeline = EffectPipeline()
        self.video_processor = VideoProcessor()
        self.current_video_path = None
        
        # Setup UI
        self.setup_ui()
        self.setup_bindings()
        
    def setup_ui(self):
        """Setup the main UI layout"""
        # Create main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create header
        self.create_header()
        
        # Create main content area with three panels
        self.create_main_content()
        
    def create_header(self):
        """Create the header with file loading and global controls"""
        header_frame = ctk.CTkFrame(self.main_container)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="🎬 ANIME EFFECTS STUDIO",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(side="left", padx=20, pady=10)
        
        # File loading section
        file_frame = ctk.CTkFrame(header_frame)
        file_frame.pack(side="left", padx=20, pady=5, fill="x", expand=True)
        
        ctk.CTkLabel(file_frame, text="📁 Load Video:", font=ctk.CTkFont(size=14)).pack(side="left", padx=5)
        
        self.file_label = ctk.CTkLabel(
            file_frame,
            text="No video selected",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.file_label.pack(side="left", padx=10)
        
        self.load_button = ctk.CTkButton(
            file_frame,
            text="Choose File",
            command=self.load_video,
            width=100
        )
        self.load_button.pack(side="left", padx=5)
        
        # Preview quality selector
        quality_frame = ctk.CTkFrame(header_frame)
        quality_frame.pack(side="right", padx=20, pady=5)
        
        ctk.CTkLabel(quality_frame, text="🎬 Preview Quality:", font=ctk.CTkFont(size=12)).pack(side="left", padx=5)
        
        self.quality_var = tk.StringVar(value="HD")
        quality_combo = ctk.CTkComboBox(
            quality_frame,
            variable=self.quality_var,
            values=["HD", "Full HD", "4K"],
            width=100
        )
        quality_combo.pack(side="left", padx=5)
        
    def create_main_content(self):
        """Create the main three-panel layout"""
        # Main content frame
        content_frame = ctk.CTkFrame(self.main_container)
        content_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Configure grid weights
        content_frame.grid_columnconfigure(0, weight=1)  # Effects library
        content_frame.grid_columnconfigure(1, weight=2)  # Pipeline editor
        content_frame.grid_columnconfigure(2, weight=2)  # Video preview
        content_frame.grid_rowconfigure(0, weight=1)
        
        # Left Panel - Effects Library
        self.effects_library = EffectsLibraryPanel(
            content_frame,
            on_effect_selected=self.add_effect_to_pipeline
        )
        self.effects_library.pack_panel(row=0, column=0, sticky="nsew", padx=(5, 2))
        
        # Center Panel - Pipeline Editor
        self.pipeline_editor = PipelineEditorPanel(
            content_frame,
            self.effect_pipeline,
            on_pipeline_changed=self.on_pipeline_changed
        )
        self.pipeline_editor.pack_panel(row=0, column=1, sticky="nsew", padx=2)
        
        # Right Panel - Video Preview
        self.video_preview = VideoPreviewPanel(
            content_frame,
            on_export=self.export_video
        )
        self.video_preview.pack_panel(row=0, column=2, sticky="nsew", padx=(2, 5))
        
    def setup_bindings(self):
        """Setup event bindings and keyboard shortcuts"""
        # Keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.load_video())
        self.root.bind('<Control-s>', lambda e: self.save_pipeline())
        self.root.bind('<Control-e>', lambda e: self.export_video())
        self.root.bind('<Delete>', lambda e: self.pipeline_editor.delete_selected())
        self.root.bind('<space>', lambda e: self.video_preview.toggle_playback())
        
        # Window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def load_video(self):
        """Open file dialog to load a video file"""
        file_types = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("MP4 files", "*.mp4"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=file_types
        )
        
        if file_path:
            self.current_video_path = file_path
            filename = os.path.basename(file_path)
            self.file_label.configure(text=filename, text_color="white")
            
            # Load video in preview panel
            self.video_preview.load_video(file_path)
            
            # Update UI state
            self.pipeline_editor.enable_editing()
            
            messagebox.showinfo("Video Loaded", f"Successfully loaded: {filename}")
    
    def add_effect_to_pipeline(self, effect_type, effect_name):
        """Add an effect to the pipeline"""
        if not self.current_video_path:
            messagebox.showwarning("No Video", "Please load a video file first.")
            return
            
        try:
            # Add effect to pipeline model
            effect_id = self.effect_pipeline.add_effect(effect_type, effect_name)
            
            # Update pipeline editor if it exists
            if hasattr(self, 'pipeline_editor') and self.pipeline_editor:
                self.pipeline_editor.refresh_pipeline()
            
            # Trigger preview update
            self.update_preview()
            
            # Update status instead of popup
            print(f"✅ Added effect: {effect_name}")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error adding effect: {error_details}")
            messagebox.showerror("Error", f"Failed to add effect: {str(e)}")
        
    def on_pipeline_changed(self):
        """Handle pipeline changes"""
        self.update_preview()
        
    def update_preview(self):
        """Update video preview with current pipeline"""
        if not self.current_video_path:
            return
            
        # Process video with current pipeline in background
        threading.Thread(
            target=self._update_preview_async,
            daemon=True
        ).start()
        
    def _update_preview_async(self):
        """Async preview update"""
        try:
            # Show processing indicator
            self.video_preview.show_processing()
            
            # Skip preview generation if no effects
            if len(self.effect_pipeline.effects) == 0:
                print("⏭️ No effects to apply, skipping preview generation")
                self.video_preview.hide_processing()
                return
            
            print(f"🔄 Generating preview with {len(self.effect_pipeline.get_enabled_effects())} effects...")
            
            # Apply effects pipeline
            preview_path = self.video_processor.generate_preview(
                self.current_video_path,
                self.effect_pipeline
            )
            
            if preview_path:
                print(f"✅ Preview generated: {preview_path}")
                # Update preview panel
                self.video_preview.update_preview(preview_path)
            else:
                print("⚠️ Preview generation returned no path")
            
        except Exception as e:
            print(f"❌ Preview generation failed: {str(e)}")
            # Don't show popup for preview errors, just log them
        finally:
            self.video_preview.hide_processing()
            
    def export_video(self):
        """Export final video with all effects applied"""
        if not self.current_video_path:
            messagebox.showwarning("No Video", "Please load a video file first.")
            return
            
        if len(self.effect_pipeline.effects) == 0:
            messagebox.showwarning("No Effects", "Please add some effects before exporting.")
            return
            
        # Get export settings from preview panel
        export_settings = self.video_preview.get_export_settings()
        
        # Choose output file
        output_path = filedialog.asksaveasfilename(
            title="Export Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        
        if output_path:
            # Start export in background
            threading.Thread(
                target=self._export_video_async,
                args=(output_path, export_settings),
                daemon=True
            ).start()
            
    def _export_video_async(self, output_path, settings):
        """Async video export"""
        try:
            # Show export progress
            self.video_preview.show_export_progress()
            
            # Process and export video
            self.video_processor.export_video(
                self.current_video_path,
                self.effect_pipeline,
                output_path,
                settings,
                progress_callback=self.video_preview.update_export_progress
            )
            
            messagebox.showinfo("Export Complete", f"Video exported successfully to:\n{output_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export video: {str(e)}")
        finally:
            self.video_preview.hide_export_progress()
            
    def save_pipeline(self):
        """Save current effects pipeline to file"""
        file_path = filedialog.asksaveasfilename(
            title="Save Pipeline",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.effect_pipeline.save_to_file(file_path)
                messagebox.showinfo("Pipeline Saved", f"Pipeline saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save pipeline: {str(e)}")
                
    def load_pipeline(self):
        """Load effects pipeline from file"""
        file_path = filedialog.askopenfilename(
            title="Load Pipeline",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.effect_pipeline.load_from_file(file_path)
                self.pipeline_editor.refresh_pipeline()
                self.update_preview()
                messagebox.showinfo("Pipeline Loaded", f"Pipeline loaded from: {file_path}")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load pipeline: {str(e)}")
                
    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit Anime Effects Studio?"):
            # Clean up temporary files
            self.video_processor.cleanup()
            self.root.destroy()
            
    def run(self):
        """Start the application"""
        self.root.mainloop()


def main():
    """Entry point for the application"""
    try:
        app = AnimeEffectsStudio()
        app.run()
    except Exception as e:
        print(f"Failed to start Anime Effects Studio: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application: {e}")


if __name__ == "__main__":
    main()