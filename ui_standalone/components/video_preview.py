"""
Video Preview Panel - Right panel with video preview and export controls
"""

import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from typing import Callable, Optional, Dict, Any
import threading
import time
import os
import sys
from pathlib import Path
from PIL import Image, ImageTk
import cv2
from .export_dialog import ExportDialog

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class VideoPreviewPanel:
    """Right panel for video preview and export controls"""
    
    def __init__(self, parent, on_export: Callable[[], None]):
        self.parent = parent
        self.on_export = on_export
        
        self.panel = None
        self.current_video_path = None
        self.preview_video_path = None
        self.is_playing = False
        self.current_time = 0.0
        self.video_duration = 0.0
        self.playback_thread = None
        
        # Export dialog
        self.export_dialog = None
        
        # Processing state
        self.is_processing = False
        self.export_progress = 0.0
        
        # Video properties
        self.video_fps = 30
        self.video_resolution = (1920, 1080)
        
    def pack_panel(self, **kwargs):
        """Pack the panel into parent with given options"""
        if self.panel is None:
            self.create_panel()
        self.panel.grid(**kwargs)
    
    def create_panel(self):
        """Create the video preview panel"""
        # Main frame
        self.panel = ctk.CTkFrame(self.parent)
        
        # Header
        header_frame = ctk.CTkFrame(self.panel)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="🎬 LIVE PREVIEW",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(side="left", padx=10, pady=5)
        
        # Status indicator
        self.status_label = ctk.CTkLabel(
            header_frame,
            text="No video loaded",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.status_label.pack(side="right", padx=10, pady=5)
        
        # Video preview area
        self.create_preview_area()
        
        # Video controls
        self.create_video_controls()
        
        # Timeline
        self.create_timeline()
    
    def create_preview_area(self):
        """Create video preview display area"""
        # Preview container with responsive sizing
        preview_container = ctk.CTkFrame(self.panel)
        preview_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Configure preview container for responsive sizing
        preview_container.grid_rowconfigure(0, weight=1)
        preview_container.grid_columnconfigure(0, weight=1)
        
        # Preview frame with responsive height
        self.preview_frame = ctk.CTkFrame(
            preview_container,
            fg_color="black",
            corner_radius=10
        )
        self.preview_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Video display canvas for actual video rendering with responsive sizing
        try:
            import tkinter as tk
            self.video_canvas = tk.Canvas(
                self.preview_frame,
                bg="black",
                highlightthickness=0
            )
            self.video_canvas.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Bind resize events for responsive video scaling
            self.video_canvas.bind('<Configure>', self.on_canvas_resize)
            
        except:
            # Fallback to label if canvas fails
            self.video_canvas = None
        
        # Video display label (fallback/placeholder)
        self.video_display = ctk.CTkLabel(
            self.preview_frame,
            text="📺\n\nVideo Preview\n\nLoad a video to see preview here",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        if not self.video_canvas:
            self.video_display.place(relx=0.5, rely=0.5, anchor="center")
        
        # Processing overlay
        self.processing_overlay = ctk.CTkFrame(
            self.preview_frame,
            fg_color=("gray90", "gray20"),
            corner_radius=10
        )
        
        self.processing_label = ctk.CTkLabel(
            self.processing_overlay,
            text="🔄 Processing...\n\nApplying effects to video",
            font=ctk.CTkFont(size=14),
            text_color=("black", "white")
        )
        self.processing_label.pack(expand=True, pady=20)
        
        # Initially hidden
        self.processing_overlay.place_forget()
        
        # Store video dimensions for aspect ratio calculation
        self.video_aspect_ratio = 16/9  # Default aspect ratio
        self.current_frame = None
    
    def on_canvas_resize(self, event):
        """Handle canvas resize events to maintain aspect ratio"""
        if self.current_frame is not None and self.video_canvas:
            # Redraw the current frame with new canvas size
            self.display_frame_on_canvas(self.current_frame)
    
    def display_frame_on_canvas(self, frame_rgb):
        """Display frame on canvas with proper aspect ratio preservation"""
        if not self.video_canvas:
            return
            
        try:
            # Get current canvas dimensions
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            # Use minimum size if canvas not yet rendered
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 640
                canvas_height = 360
            
            # Calculate video dimensions that maintain aspect ratio and fit in canvas
            canvas_aspect = canvas_width / canvas_height
            
            if self.video_aspect_ratio > canvas_aspect:
                # Video is wider than canvas - fit to width
                video_width = canvas_width
                video_height = int(canvas_width / self.video_aspect_ratio)
            else:
                # Video is taller than canvas - fit to height  
                video_height = canvas_height
                video_width = int(canvas_height * self.video_aspect_ratio)
            
            # Resize frame maintaining aspect ratio
            image = Image.fromarray(frame_rgb)
            image = image.resize((video_width, video_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and display centered
            photo = ImageTk.PhotoImage(image)
            self.video_canvas.delete("all")
            
            # Center the video in the canvas
            x_offset = canvas_width // 2
            y_offset = canvas_height // 2
            
            self.video_canvas.create_image(x_offset, y_offset, image=photo, anchor="center")
            
            # Keep reference to prevent garbage collection
            self.video_canvas.photo = photo
            
        except Exception as e:
            print(f"Error displaying frame: {e}")
    
    def create_video_controls(self):
        """Create video playback controls"""
        controls_frame = ctk.CTkFrame(self.panel)
        controls_frame.pack(fill="x", padx=5, pady=(0, 5))
        
        # Playback buttons row
        playback_frame = ctk.CTkFrame(controls_frame)
        playback_frame.pack(pady=8)
        
        # Left side - playback controls
        left_controls = ctk.CTkFrame(playback_frame)
        left_controls.pack(side="left", padx=10)
        
        self.play_button = ctk.CTkButton(
            left_controls,
            text="▶️",
            width=50,
            height=35,
            command=self.toggle_playback,
            font=ctk.CTkFont(size=14),
            state="disabled"  # Disabled until video is loaded
        )
        self.play_button.pack(side="left", padx=2)
        
        self.stop_button = ctk.CTkButton(
            left_controls,
            text="⏹️",
            width=50,
            height=35,
            command=self.stop_playback,
            font=ctk.CTkFont(size=14),
            state="disabled"  # Disabled until video is loaded
        )
        self.stop_button.pack(side="left", padx=2)
        
        # Center - time display
        center_frame = ctk.CTkFrame(playback_frame)
        center_frame.pack(side="left", padx=20)
        
        self.time_label = ctk.CTkLabel(
            center_frame,
            text="00:00 / 00:00",
            font=ctk.CTkFont(size=12),
            width=120
        )
        self.time_label.pack(pady=8)
        
        # Right side - volume control and export
        right_controls = ctk.CTkFrame(playback_frame)
        right_controls.pack(side="right", padx=10)
        
        # Export button
        self.export_button = ctk.CTkButton(
            right_controls,
            text="🎬 Render",
            command=self.show_export_dialog,
            width=80,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.export_button.pack(side="left", padx=5)
        
        # Volume controls
        volume_label = ctk.CTkLabel(right_controls, text="🔊", font=ctk.CTkFont(size=14))
        volume_label.pack(side="left", padx=5)
        
        self.volume_slider = ctk.CTkSlider(
            right_controls,
            from_=0,
            to=100,
            width=80,
            number_of_steps=20,
            command=self.on_volume_change
        )
        self.volume_slider.set(70)
        self.volume_slider.pack(side="left", padx=5)
    
    def on_volume_change(self, value):
        """Handle volume slider changes"""
        # This would be implemented for audio playback
        pass
    
    def create_timeline(self):
        """Create video timeline scrubber"""
        timeline_frame = ctk.CTkFrame(self.panel)
        timeline_frame.pack(fill="x", padx=5, pady=(0, 5))
        
        # Timeline header
        timeline_header = ctk.CTkFrame(timeline_frame)
        timeline_header.pack(fill="x", padx=5, pady=5)
        
        timeline_label = ctk.CTkLabel(
            timeline_header,
            text="🎬 Timeline:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        timeline_label.pack(side="left", padx=5)
        
        # Timeline progress info
        self.timeline_info = ctk.CTkLabel(
            timeline_header,
            text="No video loaded",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.timeline_info.pack(side="right", padx=5)
        
        # Timeline slider container
        slider_container = ctk.CTkFrame(timeline_frame)
        slider_container.pack(fill="x", padx=10, pady=5)
        
        # Timeline slider
        self.timeline_slider = ctk.CTkSlider(
            slider_container,
            from_=0,
            to=100,
            command=self.seek_video,
            height=20,
            state="disabled"  # Disabled until video is loaded
        )
        self.timeline_slider.pack(fill="x", padx=5, pady=5)
        
        # Timeline markers (for effects) - smaller container
        markers_container = ctk.CTkFrame(timeline_frame)
        markers_container.pack(fill="x", padx=10, pady=(0, 5))
        
        self.timeline_markers = ctk.CTkLabel(
            markers_container,
            text="📍 Effect markers will appear here when effects are applied",
            font=ctk.CTkFont(size=9),
            text_color="gray",
            height=20
        )
        self.timeline_markers.pack(fill="x", padx=5, pady=2)
    
    
    def show_export_dialog(self):
        """Show the export dialog"""
        if not self.export_dialog:
            self.export_dialog = ExportDialog(self.panel, self.handle_export_request)
        self.export_dialog.show_dialog()
    
    def handle_export_request(self, export_settings):
        """Handle export request from dialog"""
        # Store export settings for the actual export process
        self.current_export_settings = export_settings
        
        # Call the original export callback with the settings
        if self.on_export:
            self.on_export(export_settings)
    
    
    def load_video(self, video_path: str):
        """Load video for preview"""
        self.current_video_path = video_path
        self.preview_video_path = video_path  # Initially same as original
        
        try:
            # Get real video information
            video_info = self.get_video_info(video_path)
            self.video_duration = video_info.get('duration', 10.0)
            self.video_fps = video_info.get('fps', 30)
            self.video_resolution = video_info.get('resolution', (1920, 1080))
            
            # Reset playback state
            self.current_time = 0.0
            self.is_playing = False
            
            # Update display
            filename = os.path.basename(video_path)
            duration_str = f"{int(self.video_duration//60):02d}:{int(self.video_duration%60):02d}"
            
            if self.video_canvas:
                # Try to load first frame for canvas
                self.load_video_frame(0)
            else:
                self.video_display.configure(
                    text=f"📺\n\n{filename}\n\nDuration: {duration_str}\nResolution: {self.video_resolution[0]}x{self.video_resolution[1]}\n\nClick ▶️ to play"
                )
            
            # Update status
            self.status_label.configure(text=f"Loaded: {filename} ({duration_str})")
            
            # Update timeline
            self.timeline_slider.configure(to=self.video_duration, state="normal")
            self.timeline_info.configure(text=f"Duration: {duration_str} | {self.video_fps} FPS")
            self.update_time_display()
            
            # Enable controls
            self.play_button.configure(state="normal")
            self.stop_button.configure(state="normal")
            
        except Exception as e:
            print(f"Error loading video: {e}")
            # Fallback to basic loading
            filename = os.path.basename(video_path)
            self.video_duration = 10.0
            self.current_time = 0.0
            
            self.video_display.configure(
                text=f"📺\n\n{filename}\n\nVideo loaded\n(Basic preview mode)\n\nClick ▶️ to play"
            )
            
            self.status_label.configure(text=f"Loaded: {filename}")
            self.timeline_slider.configure(to=self.video_duration, state="normal")
            self.play_button.configure(state="normal")
            self.stop_button.configure(state="normal")
    
    def get_video_info(self, video_path: str):
        """Get video information"""
        try:
            # Try using MoviePy if available
            try:
                from moviepy.editor import VideoFileClip
                clip = VideoFileClip(video_path)
                info = {
                    'duration': clip.duration,
                    'fps': clip.fps,
                    'resolution': (clip.w, clip.h),
                    'has_audio': clip.audio is not None
                }
                clip.close()
                return info
            except ImportError:
                pass
            
            # Try using OpenCV if available
            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 10.0
                cap.release()
                
                return {
                    'duration': duration,
                    'fps': fps,
                    'resolution': (width, height),
                    'has_audio': False  # OpenCV doesn't handle audio
                }
            except ImportError:
                pass
                
        except Exception as e:
            print(f"Error getting video info: {e}")
        
        # Fallback values
        return {
            'duration': 10.0,
            'fps': 30,
            'resolution': (1920, 1080),
            'has_audio': False
        }
    
    def load_video_frame(self, time_seconds: float):
        """Load a specific frame from the video"""
        if not self.video_canvas or not self.current_video_path:
            return
            
        try:
            # Try using OpenCV to load frame
            
            # Use preview video if available, otherwise use original
            video_path = self.preview_video_path if self.preview_video_path else self.current_video_path
            cap = cv2.VideoCapture(video_path)
            
            # Set frame position
            frame_number = int(time_seconds * self.video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Store current frame and calculate aspect ratio
                self.current_frame = frame_rgb
                frame_height, frame_width = frame_rgb.shape[:2]
                self.video_aspect_ratio = frame_width / frame_height
                
                # Display frame with proper aspect ratio
                self.display_frame_on_canvas(frame_rgb)
                
                # Hide placeholder text
                if hasattr(self, 'video_display'):
                    self.video_display.place_forget()
                    
            cap.release()
            
        except Exception as e:
            print(f"Error loading video frame: {e}")
            # Show error in placeholder
            if hasattr(self, 'video_display'):
                self.video_display.place(relx=0.5, rely=0.5, anchor="center")
                self.video_display.configure(text=f"📺\n\nVideo Preview\n\nFrame loading failed\nBasic playback mode")

    def toggle_playback(self):
        """Toggle video playback"""
        if not self.current_video_path:
            messagebox.showwarning("No Video", "Please load a video first.")
            return
            
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """Start video playback"""
        if not self.current_video_path:
            return
            
        self.is_playing = True
        self.play_button.configure(text="⏸️")
        
        # Load initial frame if canvas exists
        if self.video_canvas:
            self.load_video_frame(self.current_time)
        
        # Start playback thread
        if not self.playback_thread or not self.playback_thread.is_alive():
            self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self.playback_thread.start()
    
    def pause_playback(self):
        """Pause video playback"""
        self.is_playing = False
        self.play_button.configure(text="▶️")
    
    def stop_playback(self):
        """Stop video playback"""
        self.is_playing = False
        self.current_time = 0.0
        self.play_button.configure(text="▶️")
        self.timeline_slider.set(0)
        self.update_time_display()
        
        # Load first frame
        if self.video_canvas:
            self.load_video_frame(0)
    
    def _playback_loop(self):
        """Background playback loop"""
        last_update = time.time()
        
        while self.is_playing and self.current_time < self.video_duration:
            current = time.time()
            delta = current - last_update
            
            if self.is_playing:
                # Update time based on real time elapsed
                self.current_time += delta
                
                # Clamp to video duration
                if self.current_time >= self.video_duration:
                    self.current_time = self.video_duration
                    
                # Update UI in main thread
                self.panel.after(0, self._update_playback_ui)
                
            last_update = current
            time.sleep(0.033)  # ~30 FPS update rate
        
        # Auto-pause at end
        if self.current_time >= self.video_duration:
            self.panel.after(0, self.pause_playback)
    
    def _update_playback_ui(self):
        """Update playback UI elements"""
        try:
            # Prevent recursive calls by temporarily disabling command
            old_command = self.timeline_slider.cget('command')
            self.timeline_slider.configure(command=None)
            self.timeline_slider.set(self.current_time)
            self.timeline_slider.configure(command=old_command)
        except:
            # Fallback if command manipulation fails
            self.timeline_slider.set(self.current_time)
        
        self.update_time_display()
        
        # Update video frame if canvas exists (but not too frequently to avoid performance issues)
        if self.video_canvas and self.is_playing and hasattr(self, '_last_frame_update'):
            if time.time() - self._last_frame_update > 0.1:  # Update frame max 10 FPS
                self.load_video_frame(self.current_time)
                self._last_frame_update = time.time()
        elif self.video_canvas and self.is_playing:
            self.load_video_frame(self.current_time)
            self._last_frame_update = time.time()
    
    def seek_video(self, time_value):
        """Seek to specific time in video"""
        try:
            new_time = float(time_value)
            self.current_time = max(0, min(new_time, self.video_duration))
            self.update_time_display()
            
            # Update video frame
            if self.video_canvas:
                self.load_video_frame(self.current_time)
                
        except (ValueError, TypeError):
            pass
    
    def update_time_display(self):
        """Update time display"""
        current_min = int(self.current_time // 60)
        current_sec = int(self.current_time % 60)
        total_min = int(self.video_duration // 60)
        total_sec = int(self.video_duration % 60)
        
        time_text = f"{current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}"
        self.time_label.configure(text=time_text)
    
    def show_processing(self):
        """Show processing overlay"""
        self.is_processing = True
        self.processing_overlay.place(relx=0.5, rely=0.5, anchor="center")
        self.status_label.configure(text="Processing effects...")
    
    def hide_processing(self):
        """Hide processing overlay"""
        self.is_processing = False
        self.processing_overlay.place_forget()
        self.status_label.configure(text="Preview ready")
    
    def update_preview(self, preview_path: str):
        """Update preview with processed video"""
        if preview_path and os.path.exists(preview_path):
            self.preview_video_path = preview_path
            filename = os.path.basename(preview_path)
            
            self.video_display.configure(
                text=f"📺\n\n{filename}\n\nEffects applied successfully\nPreview updated"
            )
            
            self.status_label.configure(text="Preview updated with effects")
            
            # Refresh the currently displayed frame to show effects
            if self.video_canvas:
                self.load_video_frame(self.current_time)
        else:
            messagebox.showerror("Preview Error", "Failed to generate preview")
    
    
    def update_export_progress(self, progress: float, status: str = ""):
        """Update export progress"""
        self.export_progress = progress
        
        # Update export button text to show progress
        if progress > 0:
            self.export_button.configure(text=f"{progress:.0f}%", state="disabled")
        
        # Update export dialog progress if it exists
        if self.export_dialog:
            self.export_dialog.update_progress(progress, status)
    
    def export_completed(self, success: bool = True):
        """Handle export completion"""
        # Reset export button
        self.export_button.configure(text="🎬 Render", state="normal")
        
        # Update export dialog if it exists
        if self.export_dialog:
            self.export_dialog.export_completed(success)
    
    def get_export_settings(self) -> Dict[str, Any]:
        """Get current export settings"""
        if hasattr(self, 'current_export_settings'):
            return self.current_export_settings
        else:
            # Default settings if no export dialog has been used yet
            return {
                "quality": "1080p",
                "fps": "30fps", 
                "format": "MP4",
                "codec": "H.264",
                "output_path": "",
                "bitrate": "Auto",
                "audio": "AAC"
            }
    
