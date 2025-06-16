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
        
        # Export settings
        self.quality_var = tk.StringVar(value="1080p")
        self.fps_var = tk.StringVar(value="60fps")
        self.format_var = tk.StringVar(value="MP4")
        self.codec_var = tk.StringVar(value="H.264")
        
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
        
        # Export section
        self.create_export_section()
        
        # Presets section
        self.create_presets_section()
    
    def create_preview_area(self):
        """Create video preview display area"""
        # Preview container with fixed proportions
        preview_container = ctk.CTkFrame(self.panel)
        preview_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Configure preview container to maintain aspect ratio
        preview_container.grid_rowconfigure(0, weight=1)
        preview_container.grid_columnconfigure(0, weight=1)
        
        # Preview frame (16:9 aspect ratio maintained)
        self.preview_frame = ctk.CTkFrame(
            preview_container,
            fg_color="black",
            corner_radius=10,
            height=300  # Fixed height for better control
        )
        self.preview_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.preview_frame.grid_propagate(False)  # Maintain size
        
        # Video display canvas for actual video rendering
        try:
            import tkinter as tk
            self.video_canvas = tk.Canvas(
                self.preview_frame,
                bg="black",
                highlightthickness=0,
                width=400,
                height=225  # 16:9 aspect ratio
            )
            self.video_canvas.place(relx=0.5, rely=0.5, anchor="center")
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
        
        # Right side - volume control
        right_controls = ctk.CTkFrame(playback_frame)
        right_controls.pack(side="right", padx=10)
        
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
    
    def create_export_section(self):
        """Create export controls section"""
        export_container = ctk.CTkFrame(self.panel)
        export_container.pack(fill="x", padx=5, pady=5)
        
        # Export header
        export_header = ctk.CTkLabel(
            export_container,
            text="📊 EXPORT SETTINGS",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        export_header.pack(pady=5)
        
        # Export settings grid
        settings_frame = ctk.CTkFrame(export_container)
        settings_frame.pack(fill="x", padx=10, pady=5)
        
        # Quality setting
        quality_frame = ctk.CTkFrame(settings_frame)
        quality_frame.pack(side="left", padx=5, fill="x", expand=True)
        
        ctk.CTkLabel(quality_frame, text="Quality:", font=ctk.CTkFont(size=11)).pack()
        
        quality_combo = ctk.CTkComboBox(
            quality_frame,
            variable=self.quality_var,
            values=["720p", "1080p", "1440p", "4K"],
            width=80
        )
        quality_combo.pack(pady=2)
        
        # FPS setting
        fps_frame = ctk.CTkFrame(settings_frame)
        fps_frame.pack(side="left", padx=5, fill="x", expand=True)
        
        ctk.CTkLabel(fps_frame, text="FPS:", font=ctk.CTkFont(size=11)).pack()
        
        fps_combo = ctk.CTkComboBox(
            fps_frame,
            variable=self.fps_var,
            values=["24fps", "30fps", "60fps"],
            width=80
        )
        fps_combo.pack(pady=2)
        
        # Format setting
        format_frame = ctk.CTkFrame(settings_frame)
        format_frame.pack(side="left", padx=5, fill="x", expand=True)
        
        ctk.CTkLabel(format_frame, text="Format:", font=ctk.CTkFont(size=11)).pack()
        
        format_combo = ctk.CTkComboBox(
            format_frame,
            variable=self.format_var,
            values=["MP4", "AVI", "MOV"],
            width=80
        )
        format_combo.pack(pady=2)
        
        # Codec setting
        codec_frame = ctk.CTkFrame(settings_frame)
        codec_frame.pack(side="left", padx=5, fill="x", expand=True)
        
        ctk.CTkLabel(codec_frame, text="Codec:", font=ctk.CTkFont(size=11)).pack()
        
        codec_combo = ctk.CTkComboBox(
            codec_frame,
            variable=self.codec_var,
            values=["H.264", "H.265", "VP9"],
            width=80
        )
        codec_combo.pack(pady=2)
        
        # Export button
        self.export_button = ctk.CTkButton(
            export_container,
            text="🎬 RENDER FINAL VIDEO",
            command=self.on_export,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.export_button.pack(pady=10)
        
        # Export progress
        self.export_progress_frame = ctk.CTkFrame(export_container)
        
        self.export_progress_bar = ctk.CTkProgressBar(self.export_progress_frame)
        self.export_progress_bar.pack(fill="x", padx=10, pady=5)
        
        self.export_progress_label = ctk.CTkLabel(
            self.export_progress_frame,
            text="Exporting... 0%",
            font=ctk.CTkFont(size=12)
        )
        self.export_progress_label.pack(pady=2)
        
        # Initially hidden
        self.export_progress_frame.pack_forget()
    
    def create_presets_section(self):
        """Create presets section"""
        presets_container = ctk.CTkFrame(self.panel)
        presets_container.pack(fill="x", padx=5, pady=5)
        
        presets_header = ctk.CTkLabel(
            presets_container,
            text="⚡ QUICK PRESETS",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        presets_header.pack(pady=5)
        
        # Preset buttons grid
        presets_grid = ctk.CTkFrame(presets_container)
        presets_grid.pack(fill="x", padx=10, pady=5)
        
        # Row 1
        row1 = ctk.CTkFrame(presets_grid)
        row1.pack(fill="x", pady=2)
        
        preset_action = ctk.CTkButton(
            row1,
            text="🥊 Action",
            command=lambda: self.apply_preset("action_scene"),
            width=80,
            height=30
        )
        preset_action.pack(side="left", padx=2, fill="x", expand=True)
        
        preset_power = ctk.CTkButton(
            row1,
            text="⚡ Power Up",
            command=lambda: self.apply_preset("power_up"),
            width=80,
            height=30
        )
        preset_power.pack(side="left", padx=2, fill="x", expand=True)
        
        # Row 2
        row2 = ctk.CTkFrame(presets_grid)
        row2.pack(fill="x", pady=2)
        
        preset_emotional = ctk.CTkButton(
            row2,
            text="💖 Emotional",
            command=lambda: self.apply_preset("emotional_moment"),
            width=80,
            height=30
        )
        preset_emotional.pack(side="left", padx=2, fill="x", expand=True)
        
        preset_speed = ctk.CTkButton(
            row2,
            text="💨 Speed",
            command=lambda: self.apply_preset("speed_boost"),
            width=80,
            height=30
        )
        preset_speed.pack(side="left", padx=2, fill="x", expand=True)
        
        # Row 3
        row3 = ctk.CTkFrame(presets_grid)
        row3.pack(fill="x", pady=2)
        
        preset_impact = ctk.CTkButton(
            row3,
            text="💥 Impact",
            command=lambda: self.apply_preset("impact_hit"),
            width=80,
            height=30
        )
        preset_impact.pack(side="left", padx=2, fill="x", expand=True)
        
        preset_custom = ctk.CTkButton(
            row3,
            text="⚙️ Custom",
            command=self.show_custom_preset_dialog,
            width=80,
            height=30
        )
        preset_custom.pack(side="left", padx=2, fill="x", expand=True)
    
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
                import cv2
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
            import cv2
            from PIL import Image, ImageTk
            
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
                
                # Resize to fit canvas
                canvas_width = self.video_canvas.winfo_width() or 400
                canvas_height = self.video_canvas.winfo_height() or 225
                
                image = Image.fromarray(frame_rgb)
                image = image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage and display
                photo = ImageTk.PhotoImage(image)
                self.video_canvas.delete("all")
                self.video_canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor="center")
                
                # Keep reference to prevent garbage collection
                self.video_canvas.photo = photo
                
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
    
    def show_export_progress(self):
        """Show export progress"""
        self.export_progress_frame.pack(fill="x", padx=10, pady=5)
        self.export_button.configure(state="disabled", text="Exporting...")
        self.export_progress = 0.0
        self.update_export_progress(0.0)
    
    def hide_export_progress(self):
        """Hide export progress"""
        self.export_progress_frame.pack_forget()
        self.export_button.configure(state="normal", text="🎬 RENDER FINAL VIDEO")
    
    def update_export_progress(self, progress: float):
        """Update export progress"""
        self.export_progress = progress
        self.export_progress_bar.set(progress / 100.0)
        self.export_progress_label.configure(text=f"Exporting... {progress:.1f}%")
    
    def get_export_settings(self) -> Dict[str, Any]:
        """Get current export settings"""
        return {
            "quality": self.quality_var.get(),
            "fps": self.fps_var.get(),
            "format": self.format_var.get(),
            "codec": self.codec_var.get()
        }
    
    def apply_preset(self, preset_name: str):
        """Apply a preset (to be connected to main app)"""
        if hasattr(self, 'on_preset_selected'):
            self.on_preset_selected(preset_name)
        else:
            messagebox.showinfo("Preset", f"Applying preset: {preset_name}")
    
    def show_custom_preset_dialog(self):
        """Show custom preset creation dialog"""
        messagebox.showinfo("Custom Preset", "Custom preset creation dialog will be implemented.")
    
    def set_preset_callback(self, callback: Callable[[str], None]):
        """Set callback for preset selection"""
        self.on_preset_selected = callback