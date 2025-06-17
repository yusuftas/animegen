"""
Export Dialog - Separate dialog for render/export settings
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from typing import Callable, Optional, Dict, Any


class ExportDialog:
    """Dialog for export/render settings and progress"""
    
    def __init__(self, parent, on_export: Callable[[Dict[str, Any]], None]):
        self.parent = parent
        self.on_export = on_export
        self.dialog = None
        
        # Export settings variables
        self.quality_var = tk.StringVar(value="1080p")
        self.fps_var = tk.StringVar(value="30fps")
        self.format_var = tk.StringVar(value="MP4")
        self.codec_var = tk.StringVar(value="H.264")
        self.output_path_var = tk.StringVar(value="")
        
        # Progress tracking
        self.is_exporting = False
        self.export_progress = 0.0
    
    def show_dialog(self):
        """Show the export dialog"""
        if self.dialog is not None:
            self.dialog.lift()
            return
            
        # Create dialog window
        self.dialog = ctk.CTkToplevel(self.parent)
        self.dialog.title("Export Video - Render Settings")
        self.dialog.geometry("600x500")
        self.dialog.resizable(False, False)
        
        # Center the dialog
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Build dialog content
        self.create_dialog_content()
        
        # Handle dialog close
        self.dialog.protocol("WM_DELETE_WINDOW", self.close_dialog)
    
    def create_dialog_content(self):
        """Create the dialog content"""
        # Main container
        main_frame = ctk.CTkFrame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="🎬 Export Video Settings",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(pady=(10, 20))
        
        # Output file section
        self.create_output_file_section(main_frame)
        
        # Quality settings section
        self.create_quality_settings_section(main_frame)
        
        # Advanced settings section
        self.create_advanced_settings_section(main_frame)
        
        # Progress section
        self.create_progress_section(main_frame)
        
        # Buttons section
        self.create_buttons_section(main_frame)
    
    def create_output_file_section(self, parent):
        """Create output file selection section"""
        output_frame = ctk.CTkFrame(parent)
        output_frame.pack(fill="x", padx=10, pady=10)
        
        # Section title
        output_title = ctk.CTkLabel(
            output_frame,
            text="📁 Output File",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        output_title.pack(pady=(10, 5))
        
        # File path selection
        path_frame = ctk.CTkFrame(output_frame)
        path_frame.pack(fill="x", padx=10, pady=5)
        
        self.output_path_entry = ctk.CTkEntry(
            path_frame,
            textvariable=self.output_path_var,
            placeholder_text="Choose output file location..."
        )
        self.output_path_entry.pack(side="left", fill="x", expand=True, padx=(5, 2))
        
        browse_button = ctk.CTkButton(
            path_frame,
            text="Browse",
            command=self.browse_output_file,
            width=80
        )
        browse_button.pack(side="right", padx=(2, 5))
    
    def create_quality_settings_section(self, parent):
        """Create quality settings section"""
        quality_frame = ctk.CTkFrame(parent)
        quality_frame.pack(fill="x", padx=10, pady=10)
        
        # Section title
        quality_title = ctk.CTkLabel(
            quality_frame,
            text="🎯 Quality Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        quality_title.pack(pady=(10, 5))
        
        # Settings grid
        settings_grid = ctk.CTkFrame(quality_frame)
        settings_grid.pack(fill="x", padx=10, pady=5)
        
        # Configure grid
        settings_grid.grid_columnconfigure(0, weight=1)
        settings_grid.grid_columnconfigure(1, weight=1)
        settings_grid.grid_columnconfigure(2, weight=1)
        settings_grid.grid_columnconfigure(3, weight=1)
        
        # Resolution
        res_frame = ctk.CTkFrame(settings_grid)
        res_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        ctk.CTkLabel(res_frame, text="Resolution", font=ctk.CTkFont(size=11, weight="bold")).pack(pady=2)
        quality_combo = ctk.CTkComboBox(
            res_frame,
            variable=self.quality_var,
            values=["720p", "1080p", "1440p", "4K"],
            width=120
        )
        quality_combo.pack(pady=2)
        
        # FPS
        fps_frame = ctk.CTkFrame(settings_grid)
        fps_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ctk.CTkLabel(fps_frame, text="Frame Rate", font=ctk.CTkFont(size=11, weight="bold")).pack(pady=2)
        fps_combo = ctk.CTkComboBox(
            fps_frame,
            variable=self.fps_var,
            values=["24fps", "30fps", "60fps"],
            width=120
        )
        fps_combo.pack(pady=2)
        
        # Format
        format_frame = ctk.CTkFrame(settings_grid)
        format_frame.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        ctk.CTkLabel(format_frame, text="Format", font=ctk.CTkFont(size=11, weight="bold")).pack(pady=2)
        format_combo = ctk.CTkComboBox(
            format_frame,
            variable=self.format_var,
            values=["MP4", "AVI", "MOV", "MKV"],
            width=120
        )
        format_combo.pack(pady=2)
        
        # Codec
        codec_frame = ctk.CTkFrame(settings_grid)
        codec_frame.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        
        ctk.CTkLabel(codec_frame, text="Codec", font=ctk.CTkFont(size=11, weight="bold")).pack(pady=2)
        codec_combo = ctk.CTkComboBox(
            codec_frame,
            variable=self.codec_var,
            values=["H.264", "H.265", "VP9", "AV1"],
            width=120
        )
        codec_combo.pack(pady=2)
    
    def create_advanced_settings_section(self, parent):
        """Create advanced settings section"""
        advanced_frame = ctk.CTkFrame(parent)
        advanced_frame.pack(fill="x", padx=10, pady=10)
        
        # Section title
        advanced_title = ctk.CTkLabel(
            advanced_frame,
            text="⚙️ Advanced Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        advanced_title.pack(pady=(10, 5))
        
        # Settings container
        advanced_settings = ctk.CTkFrame(advanced_frame)
        advanced_settings.pack(fill="x", padx=10, pady=5)
        
        # Bitrate
        bitrate_frame = ctk.CTkFrame(advanced_settings)
        bitrate_frame.pack(side="left", fill="x", expand=True, padx=5)
        
        ctk.CTkLabel(bitrate_frame, text="Bitrate", font=ctk.CTkFont(size=11)).pack(pady=2)
        self.bitrate_var = tk.StringVar(value="Auto")
        bitrate_combo = ctk.CTkComboBox(
            bitrate_frame,
            variable=self.bitrate_var,
            values=["Auto", "5 Mbps", "10 Mbps", "20 Mbps", "50 Mbps"],
            width=100
        )
        bitrate_combo.pack(pady=2)
        
        # Audio settings
        audio_frame = ctk.CTkFrame(advanced_settings)
        audio_frame.pack(side="left", fill="x", expand=True, padx=5)
        
        ctk.CTkLabel(audio_frame, text="Audio", font=ctk.CTkFont(size=11)).pack(pady=2)
        self.audio_var = tk.StringVar(value="AAC")
        audio_combo = ctk.CTkComboBox(
            audio_frame,
            variable=self.audio_var,
            values=["AAC", "MP3", "No Audio"],
            width=100
        )
        audio_combo.pack(pady=2)
    
    def create_progress_section(self, parent):
        """Create progress display section"""
        self.progress_frame = ctk.CTkFrame(parent)
        self.progress_frame.pack(fill="x", padx=10, pady=10)
        
        # Progress title
        self.progress_title = ctk.CTkLabel(
            self.progress_frame,
            text="📊 Export Progress",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.progress_title.pack(pady=(10, 5))
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0)
        
        # Progress label
        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="Ready to export",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.progress_label.pack(pady=2)
        
        # Initially hide progress section
        self.progress_frame.pack_forget()
    
    def create_buttons_section(self, parent):
        """Create dialog buttons"""
        buttons_frame = ctk.CTkFrame(parent)
        buttons_frame.pack(fill="x", padx=10, pady=(10, 20))
        
        # Cancel button
        cancel_button = ctk.CTkButton(
            buttons_frame,
            text="Cancel",
            command=self.close_dialog,
            width=120,
            fg_color="gray"
        )
        cancel_button.pack(side="left", padx=10)
        
        # Export button
        self.export_button = ctk.CTkButton(
            buttons_frame,
            text="🎬 Start Export",
            command=self.start_export,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="green"
        )
        self.export_button.pack(side="right", padx=10)
    
    def browse_output_file(self):
        """Browse for output file location"""
        file_types = [
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("MOV files", "*.mov"),
            ("MKV files", "*.mkv"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Save Export As",
            filetypes=file_types,
            defaultextension=".mp4"
        )
        
        if filename:
            self.output_path_var.set(filename)
    
    def start_export(self):
        """Start the export process"""
        # Validate settings
        if not self.output_path_var.get():
            messagebox.showerror("Error", "Please select an output file location.")
            return
        
        # Gather export settings
        export_settings = {
            'output_path': self.output_path_var.get(),
            'quality': self.quality_var.get(),
            'fps': self.fps_var.get(),
            'format': self.format_var.get(),
            'codec': self.codec_var.get(),
            'bitrate': self.bitrate_var.get(),
            'audio': self.audio_var.get()
        }
        
        # Show progress section
        self.progress_frame.pack(fill="x", padx=10, pady=10)
        
        # Update UI state
        self.is_exporting = True
        self.export_button.configure(text="Exporting...", state="disabled")
        self.progress_label.configure(text="Starting export...")
        
        # Start export
        self.on_export(export_settings)
    
    def update_progress(self, progress: float, status: str = ""):
        """Update export progress"""
        self.export_progress = progress
        
        if self.dialog and self.progress_bar:
            self.progress_bar.set(progress / 100.0)
            
            if status:
                self.progress_label.configure(text=status)
            else:
                self.progress_label.configure(text=f"Exporting... {progress:.1f}%")
    
    def export_completed(self, success: bool = True):
        """Handle export completion"""
        self.is_exporting = False
        
        if self.dialog:
            if success:
                self.progress_label.configure(text="Export completed successfully!")
                self.export_button.configure(text="Export Complete ✓", fg_color="green")
                
                # Auto-close after 2 seconds
                self.dialog.after(2000, self.close_dialog)
            else:
                self.progress_label.configure(text="Export failed!")
                self.export_button.configure(text="🎬 Start Export", state="normal", fg_color="green")
    
    def close_dialog(self):
        """Close the export dialog"""
        if self.is_exporting:
            if messagebox.askyesno("Export in Progress", "Export is currently running. Do you want to cancel it?"):
                # Here you would cancel the export process
                pass
            else:
                return
        
        if self.dialog:
            self.dialog.grab_release()
            self.dialog.destroy()
            self.dialog = None


def show_export_dialog(parent, on_export: Callable[[Dict[str, Any]], None]) -> ExportDialog:
    """Convenience function to show export dialog"""
    dialog = ExportDialog(parent, on_export)
    dialog.show_dialog()
    return dialog