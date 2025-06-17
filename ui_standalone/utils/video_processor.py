"""
Video Processor - Handles video processing and effects application
"""

import os
import tempfile
import threading
from typing import Callable, Optional, Dict, Any, List
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from moviepy.editor import VideoFileClip, CompositeVideoClip
    import cv2
    import numpy as np
except ImportError:
    print("Warning: MoviePy or OpenCV not available. Video processing will be limited.")
    VideoFileClip = None
    CompositeVideoClip = None
    cv2 = None
    np = None

from ui_standalone.models.effect_pipeline import EffectPipeline
from ui_standalone.models.effect_models import BaseEffect, EffectCategory
from ui_standalone.models.effect_adapter import ProductionEffectFactory, ProductionEffect

class VideoProcessor:
    """Handles video processing with effects pipeline"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="anime_effects_")
        self.processing_lock = threading.Lock()
        
        # Cache for processed clips
        self.clip_cache = {}
        
        
    
    def generate_preview(self, video_path: str, pipeline: EffectPipeline) -> Optional[str]:
        """Generate preview video with effects applied"""
        if not VideoFileClip:
            print("⚠️ MoviePy not available, returning original video")
            return video_path
            
        try:
            with self.processing_lock:
                print(f"🎬 Loading video: {video_path}")
                # Load video
                clip = VideoFileClip(video_path)
                
                print(f"📐 Video info: {clip.duration:.1f}s, {clip.fps}fps, {clip.size}")
                
                # Apply effects
                processed_clip = self._apply_effects_pipeline(clip, pipeline)
                
                # Generate preview (lower quality for speed)
                preview_filename = f"preview_{os.path.basename(video_path)}"
                preview_path = os.path.join(self.temp_dir, preview_filename)
                
                print(f"💾 Exporting preview to: {preview_path}")
                
                # Export preview with lower quality and no audio for speed
                processed_clip.write_videofile(
                    preview_path,
                    codec='libx264',
                    audio=False,  # Skip audio for faster preview
                    temp_audiofile=None,
                    remove_temp=True,
                    preset='ultrafast',
                    ffmpeg_params=['-crf', '28'],  # Lower quality for speed
                    verbose=False,
                    logger=None  # Suppress moviepy logs
                )
                
                # Clean up
                clip.close()
                processed_clip.close()
                
                print(f"✅ Preview generated successfully")
                return preview_path
                
        except Exception as e:
            print(f"❌ Error generating preview: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def export_video(self, video_path: str, pipeline: EffectPipeline, output_path: str, 
                     settings: Dict[str, Any], progress_callback: Optional[Callable[[float], None]] = None):
        """Export final video with effects applied"""
        if not VideoFileClip:
            raise RuntimeError("MoviePy not available for video processing")
            
        try:
            with self.processing_lock:
                # Load video
                clip = VideoFileClip(video_path)
                
                # Apply effects
                processed_clip = self._apply_effects_pipeline(clip, pipeline, progress_callback)
                
                # Get export parameters
                codec_params = self._get_export_params(settings)
                
                # Custom progress tracking if callback provided
                if progress_callback:
                    # Start export in thread and simulate progress
                    import threading
                    
                    def export_with_progress():
                        try:
                            # Prepare export parameters, filtering out unsupported ones
                            export_params = {
                                'codec': codec_params['video_codec'],
                                'preset': codec_params['preset'],
                                'verbose': False,
                                'logger': None
                            }
                            
                            # Add audio codec if the clip has audio
                            if processed_clip.audio is not None:
                                export_params['audio_codec'] = codec_params['audio_codec']
                            
                            # Add ffmpeg params if specified
                            if codec_params['ffmpeg_params']:
                                export_params['ffmpeg_params'] = codec_params['ffmpeg_params']
                            
                            processed_clip.write_videofile(output_path, **export_params)
                        except Exception as e:
                            print(f"Export error: {e}")
                            raise
                    
                    # Start export
                    export_thread = threading.Thread(target=export_with_progress)
                    export_thread.start()
                    
                    # Simulate progress while export runs
                    progress = 0
                    while export_thread.is_alive() and progress < 95:
                        import time
                        time.sleep(0.5)
                        progress += 5
                        progress_callback(min(progress, 95))
                    
                    # Wait for completion
                    export_thread.join()
                    progress_callback(100.0)
                    
                else:
                    # Simple export without progress
                    export_params = {
                        'codec': codec_params['video_codec'],
                        'preset': codec_params['preset'],
                        'verbose': False,
                        'logger': None
                    }
                    
                    # Add audio codec if the clip has audio
                    if processed_clip.audio is not None:
                        export_params['audio_codec'] = codec_params['audio_codec']
                    
                    # Add ffmpeg params if specified
                    if codec_params['ffmpeg_params']:
                        export_params['ffmpeg_params'] = codec_params['ffmpeg_params']
                    
                    processed_clip.write_videofile(output_path, **export_params)
                
                # Clean up
                clip.close()
                processed_clip.close()
                    
        except Exception as e:
            print(f"Error exporting video: {e}")
            raise
    
    def _apply_effects_pipeline(self, clip, pipeline: EffectPipeline, 
                               progress_callback: Optional[Callable[[float], None]] = None):
        """Apply all effects in the pipeline to the clip"""
        processed_clip = clip
        enabled_effects = pipeline.get_enabled_effects()
        
        for i, effect in enumerate(enabled_effects):
            try:
                # Check if this is a production effect (new adapter system)
                if isinstance(effect, ProductionEffect):
                    # Use production engines directly
                    processed_clip = ProductionEffectFactory.apply_effect_to_clip(processed_clip, effect)
                    print(f"Applied production effect: {effect.name}")
                else:
                    # Legacy effect fallback (minimal support)
                    print(f"Warning: Legacy effect not supported: {effect.name}")
                
                # Update progress
                if progress_callback:
                    progress = ((i + 1) / len(enabled_effects)) * 90  # 90% for effects, 10% for export
                    progress_callback(progress)
                    
            except Exception as e:
                print(f"Error applying effect {effect.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return processed_clip
    
    def _get_export_params(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Get export parameters from settings"""
        quality = settings.get('quality', '1080p')
        fps = settings.get('fps', '60fps')
        format_type = settings.get('format', 'MP4')
        codec = settings.get('codec', 'H.264')
        
        # Map settings to codec parameters
        codec_map = {
            'H.264': 'libx264',
            'H.265': 'libx265',
            'VP9': 'libvpx-vp9'
        }
        
        quality_map = {
            '720p': ['-crf', '23'],
            '1080p': ['-crf', '20'],
            '1440p': ['-crf', '18'],
            '4K': ['-crf', '16']
        }
        
        return {
            'video_codec': codec_map.get(codec, 'libx264'),
            'audio_codec': 'aac',
            'preset': 'medium',
            'ffmpeg_params': quality_map.get(quality, ['-crf', '20'])
        }
    
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information"""
        if not VideoFileClip or not os.path.exists(video_path):
            return {
                'duration': 0.0,
                'fps': 30,
                'resolution': (1920, 1080),
                'has_audio': False
            }
        
        try:
            clip = VideoFileClip(video_path)
            info = {
                'duration': clip.duration,
                'fps': clip.fps,
                'resolution': (clip.w, clip.h),
                'has_audio': clip.audio is not None
            }
            clip.close()
            return info
        except Exception as e:
            print(f"Error getting video info: {e}")
            return {
                'duration': 0.0,
                'fps': 30,
                'resolution': (1920, 1080),
                'has_audio': False
            }
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")
    
    def __del__(self):
        """Destructor to clean up resources"""
        self.cleanup()