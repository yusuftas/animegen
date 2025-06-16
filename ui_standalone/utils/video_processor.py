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
        
        # Initialize effect processors
        self.effect_processors = self._setup_effect_processors()
        
    def _setup_effect_processors(self):
        """Setup effect processing functions"""
        return {
            # Motion Effects
            "speed_ramp": self._apply_speed_ramp,
            "zoom_punch": self._apply_zoom_punch,
            "camera_shake": self._apply_camera_shake,
            
            # Anime Effects
            "speed_lines": self._apply_speed_lines,
            "impact_frame": self._apply_impact_frame,
            "energy_aura": self._apply_energy_aura,
            
            # Color Effects
            "color_grade": self._apply_color_grade,
            "chromatic_aberration": self._apply_chromatic_aberration,
            "bloom": self._apply_bloom,
            
            # Text Effects
            "animated_text": self._apply_animated_text,
            "sound_fx_text": self._apply_sound_fx_text,
            
            # Audio Sync Effects
            "beat_flash": self._apply_beat_flash,
            "beat_zoom": self._apply_beat_zoom,
            
            # Transition Effects
            "iris_transition": self._apply_iris_transition,
            "swipe_transition": self._apply_swipe_transition
        }
    
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
                    # Fallback to legacy effect processing
                    class_name = effect.__class__.__name__
                    
                    # Map class names to processor keys
                    effect_type_mapping = {
                        'SpeedRampEffect': 'speed_ramp',
                        'ZoomPunchEffect': 'zoom_punch', 
                        'CameraShakeEffect': 'camera_shake',
                        'SpeedLinesEffect': 'speed_lines',
                        'ImpactFrameEffect': 'impact_frame',
                        'EnergyAuraEffect': 'energy_aura',
                        'ColorGradeEffect': 'color_grade',
                        'ChromaticAberrationEffect': 'chromatic_aberration',
                        'BloomEffect': 'bloom',
                        'AnimatedTextEffect': 'animated_text',
                        'SoundEffectTextEffect': 'sound_fx_text',
                        'BeatFlashEffect': 'beat_flash',
                        'BeatZoomEffect': 'beat_zoom',
                        'IrisTransitionEffect': 'iris_transition',
                        'SwipeTransitionEffect': 'swipe_transition'
                    }
                    
                    effect_type = effect_type_mapping.get(class_name, class_name.replace('Effect', '').lower())
                    
                    # Apply effect if processor exists
                    if effect_type in self.effect_processors:
                        processed_clip = self.effect_processors[effect_type](processed_clip, effect)
                        print(f"Applied legacy effect: {effect.name} ({effect_type})")
                    else:
                        print(f"Warning: No processor for effect type: {effect_type} (class: {class_name})")
                
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
    
    # Effect processing methods (placeholders for now)
    
    def _apply_speed_ramp(self, clip, effect: BaseEffect):
        """Apply speed ramp effect"""
        try:
            from moviepy.video.fx import speedx
            speed_points_str = effect.get_parameter_value('speed_points') or "0,1.0;2,1.5;4,1.0"
            
            # Parse speed points: "0,1.0;2,1.5;4,1.0" -> [(0, 1.0), (2, 1.5), (4, 1.0)]
            speed_points = []
            for point in speed_points_str.split(';'):
                time_str, speed_str = point.split(',')
                speed_points.append((float(time_str), float(speed_str)))
            
            # Apply average speed change (simple implementation)
            avg_speed = sum(speed for _, speed in speed_points) / len(speed_points)
            print(f"  → Speed ramp: average speed {avg_speed:.2f}x")
            
            return clip.fx(speedx, avg_speed)
            
        except Exception as e:
            print(f"  → Speed ramp failed: {e}")
            return clip
    
    def _apply_zoom_punch(self, clip, effect: BaseEffect):
        """Apply zoom punch effect"""
        try:
            zoom_factor = effect.get_parameter_value('zoom_factor') or 1.5
            print(f"  → Zoom punch: {zoom_factor}x zoom")
            
            # Apply zoom by resizing the clip
            return clip.resize(zoom_factor)
            
        except Exception as e:
            print(f"  → Zoom punch failed: {e}")
            return clip
    
    def _apply_camera_shake(self, clip, effect: BaseEffect):
        """Apply camera shake effect"""
        try:
            import random
            intensity = effect.get_parameter_value('intensity') or 10
            print(f"  → Camera shake: intensity {intensity}")
            
            def shake_transform(get_frame, t):
                frame = get_frame(t)
                if cv2 is not None and np is not None:
                    # Apply random translation
                    shake_x = random.randint(-intensity, intensity)
                    shake_y = random.randint(-intensity, intensity)
                    
                    h, w = frame.shape[:2]
                    M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
                    shaken_frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                    return shaken_frame
                return frame
            
            return clip.fl(shake_transform)
            
        except Exception as e:
            print(f"  → Camera shake failed: {e}")
            return clip
    
    def _apply_speed_lines(self, clip, effect: BaseEffect):
        """Apply speed lines effect"""
        try:
            direction = effect.get_parameter_value('direction') or 'right'
            intensity = effect.get_parameter_value('intensity') or 0.8
            print(f"  → Speed lines: {direction} direction, intensity {intensity}")
            
            def add_speed_lines(get_frame, t):
                frame = get_frame(t)
                if cv2 is not None and np is not None:
                    h, w = frame.shape[:2]
                    overlay = np.zeros_like(frame)
                    
                    num_lines = int(20 * intensity)
                    
                    for i in range(num_lines):
                        if direction == "right":
                            y = np.random.randint(0, h)
                            thickness = np.random.randint(1, 4)
                            length = np.random.randint(w//4, w//2)
                            x_start = np.random.randint(0, w - length)
                            
                            cv2.line(overlay, (x_start, y), (x_start + length, y), 
                                    (255, 255, 255), thickness)
                        elif direction == "radial":
                            center_x, center_y = w//2, h//2
                            angle = np.random.uniform(0, 2*np.pi)
                            length = np.random.randint(50, min(w, h)//4)
                            
                            end_x = int(center_x + length * np.cos(angle))
                            end_y = int(center_y + length * np.sin(angle))
                            
                            cv2.line(overlay, (center_x, center_y), (end_x, end_y), 
                                    (255, 255, 255), 2)
                    
                    alpha = 0.3
                    return cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
                return frame
            
            return clip.fl(add_speed_lines)
            
        except Exception as e:
            print(f"  → Speed lines failed: {e}")
            return clip
    
    def _apply_impact_frame(self, clip, effect: BaseEffect):
        """Apply impact frame effect"""
        try:
            style = effect.get_parameter_value('style') or 'energy'
            print(f"  → Impact frame: {style} style")
            
            def create_impact_effect(get_frame, t):
                frame = get_frame(t)
                if cv2 is not None and np is not None:
                    if style == "manga":
                        # High contrast black and white
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                        impact_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
                        return impact_frame
                    elif style == "energy":
                        # Boost saturation and brightness
                        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2, 0, 255)  # Saturation
                        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)  # Brightness
                        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                return frame
            
            return clip.fl(create_impact_effect)
            
        except Exception as e:
            print(f"  → Impact frame failed: {e}")
            return clip
    
    def _apply_energy_aura(self, clip, effect: BaseEffect):
        """Apply energy aura effect"""
        try:
            intensity = effect.get_parameter_value('intensity') or 1.0
            pulse_rate = effect.get_parameter_value('pulse_rate') or 6.0
            print(f"  → Energy aura: intensity {intensity}, pulse rate {pulse_rate}")
            
            def add_energy_aura(get_frame, t):
                frame = get_frame(t)
                if cv2 is not None and np is not None:
                    # Create pulsing effect based on time
                    pulse = (np.sin(t * pulse_rate) + 1) / 2  # 0 to 1
                    current_intensity = intensity * pulse
                    
                    # Boost brightness and add glow
                    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + current_intensity * 0.5), 0, 255)
                    
                    energy_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                    
                    # Add outer glow effect
                    if current_intensity > 0.5:
                        kernel = np.ones((5, 5), np.uint8)
                        dilated = cv2.dilate(energy_frame, kernel, iterations=1)
                        energy_frame = cv2.addWeighted(energy_frame, 0.8, dilated, 0.2, 0)
                    
                    return energy_frame
                return frame
            
            return clip.fl(add_energy_aura)
            
        except Exception as e:
            print(f"  → Energy aura failed: {e}")
            return clip
    
    def _apply_color_grade(self, clip, effect: BaseEffect):
        """Apply color grading effect"""
        try:
            style = effect.get_parameter_value('style') or 'vibrant'
            intensity = effect.get_parameter_value('intensity') or 0.9
            print(f"  → Color grade: {style} style, intensity {intensity}")
            
            def apply_color_grading(get_frame, t):
                frame = get_frame(t)
                if cv2 is not None and np is not None:
                    if style == "vibrant":
                        # Boost saturation and contrast
                        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + intensity), 0, 255)  # Saturation
                        graded_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                        # Boost contrast
                        graded_frame = np.clip(graded_frame * (1 + intensity * 0.3), 0, 255).astype(np.uint8)
                        return graded_frame
                    
                    elif style == "sunset":
                        # Warm color temperature
                        frame_float = frame.astype(np.float32) / 255.0
                        warm_matrix = np.array([
                            [1.2, 0.1, 0.0],  # More red
                            [0.1, 1.0, 0.0],  # Normal green
                            [0.0, 0.0, 0.8]   # Less blue
                        ]) * intensity + np.eye(3) * (1 - intensity)
                        
                        for i in range(3):
                            frame_float[:,:,i] = np.clip(
                                frame_float[:,:,0] * warm_matrix[i,0] +
                                frame_float[:,:,1] * warm_matrix[i,1] +
                                frame_float[:,:,2] * warm_matrix[i,2], 0, 1
                            )
                        
                        return (frame_float * 255).astype(np.uint8)
                    
                    elif style == "dramatic":
                        # High contrast with slight desaturation
                        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 - intensity * 0.3), 0, 255)  # Less saturation
                        graded_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                        # High contrast
                        graded_frame = np.clip((graded_frame - 128) * (1 + intensity) + 128, 0, 255).astype(np.uint8)
                        return graded_frame
                        
                return frame
            
            return clip.fl(apply_color_grading)
            
        except Exception as e:
            print(f"  → Color grade failed: {e}")
            return clip
    
    def _apply_chromatic_aberration(self, clip, effect: BaseEffect):
        """Apply chromatic aberration effect"""
        # Placeholder
        return clip
    
    def _apply_bloom(self, clip, effect: BaseEffect):
        """Apply bloom effect"""
        # Placeholder
        return clip
    
    def _apply_animated_text(self, clip, effect: BaseEffect):
        """Apply animated text effect"""
        text = effect.get_parameter_value('text') or 'TEXT'
        fontsize = effect.get_parameter_value('fontsize') or 50
        
        if VideoFileClip:
            try:
                from moviepy.editor import TextClip, CompositeVideoClip
                
                # Create text clip
                text_clip = TextClip(
                    text,
                    fontsize=fontsize,
                    color='white',
                    stroke_color='black',
                    stroke_width=2
                ).set_duration(effect.duration or 2.0).set_position('center')
                
                if effect.start_time > 0:
                    text_clip = text_clip.set_start(effect.start_time)
                
                # Composite with main clip
                return CompositeVideoClip([clip, text_clip])
            except:
                pass
        return clip
    
    def _apply_sound_fx_text(self, clip, effect: BaseEffect):
        """Apply sound effect text"""
        # Similar to animated text but with different styling
        return self._apply_animated_text(clip, effect)
    
    def _apply_beat_flash(self, clip, effect: BaseEffect):
        """Apply beat flash effect"""
        # Placeholder - would sync with audio beats
        return clip
    
    def _apply_beat_zoom(self, clip, effect: BaseEffect):
        """Apply beat zoom effect"""
        # Placeholder - would sync zoom with audio beats
        return clip
    
    def _apply_iris_transition(self, clip, effect: BaseEffect):
        """Apply iris transition effect"""
        # Placeholder - would create circular transition
        return clip
    
    def _apply_swipe_transition(self, clip, effect: BaseEffect):
        """Apply swipe transition effect"""
        # Placeholder - would create directional wipe
        return clip
    
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