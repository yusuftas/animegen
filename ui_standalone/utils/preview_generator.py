"""
Preview Generator - Handles real-time preview generation for UI
"""

import threading
import queue
import time
from typing import Callable, Optional, Dict, Any
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ui_standalone.models.effect_pipeline import EffectPipeline
from ui_standalone.utils.video_processor import VideoProcessor

class PreviewGenerator:
    """Manages real-time preview generation with caching and threading"""
    
    def __init__(self):
        self.video_processor = VideoProcessor()
        
        # Preview generation queue
        self.preview_queue = queue.Queue()
        self.preview_thread = None
        self.is_running = False
        
        # Cache management
        self.preview_cache = {}
        self.cache_max_size = 10
        
        # Current preview state
        self.current_video_path = None
        self.current_pipeline_hash = None
        self.preview_callbacks = []
        
        # Performance settings
        self.preview_resolution = (854, 480)  # 480p for fast preview
        self.preview_fps = 15  # Lower FPS for performance
        
    def start(self):
        """Start the preview generation thread"""
        if not self.is_running:
            self.is_running = True
            self.preview_thread = threading.Thread(target=self._preview_worker, daemon=True)
            self.preview_thread.start()
    
    def stop(self):
        """Stop the preview generation thread"""
        self.is_running = False
        if self.preview_thread:
            self.preview_thread.join(timeout=1.0)
    
    def generate_preview(self, video_path: str, pipeline: EffectPipeline, 
                        callback: Callable[[Optional[str]], None]):
        """Request preview generation"""
        # Generate pipeline hash for caching
        pipeline_hash = self._generate_pipeline_hash(pipeline)
        
        # Check cache first
        cache_key = f"{video_path}_{pipeline_hash}"
        if cache_key in self.preview_cache:
            callback(self.preview_cache[cache_key])
            return
        
        # Queue preview generation
        preview_request = {
            'video_path': video_path,
            'pipeline': pipeline,
            'pipeline_hash': pipeline_hash,
            'callback': callback,
            'timestamp': time.time()
        }
        
        # Clear old requests for same video
        self._clear_old_requests(video_path)
        
        # Add new request
        self.preview_queue.put(preview_request)
        
        # Start worker if not running
        if not self.is_running:
            self.start()
    
    def _preview_worker(self):
        """Background worker for preview generation"""
        while self.is_running:
            try:
                # Get request from queue (with timeout)
                request = self.preview_queue.get(timeout=1.0)
                
                # Process the request
                self._process_preview_request(request)
                
                # Mark task as done
                self.preview_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in preview worker: {e}")
    
    def _process_preview_request(self, request: Dict[str, Any]):
        """Process a single preview request"""
        video_path = request['video_path']
        pipeline = request['pipeline']
        pipeline_hash = request['pipeline_hash']
        callback = request['callback']
        
        try:
            # Generate cache key
            cache_key = f"{video_path}_{pipeline_hash}"
            
            # Check if already cached (race condition protection)
            if cache_key in self.preview_cache:
                callback(self.preview_cache[cache_key])
                return
            
            # Generate preview
            preview_path = self.video_processor.generate_preview(video_path, pipeline)
            
            # Cache the result
            if preview_path:
                self._add_to_cache(cache_key, preview_path)
            
            # Call callback with result
            callback(preview_path)
            
        except Exception as e:
            print(f"Error processing preview request: {e}")
            callback(None)
    
    def _generate_pipeline_hash(self, pipeline: EffectPipeline) -> str:
        """Generate a hash for the pipeline state"""
        # Simple hash based on enabled effects and their parameters
        hash_components = []
        
        for effect in pipeline.get_enabled_effects():
            effect_data = f"{effect.__class__.__name__}_{effect.start_time}_{effect.duration}"
            
            # Add parameter values
            for param_name, param in effect.parameters.items():
                effect_data += f"_{param_name}_{param.value}"
            
            hash_components.append(effect_data)
        
        # Create hash
        import hashlib
        hash_string = "|".join(sorted(hash_components))
        return hashlib.md5(hash_string.encode()).hexdigest()[:12]
    
    def _clear_old_requests(self, video_path: str):
        """Clear old requests for the same video"""
        # Create new queue with non-matching requests
        new_queue = queue.Queue()
        
        while not self.preview_queue.empty():
            try:
                request = self.preview_queue.get_nowait()
                if request['video_path'] != video_path:
                    new_queue.put(request)
            except queue.Empty:
                break
        
        # Replace the queue
        self.preview_queue = new_queue
    
    def _add_to_cache(self, cache_key: str, preview_path: str):
        """Add preview to cache with size management"""
        # Remove oldest entries if cache is full
        if len(self.preview_cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.preview_cache))
            old_path = self.preview_cache.pop(oldest_key)
            
            # Clean up old preview file
            try:
                import os
                if os.path.exists(old_path):
                    os.remove(old_path)
            except Exception as e:
                print(f"Error removing old preview: {e}")
        
        # Add new entry
        self.preview_cache[cache_key] = preview_path
    
    def invalidate_cache(self, video_path: Optional[str] = None):
        """Invalidate preview cache"""
        if video_path:
            # Remove entries for specific video
            keys_to_remove = [key for key in self.preview_cache.keys() 
                             if key.startswith(video_path)]
            for key in keys_to_remove:
                self.preview_cache.pop(key, None)
        else:
            # Clear entire cache
            self.preview_cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.preview_cache),
            'max_cache_size': self.cache_max_size,
            'queue_size': self.preview_queue.qsize(),
            'is_running': self.is_running
        }
    
    def set_preview_quality(self, resolution: tuple, fps: int):
        """Set preview quality settings"""
        self.preview_resolution = resolution
        self.preview_fps = fps
    
    def cleanup(self):
        """Clean up resources"""
        self.stop()
        
        # Clean up cached previews
        for preview_path in self.preview_cache.values():
            try:
                import os
                if os.path.exists(preview_path):
                    os.remove(preview_path)
            except Exception as e:
                print(f"Error cleaning up preview: {e}")
        
        self.preview_cache.clear()
        
        # Clean up video processor
        self.video_processor.cleanup()
    
    def __del__(self):
        """Destructor to clean up resources"""
        self.cleanup()


class RealTimePreviewManager:
    """Manages real-time preview updates for UI components"""
    
    def __init__(self):
        self.preview_generator = PreviewGenerator()
        self.current_video_path = None
        self.current_pipeline = None
        self.preview_callbacks = []
        
        # Debouncing for rapid changes
        self.update_timer = None
        self.update_delay = 0.5  # 500ms delay before generating preview
        
    def set_video(self, video_path: str):
        """Set the current video for preview"""
        self.current_video_path = video_path
        self.preview_generator.invalidate_cache(video_path)
    
    def update_pipeline(self, pipeline: EffectPipeline):
        """Update the effects pipeline and trigger preview update"""
        self.current_pipeline = pipeline
        
        # Cancel existing timer
        if self.update_timer:
            self.update_timer.cancel()
        
        # Start new timer for debounced update
        self.update_timer = threading.Timer(
            self.update_delay, 
            self._generate_preview_debounced
        )
        self.update_timer.start()
    
    def _generate_preview_debounced(self):
        """Generate preview after debounce delay"""
        if self.current_video_path and self.current_pipeline:
            self.preview_generator.generate_preview(
                self.current_video_path,
                self.current_pipeline,
                self._on_preview_ready
            )
    
    def _on_preview_ready(self, preview_path: Optional[str]):
        """Handle preview generation completion"""
        # Notify all registered callbacks
        for callback in self.preview_callbacks:
            try:
                callback(preview_path)
            except Exception as e:
                print(f"Error in preview callback: {e}")
    
    def add_preview_callback(self, callback: Callable[[Optional[str]], None]):
        """Add callback for preview updates"""
        self.preview_callbacks.append(callback)
    
    def remove_preview_callback(self, callback: Callable[[Optional[str]], None]):
        """Remove preview callback"""
        if callback in self.preview_callbacks:
            self.preview_callbacks.remove(callback)
    
    def force_update(self):
        """Force immediate preview update"""
        if self.update_timer:
            self.update_timer.cancel()
        self._generate_preview_debounced()
    
    def set_update_delay(self, delay: float):
        """Set the debounce delay for preview updates"""
        self.update_delay = delay
    
    def get_status(self) -> Dict[str, Any]:
        """Get preview manager status"""
        return {
            'has_video': self.current_video_path is not None,
            'has_pipeline': self.current_pipeline is not None,
            'callback_count': len(self.preview_callbacks),
            'cache_info': self.preview_generator.get_cache_info(),
            'update_delay': self.update_delay
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.update_timer:
            self.update_timer.cancel()
        
        self.preview_generator.cleanup()
        self.preview_callbacks.clear()
    
    def __del__(self):
        """Destructor to clean up resources"""
        self.cleanup()