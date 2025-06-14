"""
Video Clip Extraction Module
Extracts video clips from scenes based on JSON metadata
"""

import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class ClipExtractionResult:
    clip_path: str
    scene_info: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    file_size_mb: Optional[float] = None

class ClipExtractor:
    """Extracts video clips from scenes using FFmpeg"""
    
    def __init__(self, output_quality: str = "high"):
        self.logger = logging.getLogger(__name__)
        self.output_quality = output_quality
        
        # Quality presets
        self.quality_presets = {
            "high": {
                "video_codec": "libx264",
                "audio_codec": "aac", 
                "crf": "18",
                "preset": "medium",
                "bitrate": None
            },
            "medium": {
                "video_codec": "libx264",
                "audio_codec": "aac",
                "crf": "23", 
                "preset": "medium",
                "bitrate": None
            },
            "fast": {
                "video_codec": "libx264",
                "audio_codec": "aac",
                "crf": "28",
                "preset": "fast",
                "bitrate": None
            },
            "youtube_shorts": {
                "video_codec": "libx264",
                "audio_codec": "aac",
                "crf": "20",
                "preset": "medium", 
                "scale": "1080:1920",  # 9:16 aspect ratio
                "bitrate": "4000k"
            }
        }
        
        # Check FFmpeg availability
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg not found. Please install FFmpeg to extract clips.")
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _format_time(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS.mmm format for FFmpeg"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def extract_clip(self, source_video: str, scene_data: Dict[str, Any], 
                    output_path: str, custom_name: Optional[str] = None) -> ClipExtractionResult:
        """
        Extract a single clip from scene data
        
        Args:
            source_video: Path to source video file
            scene_data: Scene information from JSON
            output_path: Directory to save the clip
            custom_name: Custom filename (optional)
            
        Returns:
            ClipExtractionResult with extraction details
        """
        try:
            # Extract scene info
            scene_info = scene_data.get('scene_info', {})
            start_time = scene_info.get('start_time', 0)
            end_time = scene_info.get('end_time', 0)
            duration = scene_info.get('duration', end_time - start_time)
            
            if duration <= 0:
                return ClipExtractionResult(
                    clip_path="",
                    scene_info=scene_info,
                    success=False,
                    error_message="Invalid scene duration"
                )
            
            # Generate output filename
            if custom_name:
                filename = f"{custom_name}.mp4"
            else:
                # Create descriptive filename
                classification = scene_data.get('classification', {})
                scene_type = classification.get('primary_type', 'scene')
                score = scene_data.get('interest_score', {}).get('total', 0)
                
                filename = f"{scene_type}_{start_time:.1f}s_score_{score:.3f}.mp4"
            
            output_file = Path(output_path) / filename
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Build FFmpeg command
            cmd = self._build_ffmpeg_command(source_video, start_time, duration, str(output_file))
            
            # Execute FFmpeg
            self.logger.info(f"Extracting clip: {start_time:.1f}s-{end_time:.1f}s ({duration:.1f}s)")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and output_file.exists():
                file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                self.logger.info(f"Successfully extracted: {filename} ({file_size:.1f} MB)")
                
                return ClipExtractionResult(
                    clip_path=str(output_file),
                    scene_info=scene_info,
                    success=True,
                    file_size_mb=file_size
                )
            else:
                error_msg = result.stderr if result.stderr else "Unknown FFmpeg error"
                self.logger.error(f"FFmpeg extraction failed: {error_msg}")
                
                return ClipExtractionResult(
                    clip_path="",
                    scene_info=scene_info,
                    success=False,
                    error_message=error_msg
                )
                
        except subprocess.TimeoutExpired:
            error_msg = f"FFmpeg timeout extracting scene {start_time:.1f}s-{end_time:.1f}s"
            self.logger.error(error_msg)
            return ClipExtractionResult(
                clip_path="",
                scene_info=scene_info,
                success=False,
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"Clip extraction error: {str(e)}"
            self.logger.error(error_msg)
            return ClipExtractionResult(
                clip_path="",
                scene_info=scene_info,
                success=False,
                error_message=error_msg
            )
    
    def _build_ffmpeg_command(self, source_video: str, start_time: float, 
                             duration: float, output_file: str) -> List[str]:
        """Build FFmpeg command based on quality preset"""
        preset = self.quality_presets[self.output_quality]
        
        cmd = [
            'ffmpeg',
            '-i', source_video,
            '-ss', self._format_time(start_time),
            '-t', self._format_time(duration),
            '-c:v', preset['video_codec'],
            '-c:a', preset['audio_codec'],
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite output file
        ]
        
        # Add quality settings
        if preset.get('crf'):
            cmd.extend(['-crf', preset['crf']])
        if preset.get('preset'):
            cmd.extend(['-preset', preset['preset']])
        if preset.get('bitrate'):
            cmd.extend(['-b:v', preset['bitrate']])
        if preset.get('scale'):
            cmd.extend(['-vf', f'scale={preset["scale"]}'])
        
        cmd.append(output_file)
        return cmd
    
    def extract_clips_from_json_files(self, json_files: List[str], source_video: str,
                                     output_dir: str = "./extracted_clips") -> List[ClipExtractionResult]:
        """
        Extract clips from multiple JSON files
        
        Args:
            json_files: List of JSON file paths containing scene data
            source_video: Path to source video file
            output_dir: Directory to save extracted clips
            
        Returns:
            List of ClipExtractionResults
        """
        self.logger.info(f"Extracting clips from {len(json_files)} JSON files")
        
        if not Path(source_video).exists():
            raise FileNotFoundError(f"Source video not found: {source_video}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        for i, json_file in enumerate(json_files, 1):
            try:
                self.logger.info(f"Processing JSON file {i}/{len(json_files)}: {Path(json_file).name}")
                
                # Load scene data
                with open(json_file, 'r') as f:
                    scene_data = json.load(f)
                
                # Generate custom name based on JSON filename
                json_name = Path(json_file).stem
                custom_name = f"{json_name}_clip"
                
                # Extract clip
                result = self.extract_clip(source_video, scene_data, str(output_path), custom_name)
                results.append(result)
                
                if not result.success:
                    self.logger.warning(f"Failed to extract clip from {json_file}: {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {json_file}: {e}")
                results.append(ClipExtractionResult(
                    clip_path="",
                    scene_info={},
                    success=False,
                    error_message=str(e)
                ))
        
        success_count = sum(1 for r in results if r.success)
        self.logger.info(f"Clip extraction complete: {success_count}/{len(results)} successful")
        
        return results
    
    def extract_clips_from_directory(self, json_dir: str, source_video: str,
                                   output_dir: str = "./extracted_clips") -> List[ClipExtractionResult]:
        """
        Extract clips from all JSON files in a directory
        
        Args:
            json_dir: Directory containing JSON scene files
            source_video: Path to source video file
            output_dir: Directory to save extracted clips
            
        Returns:
            List of ClipExtractionResults
        """
        json_path = Path(json_dir)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON directory not found: {json_dir}")
        
        # Find all JSON files
        json_files = list(json_path.glob("*.json"))
        if not json_files:
            self.logger.warning(f"No JSON files found in {json_dir}")
            return []
        
        self.logger.info(f"Found {len(json_files)} JSON files in {json_dir}")
        return self.extract_clips_from_json_files([str(f) for f in json_files], 
                                                source_video, output_dir)
    
    def create_clips_summary(self, results: List[ClipExtractionResult], 
                           output_dir: str) -> str:
        """Create a summary report of clip extraction results"""
        summary_path = Path(output_dir) / "extraction_summary.md"
        
        successful_clips = [r for r in results if r.success]
        failed_clips = [r for r in results if not r.success]
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# Clip Extraction Summary\n\n")
            f.write(f"**Total clips processed**: {len(results)}\n")
            f.write(f"**Successfully extracted**: {len(successful_clips)}\n")
            f.write(f"**Failed extractions**: {len(failed_clips)}\n\n")
            
            if successful_clips:
                total_size = sum(r.file_size_mb for r in successful_clips if r.file_size_mb)
                f.write(f"**Total file size**: {total_size:.1f} MB\n\n")
                
                f.write("## Successfully Extracted Clips\n\n")
                for i, result in enumerate(successful_clips, 1):
                    clip_name = Path(result.clip_path).name
                    scene_info = result.scene_info
                    start_time = scene_info.get('start_time', 0)
                    duration = scene_info.get('duration', 0)
                    
                    f.write(f"{i}. **{clip_name}**\n")
                    f.write(f"   - Start: {start_time:.2f}s\n")
                    f.write(f"   - Duration: {duration:.2f}s\n")
                    f.write(f"   - Size: {result.file_size_mb:.1f} MB\n\n")
            
            if failed_clips:
                f.write("## Failed Extractions\n\n")
                for i, result in enumerate(failed_clips, 1):
                    f.write(f"{i}. **Error**: {result.error_message}\n")
                    if result.scene_info:
                        start_time = result.scene_info.get('start_time', 'unknown')
                        f.write(f"   - Scene: {start_time}s\n")
                    f.write("\n")
        
        return str(summary_path)