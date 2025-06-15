#!/usr/bin/env python3
"""
Rick and Morty S08E01 - Scene Clip Extractor
Extracts video clips for all ranked scenes based on analysis results
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def format_time(seconds):
    """Convert seconds to HH:MM:SS.mmm format for FFmpeg"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def extract_clips():
    """Extract video clips for all ranked scenes"""
    
    # Paths
    base_dir = Path("output/rickmorty_analysis_fixed/S08E01_fixed_analysis_20250613_162107")
    rankings_file = base_dir / "01_scene_rankings" / "complete_scene_rankings.json"
    source_video = Path("assets/RickandMortyS08E01.mkv")
    clips_dir = base_dir / "08_extracted_clips"
    
    # Create clips directory
    clips_dir.mkdir(exist_ok=True)
    
    # Check if source video exists
    if not source_video.exists():
        print(f"❌ Source video not found: {source_video}")
        return False
    
    # Load scene rankings
    try:
        with open(rankings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading rankings: {e}")
        return False
    
    scenes = data.get('scenes', [])
    total_scenes = len(scenes)
    
    print(f"🎬 Extracting {total_scenes} scene clips from Rick and Morty S08E01")
    print(f"📁 Output directory: {clips_dir}")
    print("=" * 60)
    
    success_count = 0
    failed_clips = []
    
    for i, scene in enumerate(scenes, 1):
        scene_id = scene['scene_id']
        rank = scene['rank']
        start_time = scene['start_time']
        end_time = scene['end_time']
        duration = scene['duration']
        score = scene['total_score']
        quality = scene['quality']
        
        # Format timestamps
        start_formatted = format_time(start_time)
        duration_formatted = format_time(duration)
        
        # Create output filename
        output_file = clips_dir / f"scene_{rank:02d}_{scene_id}_score_{score:.3f}.mp4"
        
        print(f"🎯 [{i}/{total_scenes}] Extracting Scene {rank} (Score: {score:.3f})")
        print(f"   ⏱️  Time: {start_formatted} → Duration: {duration_formatted}")
        print(f"   📊 Quality: {quality}")
        
        # FFmpeg command for high-quality extraction
        cmd = [
            'ffmpeg',
            '-i', str(source_video),
            '-ss', start_formatted,
            '-t', duration_formatted,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-crf', '18',  # High quality
            '-preset', 'medium',
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite existing files
            str(output_file)
        ]
        
        try:
            # Run FFmpeg with suppressed output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout per clip
            )
            
            if result.returncode == 0 and output_file.exists():
                file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                print(f"   ✅ Success! File size: {file_size:.1f} MB")
                success_count += 1
            else:
                error_msg = result.stderr if result.stderr else "Unknown error"
                print(f"   ❌ Failed: {error_msg[:100]}...")
                failed_clips.append(f"Scene {rank}: {error_msg[:50]}...")
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Timeout extracting scene {rank}")
            failed_clips.append(f"Scene {rank}: Timeout")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            failed_clips.append(f"Scene {rank}: {str(e)}")
        
        print()
    
    # Summary
    print("=" * 60)
    print(f"🎬 EXTRACTION COMPLETE")
    print(f"✅ Successfully extracted: {success_count}/{total_scenes} clips")
    print(f"📁 Clips saved to: {clips_dir}")
    
    if failed_clips:
        print(f"\n❌ Failed extractions ({len(failed_clips)}):")
        for failure in failed_clips:
            print(f"   • {failure}")
    
    # Create clips summary
    create_clips_summary(clips_dir, scenes, success_count, failed_clips)
    
    return success_count == total_scenes

def create_clips_summary(clips_dir, scenes, success_count, failed_clips):
    """Create a summary file for the extracted clips"""
    
    summary_file = clips_dir / "CLIPS_SUMMARY.md"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Rick and Morty S08E01 - Extracted Scene Clips\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Extraction Summary\n")
        f.write(f"- **Total Scenes**: {len(scenes)}\n")
        f.write(f"- **Successfully Extracted**: {success_count}\n")
        f.write(f"- **Failed Extractions**: {len(failed_clips)}\n\n")
        
        f.write("## Clip Details\n\n")
        
        for scene in scenes:
            rank = scene['rank']
            scene_id = scene['scene_id']
            start_time = scene['start_time']
            duration = scene['duration']
            score = scene['total_score']
            quality = scene['quality']
            
            # Convert to minutes:seconds for readability
            start_min = int(start_time // 60)
            start_sec = start_time % 60
            
            f.write(f"### Scene {rank} - {scene_id}\n")
            f.write(f"- **Score**: {score:.3f} ({quality} Quality)\n")
            f.write(f"- **Timestamp**: {start_min}:{start_sec:05.2f}\n")
            f.write(f"- **Duration**: {duration:.2f}s\n")
            f.write(f"- **File**: `scene_{rank:02d}_{scene_id}_score_{score:.3f}.mp4`\n\n")
        
        if failed_clips:
            f.write("## Failed Extractions\n\n")
            for failure in failed_clips:
                f.write(f"- {failure}\n")
    
    print(f"📄 Clips summary saved: {summary_file}")

def main():
    """Main execution function"""
    print("🎬 Rick and Morty S08E01 - Scene Clip Extractor")
    print("=" * 50)
    
    # Check if FFmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found. Please install FFmpeg to extract clips.")
        print("   Download from: https://ffmpeg.org/download.html")
        return False
    
    # Extract clips
    success = extract_clips()
    
    if success:
        print("\n🎉 All clips extracted successfully!")
        print("📁 Check the 08_extracted_clips folder for your video clips")
    else:
        print("\n⚠️  Some clips failed to extract. Check the summary for details.")
    
    return success

if __name__ == "__main__":
    main() 