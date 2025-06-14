#!/usr/bin/env python3
"""
Anime YouTube Shorts Automation System
Main entry point for the application
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from automation.pipeline import AnimeShortsPipeline
from video_processing.clip_extractor import ClipExtractor
from utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Anime YouTube Shorts Generator")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Generate mode (default behavior)
    generate_parser = subparsers.add_parser('generate', help='Generate scene analysis and JSON files')
    generate_parser.add_argument("--video", required=True, help="Path to anime video file")
    generate_parser.add_argument("--anime-name", required=True, help="Name of the anime")
    generate_parser.add_argument("--output-dir", default="./output", help="Output directory for generated shorts")
    generate_parser.add_argument("--max-shorts", type=int, default=5, help="Maximum number of shorts to generate")
    generate_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    # Extract clips mode
    extract_parser = subparsers.add_parser('extract-clips', help='Extract video clips from existing JSON files')
    extract_parser.add_argument("--video", required=True, help="Path to source video file")
    extract_parser.add_argument("--json-dir", help="Directory containing JSON scene files")
    extract_parser.add_argument("--json-files", nargs="*", help="Specific JSON files to process")
    extract_parser.add_argument("--output-dir", default="./extracted_clips", help="Output directory for video clips")
    extract_parser.add_argument("--quality", choices=["high", "medium", "fast", "youtube_shorts"], 
                               default="high", help="Output video quality preset")
    extract_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    # For backward compatibility, if no subcommand is provided, assume generate mode
    args = parser.parse_args()
    if args.mode is None:
        # Legacy mode - treat as generate
        parser = argparse.ArgumentParser(description="Anime YouTube Shorts Generator")
        parser.add_argument("--video", required=True, help="Path to anime video file")
        parser.add_argument("--anime-name", required=True, help="Name of the anime")
        parser.add_argument("--output-dir", default="./output", help="Output directory for generated shorts")
        parser.add_argument("--max-shorts", type=int, default=5, help="Maximum number of shorts to generate")
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
        args = parser.parse_args()
        args.mode = 'generate'
    
    logger = setup_logger(verbose=args.verbose)
    
    if args.mode == 'generate':
        run_generate_mode(args, logger)
    elif args.mode == 'extract-clips':
        run_extract_clips_mode(args, logger)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

def run_generate_mode(args, logger):
    """Run the scene analysis and JSON generation pipeline"""
    logger.info("Starting Anime Shorts Generation Pipeline")
    
    pipeline = AnimeShortsPipeline()
    
    try:
        shorts = pipeline.process_anime_episode(
            video_path=args.video,
            anime_name=args.anime_name,
            output_dir=args.output_dir,
            max_shorts=args.max_shorts
        )
        
        logger.info(f"Successfully generated {len(shorts)} scene analysis files")
        for i, short_path in enumerate(shorts, 1):
            logger.info(f"Scene {i}: {short_path}")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

def run_extract_clips_mode(args, logger):
    """Run the clip extraction pipeline"""
    logger.info("Starting Clip Extraction Pipeline")
    
    try:
        extractor = ClipExtractor(output_quality=args.quality)
        
        # Determine JSON files to process
        if args.json_files:
            # Process specific JSON files
            json_files = args.json_files
            logger.info(f"Processing {len(json_files)} specified JSON files")
        elif args.json_dir:
            # Process all JSON files in directory
            json_path = Path(args.json_dir)
            if not json_path.exists():
                logger.error(f"JSON directory not found: {args.json_dir}")
                sys.exit(1)
            json_files = list(json_path.glob("*.json"))
            if not json_files:
                logger.error(f"No JSON files found in {args.json_dir}")
                sys.exit(1)
            json_files = [str(f) for f in json_files]
            logger.info(f"Found {len(json_files)} JSON files in {args.json_dir}")
        else:
            logger.error("Must specify either --json-dir or --json-files")
            sys.exit(1)
        
        # Extract clips
        results = extractor.extract_clips_from_json_files(
            json_files=json_files,
            source_video=args.video,
            output_dir=args.output_dir
        )
        
        # Generate summary
        summary_path = extractor.create_clips_summary(results, args.output_dir)
        
        # Report results
        successful_clips = [r for r in results if r.success]
        failed_clips = [r for r in results if not r.success]
        
        logger.info(f"Clip extraction complete:")
        logger.info(f"  • Successfully extracted: {len(successful_clips)} clips")
        logger.info(f"  • Failed extractions: {len(failed_clips)} clips")
        logger.info(f"  • Output directory: {args.output_dir}")
        logger.info(f"  • Summary report: {summary_path}")
        
        if successful_clips:
            total_size = sum(r.file_size_mb for r in successful_clips if r.file_size_mb)
            logger.info(f"  • Total file size: {total_size:.1f} MB")
            
            logger.info("Extracted clips:")
            for result in successful_clips:
                clip_name = Path(result.clip_path).name
                logger.info(f"    - {clip_name} ({result.file_size_mb:.1f} MB)")
        
        if failed_clips:
            logger.warning("Failed extractions:")
            for result in failed_clips:
                logger.warning(f"    - {result.error_message}")
            
    except Exception as e:
        logger.error(f"Clip extraction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()