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
from utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Anime YouTube Shorts Generator")
    parser.add_argument("--video", required=True, help="Path to anime video file")
    parser.add_argument("--anime-name", required=True, help="Name of the anime")
    parser.add_argument("--output-dir", default="./output", help="Output directory for generated shorts")
    parser.add_argument("--max-shorts", type=int, default=5, help="Maximum number of shorts to generate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    logger = setup_logger(verbose=args.verbose)
    logger.info("Starting Anime Shorts Generation Pipeline")
    
    pipeline = AnimeShortsPipeline()
    
    try:
        shorts = pipeline.process_anime_episode(
            video_path=args.video,
            anime_name=args.anime_name,
            output_dir=args.output_dir,
            max_shorts=args.max_shorts
        )
        
        logger.info(f"Successfully generated {len(shorts)} shorts")
        for i, short_path in enumerate(shorts, 1):
            logger.info(f"Short {i}: {short_path}")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()