#!/usr/bin/env python3
"""
Anime Effects Studio - Launch Script
Simplified launcher for the standalone video effects UI
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = [
        'tkinter',
        'customtkinter', 
        'moviepy',
        'cv2',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'customtkinter':
                import customtkinter
            elif package == 'moviepy':
                import moviepy
            elif package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def setup_environment():
    """Setup environment and paths"""
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Add parent directory (project root) to path
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))
    
    print(f"🚀 Starting Anime Effects Studio...")
    print(f"📁 Working directory: {current_dir}")
    print(f"🔧 Project root: {project_root}")

def main():
    """Main launcher function"""
    print("🎬 Anime Effects Studio - Video Effects Pipeline")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    try:
        # Import and start the application
        print("🔄 Importing main application...")
        from main_ui import AnimeEffectsStudio
        
        print("✅ Dependencies loaded successfully")
        print("🎭 Launching UI...")
        
        # Create and run application
        app = AnimeEffectsStudio()
        print("🚀 Application created, starting main loop...")
        app.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the correct directory")
        print("💡 Try running: python test_ui.py first to check components")
        sys.exit(1)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Unexpected error: {e}")
        print(f"🔍 Full error details:\n{error_details}")
        print("💡 Check the error details above")
        sys.exit(1)

if __name__ == "__main__":
    main()