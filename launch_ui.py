#!/usr/bin/env python3
"""
Launch Script for Anime Video Effects UI

This script launches the web-based video effects interface.
"""

import sys
import subprocess
import webbrowser
import time
import threading
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import flask
        import moviepy
        print("✅ Flask and MoviePy are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_effects_modules():
    """Check if video effects modules are available."""
    try:
        sys.path.append(str(Path(__file__).parent / "src"))
        from video_effects import MotionEffectsEngine
        print("✅ Video effects modules are available")
        return True
    except ImportError as e:
        print(f"⚠️  Video effects modules not fully available: {e}")
        print("The UI will still work but with limited functionality")
        return False

def create_sample_structure():
    """Create necessary directory structure."""
    directories = [
        "ui/uploads",
        "ui/outputs", 
        "extracted_clips"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def launch_browser():
    """Launch browser after a short delay."""
    time.sleep(2)
    try:
        webbrowser.open('http://localhost:5000')
        print("🌐 Opened browser to http://localhost:5000")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print("Please manually navigate to: http://localhost:5000")

def main():
    """Main launch function."""
    print("🎌 Anime Video Effects UI Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check effects modules
    effects_available = check_effects_modules()
    
    # Create directory structure
    create_sample_structure()
    
    # Launch browser in background
    browser_thread = threading.Thread(target=launch_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("\n🚀 Starting Video Effects UI...")
    print("📍 Access the interface at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    
    if not effects_available:
        print("\n⚠️  Note: Some effects may not be available due to missing modules")
    
    print("\n" + "=" * 50)
    
    # Launch the Flask app
    try:
        from ui.effects_app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Video Effects UI...")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        print("Make sure all dependencies are installed and try again.")

if __name__ == "__main__":
    main()