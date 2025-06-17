#!/usr/bin/env python3
"""
Simple test script for the Anime Effects Studio UI
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Add parent directory (project root) to path
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    try:
        import tkinter
        print("✅ tkinter imported successfully")
    except ImportError as e:
        print(f"❌ tkinter import failed: {e}")
        return False
    
    try:
        import customtkinter as ctk
        print("✅ customtkinter imported successfully")
    except ImportError as e:
        print(f"❌ customtkinter import failed: {e}")
        return False
    
    try:
        from ui_standalone.models.effect_adapter import ProductionEffectFactory
        print("✅ effect_adapter imported successfully")
    except ImportError as e:
        print(f"❌ effect_adapter import failed: {e}")
        return False
    
    try:
        from ui_standalone.models.effect_pipeline import EffectPipeline
        print("✅ effect_pipeline imported successfully")
    except ImportError as e:
        print(f"❌ effect_pipeline import failed: {e}")
        return False
    
    try:
        from ui_standalone.components.effects_library import EffectsLibraryPanel
        print("✅ effects_library imported successfully")
    except ImportError as e:
        print(f"❌ effects_library import failed: {e}")
        return False
    
    try:
        from ui_standalone.components.pipeline_editor import PipelineEditorPanel
        print("✅ pipeline_editor imported successfully")
    except ImportError as e:
        print(f"❌ pipeline_editor import failed: {e}")
        return False
    
    try:
        from ui_standalone.components.video_preview import VideoPreviewPanel
        print("✅ video_preview imported successfully")
    except ImportError as e:
        print(f"❌ video_preview import failed: {e}")
        return False
    
    return True

def test_effect_factory():
    """Test the effect factory"""
    print("\nTesting effect factory...")
    
    try:
        from ui_standalone.models.effect_adapter import ProductionEffectFactory
        # Test getting available effects from production system
        effects = ProductionEffectFactory.get_available_effects()
        print(f"✅ Found {len(effects)} effect categories")
        
        for category, effect_list in effects.items():
            print(f"   {category}: {len(effect_list)} effects")
        
        # Test creating a production effect
        effect = ProductionEffectFactory.create_effect("anime_speed_lines")
        print(f"✅ Created production effect: {effect.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Effect factory test failed: {e}")
        return False

def test_pipeline():
    """Test the effect pipeline"""
    print("\nTesting effect pipeline...")
    
    try:
        from ui_standalone.models.effect_pipeline import EffectPipeline
        
        pipeline = EffectPipeline()
        print("✅ Created empty pipeline")
        
        # Add an effect
        effect_id = pipeline.add_effect("speed_lines", "Test Speed Lines")
        print(f"✅ Added effect with ID: {effect_id}")
        
        # Check pipeline info
        info = pipeline.get_pipeline_info()
        print(f"✅ Pipeline info: {info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎬 Anime Effects Studio - Component Tests")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test effect factory
    if test_effect_factory():
        tests_passed += 1
    
    # Test pipeline
    if test_pipeline():
        tests_passed += 1
    
    print(f"\n🎯 Tests completed: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! UI should work correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)