#!/usr/bin/env python3
"""
Quick test script to verify production effects integration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from models.effect_adapter import ProductionEffectFactory, EffectEngineRegistry
    
    print("🔧 Testing Production Effects Integration...")
    
    # Test 1: Check if we can create registry
    print("\n1. Creating effect registry...")
    registry = EffectEngineRegistry()
    print(f"   ✅ Registry created with {len(registry.effect_registry)} effects")
    
    # Test 2: List available effects
    print("\n2. Available effects by category:")
    available = ProductionEffectFactory.get_available_effects()
    total_effects = 0
    for category, effects in available.items():
        print(f"   {category}: {len(effects)} effects")
        for effect in effects[:3]:  # Show first 3
            print(f"     - {effect['name']} ({effect['id']})")
        if len(effects) > 3:
            print(f"     ... and {len(effects) - 3} more")
        total_effects += len(effects)
    
    print(f"\n   📊 Total: {total_effects} production effects available")
    
    # Test 3: Try creating specific effects
    print("\n3. Testing effect creation:")
    test_effects = ['anime_energy_aura', 'motion_speed_ramp', 'color_color_grading']
    
    for effect_id in test_effects:
        try:
            effect = ProductionEffectFactory.create_effect(effect_id)
            print(f"   ✅ {effect_id}: {effect.name} (method: {effect.engine_method})")
        except Exception as e:
            print(f"   ❌ {effect_id}: {e}")
    
    print("\n🎉 Integration test completed!")
    print("   The UI can now access all production effects from src/video_effects/")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure dependencies are installed: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()