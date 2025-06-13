"""
Basic tests for the anime shorts generation pipeline
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from automation.pipeline import AnimeShortsPipeline
from utils.config import AppConfig

class TestPipeline(unittest.TestCase):
    """Test the main pipeline functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = AnimeShortsPipeline()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """Test that pipeline initializes correctly"""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.config)
        self.assertIsNotNone(self.pipeline.scene_extractor)
        self.assertIsNotNone(self.pipeline.interest_detector)
        self.assertIsNotNone(self.pipeline.audio_analyzer)
    
    def test_config_loading(self):
        """Test configuration loading"""
        config = self.pipeline.config
        self.assertIsInstance(config, AppConfig)
        self.assertGreater(config.scene_analysis.threshold, 0)
        self.assertGreater(config.scene_analysis.min_scene_length, 0)
        self.assertGreater(config.scene_analysis.max_scene_length, 
                          config.scene_analysis.min_scene_length)
    
    def test_pipeline_stats(self):
        """Test pipeline statistics"""
        stats = self.pipeline.get_pipeline_stats()
        self.assertIn('config', stats)
        self.assertIn('capabilities', stats)
        
        capabilities = stats['capabilities']
        self.assertTrue(capabilities['scene_detection'])
        self.assertTrue(capabilities['interest_scoring'])
        self.assertTrue(capabilities['audio_analysis'])
    
    def test_nonexistent_video_handling(self):
        """Test handling of non-existent video files"""
        fake_video = "/path/that/does/not/exist.mp4"
        
        with self.assertRaises(FileNotFoundError):
            self.pipeline.process_anime_episode(fake_video, "Test Anime")
    
    def test_batch_processing_empty_directory(self):
        """Test batch processing with empty directory"""
        results = self.pipeline.batch_process_episodes(
            self.temp_dir, "Test Anime", self.temp_dir
        )
        self.assertEqual(len(results), 0)

class TestConfigSystem(unittest.TestCase):
    """Test the configuration system"""
    
    def test_default_config_creation(self):
        """Test that default config can be created"""
        from utils.config import ConfigManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.yaml")
            manager = ConfigManager(config_path)
            config = manager.load_config()
            
            self.assertIsInstance(config, AppConfig)
            self.assertTrue(Path(config_path).exists())

class TestSceneAnalysis(unittest.TestCase):
    """Test scene analysis components"""
    
    def test_scene_extractor_initialization(self):
        """Test scene extractor initialization"""
        from scene_analysis.scene_extractor import SceneExtractor
        
        extractor = SceneExtractor()
        self.assertIsNotNone(extractor)
        self.assertGreater(extractor.threshold, 0)
        self.assertGreater(extractor.min_scene_length, 0)
    
    def test_interest_detector_initialization(self):
        """Test interest detector initialization"""
        from scene_analysis.interest_detector import InterestDetector
        
        detector = InterestDetector()
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.config)
    
    def test_audio_analyzer_initialization(self):
        """Test audio analyzer initialization"""
        from scene_analysis.audio_analyzer import AudioAnalyzer
        
        analyzer = AudioAnalyzer()
        self.assertIsNotNone(analyzer)
        self.assertGreater(analyzer.sample_rate, 0)

if __name__ == '__main__':
    unittest.main()