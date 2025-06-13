"""
Tests for Phase 2 Content Intelligence features
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))

from content_intelligence.anime_knowledge_base import AnimeKnowledgeBase, AnimeInfo, AnimeTrivia
from content_intelligence.moment_classifier import MomentClassifier, MomentType, MomentClassification
from content_intelligence.content_matcher import ContentMatcher, ContentMatch
from script_generation.script_generator import ScriptGenerator, ScriptStyle
from script_generation.fact_integrator import FactIntegrator, FactIntegrationContext
from scene_analysis.scene_extractor import Scene
from scene_analysis.interest_detector import InterestScore

class TestAnimeKnowledgeBase(unittest.TestCase):
    """Test anime knowledge base functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.kb = AnimeKnowledgeBase(db_path=f"{self.temp_dir}/test_kb.db")
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_initialization(self):
        """Test that database initializes correctly"""
        self.assertTrue(Path(self.kb.db_path).exists())
        
        # Test statistics
        stats = self.kb.get_statistics()
        self.assertEqual(stats['cached_anime'], 0)
        self.assertEqual(stats['trivia_entries'], 0)
    
    def test_trivia_management(self):
        """Test adding and retrieving trivia"""
        anime_id = 12345
        
        # Add some trivia
        self.kb.add_trivia(
            anime_id=anime_id,
            trivia_type="production",
            content="This scene took 6 months to animate",
            source="Official interview",
            verified=True,
            relevance_score=0.8
        )
        
        # Retrieve trivia
        trivia_list = self.kb.get_relevant_trivia(anime_id, limit=5)
        self.assertEqual(len(trivia_list), 1)
        self.assertEqual(trivia_list[0].content, "This scene took 6 months to animate")
        self.assertTrue(trivia_list[0].verified)

class TestMomentClassifier(unittest.TestCase):
    """Test moment classification system"""
    
    def setUp(self):
        self.classifier = MomentClassifier()
    
    def test_classifier_initialization(self):
        """Test classifier initializes with proper rules"""
        self.assertIsNotNone(self.classifier.classification_rules)
        self.assertIsNotNone(self.classifier.feature_weights)
        self.assertIn(MomentType.ACTION_SEQUENCE.value, self.classifier.classification_rules)
    
    def test_feature_extraction(self):
        """Test feature extraction from scene data"""
        # Create mock scene and interest score
        scene = Scene(
            start_time=10.0,
            end_time=15.0,
            duration=5.0,
            start_frame=300,
            end_frame=450,
            confidence=0.8,
            metadata={}
        )
        
        interest_score = InterestScore(
            total_score=0.7,
            motion_score=0.8,
            face_score=0.6,
            color_variance_score=0.5,
            composition_score=0.7,
            audio_peak_score=0.4,
            metadata={}
        )
        
        # Mock anime info
        anime_info = AnimeInfo(
            mal_id=1,
            title="Test Anime",
            title_english="Test Anime",
            title_japanese="テストアニメ",
            synopsis="Test synopsis",
            genres=["Action", "Adventure"],
            themes=["Friendship"],
            studios=["Test Studio"],
            year=2023,
            season="Spring",
            episodes=12,
            score=8.5,
            popularity=1000,
            characters=[],
            staff=[],
            relations=[],
            external_links=[],
            cached_at=None
        )
        
        features = self.classifier._extract_scene_features("", scene, interest_score)
        
        # Verify features are extracted
        self.assertIn('motion_intensity', features)
        self.assertIn('face_presence', features)
        self.assertIn('scene_duration', features)
        self.assertEqual(features['motion_intensity'], 0.8)
        self.assertEqual(features['scene_duration'], 5.0)

class TestContentMatcher(unittest.TestCase):
    """Test content matching functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.matcher = ContentMatcher(db_path=f"{self.temp_dir}/test_matcher.db")
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_initialization(self):
        """Test content matcher database setup"""
        self.assertTrue(Path(self.matcher.db_path).exists())
        
        stats = self.matcher.get_statistics()
        self.assertEqual(stats['total_scenes'], 0)
        self.assertEqual(stats['total_matches'], 0)
    
    def test_scene_embedding_storage(self):
        """Test storing and retrieving scene embeddings"""
        # Create test data
        scene = Scene(
            start_time=5.0,
            end_time=10.0,
            duration=5.0,
            start_frame=150,
            end_frame=300,
            confidence=0.9,
            metadata={}
        )
        
        classification = MomentClassification(
            primary_type=MomentType.ACTION_SEQUENCE,
            confidence=0.8,
            secondary_types=[(MomentType.FIGHT_SCENE, 0.6)],
            features={'motion_intensity': 0.9},
            metadata={}
        )
        
        interest_score = InterestScore(0.8, 0.9, 0.7, 0.6, 0.8, 0.5, {})
        
        # Store embedding
        scene_id = self.matcher.store_scene_embedding(
            scene, anime_id=123, episode=1, 
            classification=classification, 
            interest_score=interest_score
        )
        
        self.assertIsNotNone(scene_id)
        
        # Verify storage
        stats = self.matcher.get_statistics()
        self.assertEqual(stats['total_scenes'], 1)

class TestScriptGenerator(unittest.TestCase):
    """Test script generation system"""
    
    def setUp(self):
        self.generator = ScriptGenerator()
    
    def test_generator_initialization(self):
        """Test script generator initializes properly"""
        self.assertIsNotNone(self.generator.styles_database)
        self.assertIn('analytical', self.generator.styles_database)
        self.assertIn('trivia_focused', self.generator.styles_database)
    
    def test_style_selection(self):
        """Test optimal style selection"""
        # Create test classification
        classification = MomentClassification(
            primary_type=MomentType.ACTION_SEQUENCE,
            confidence=0.8,
            secondary_types=[],
            features={},
            metadata={}
        )
        
        anime_info = AnimeInfo(
            mal_id=1,
            title="Test Action Anime",
            title_english="Test Action Anime",
            title_japanese="テストアクションアニメ",
            synopsis="Action-packed anime",
            genres=["Action", "Shounen"],
            themes=["Friendship", "Power"],
            studios=["Action Studio"],
            year=2023,
            season="Spring",
            episodes=24,
            score=8.0,
            popularity=5000,
            characters=[],
            staff=[],
            relations=[],
            external_links=[],
            cached_at=None
        )
        
        style = self.generator._select_optimal_style(classification, anime_info)
        
        # Should prefer enthusiastic or analytical for action sequences
        self.assertIn(style, [ScriptStyle.ENTHUSIASTIC, ScriptStyle.ANALYTICAL])
    
    def test_script_generation_structure(self):
        """Test that generated scripts have proper structure"""
        # Create minimal test data
        scene = Scene(10.0, 20.0, 10.0, 300, 600, 0.8, {})
        classification = MomentClassification(
            MomentType.EMOTIONAL_MOMENT, 0.7, [], {}, {}
        )
        interest_score = InterestScore(0.6, 0.5, 0.8, 0.6, 0.7, 0.4, {})
        anime_info = AnimeInfo(
            1, "Test Anime", "Test Anime", "テスト", "Synopsis",
            ["Drama"], ["Love"], ["Studio"], 2023, "Spring", 12,
            8.5, 1000, [], [], [], [], None
        )
        
        script = self.generator.generate_script(
            scene, classification, interest_score, anime_info,
            style=ScriptStyle.ANALYTICAL
        )
        
        # Verify script structure
        self.assertIsNotNone(script)
        self.assertEqual(script.style, ScriptStyle.ANALYTICAL)
        self.assertGreater(len(script.segments), 0)
        self.assertGreater(script.word_count, 0)
        
        # Verify segments have required fields
        for segment in script.segments:
            self.assertIsInstance(segment.text, str)
            self.assertGreaterEqual(segment.timing, 0.0)
            self.assertIn(segment.emphasis, ['normal', 'excited', 'whisper', 'dramatic', 'analytical'])

class TestFactIntegrator(unittest.TestCase):
    """Test fact integration system"""
    
    def setUp(self):
        self.integrator = FactIntegrator()
    
    def test_integrator_initialization(self):
        """Test fact integrator initializes properly"""
        self.assertIsNotNone(self.integrator.integration_templates)
        self.assertIsNotNone(self.integrator.relevance_weights)
    
    def test_fact_relevance_scoring(self):
        """Test fact relevance scoring"""
        # Create test trivia
        trivia = AnimeTrivia(
            anime_id=123,
            trivia_type="production",
            content="This fight scene took 6 months to animate",
            source="Official interview",
            verified=True,
            relevance_score=0.8
        )
        
        # Create test context
        scene = Scene(5.0, 15.0, 10.0, 150, 450, 0.9, {})
        classification = MomentClassification(
            MomentType.FIGHT_SCENE, 0.9, [], {}, {}
        )
        anime_info = AnimeInfo(
            123, "Test Fighter", "Test Fighter", "テストファイター", "Fighting anime",
            ["Action", "Martial Arts"], ["Competition"], ["Fight Studio"], 
            2023, "Summer", 24, 8.8, 2000, [], [], [], [], None
        )
        
        context = FactIntegrationContext(
            anime_info=anime_info,
            scene=scene,
            moment_classification=classification,
            script_style="analytical",
            available_trivia=[trivia],
            scene_context={'fight_type': 'martial_arts'}
        )
        
        # Score relevance
        scored_facts = self.integrator._score_fact_relevance(context)
        self.assertEqual(len(scored_facts), 1)
        
        fact, score = scored_facts[0]
        self.assertEqual(fact.content, "This fight scene took 6 months to animate")
        self.assertGreater(score, 0.5)  # Should be reasonably relevant
    
    def test_fact_integration_methods(self):
        """Test different fact integration methods"""
        # Test integration method selection
        method = self.integrator._select_integration_method(
            "trivia_focused", 
            self.integrator.FactRelevance.HIGH,
            AnimeTrivia(1, "production", "Test fact", "source", True, 0.8)
        )
        
        self.assertIn(method, ['natural', 'standalone', 'transition', 'parenthetical'])

class TestIntegratedPipeline(unittest.TestCase):
    """Test integration between all components"""
    
    def test_pipeline_with_phase2_features(self):
        """Test that pipeline can be initialized with all Phase 2 features"""
        from automation.pipeline import AnimeShortsPipeline
        
        pipeline = AnimeShortsPipeline()
        
        # Verify all Phase 2 components are initialized
        self.assertIsNotNone(pipeline.anime_knowledge_base)
        self.assertIsNotNone(pipeline.moment_classifier)
        self.assertIsNotNone(pipeline.content_matcher)
        self.assertIsNotNone(pipeline.script_generator)
        self.assertIsNotNone(pipeline.fact_integrator)
        
        # Check capabilities
        stats = pipeline.get_pipeline_stats()
        capabilities = stats['capabilities']
        
        self.assertTrue(capabilities['anime_knowledge_base'])
        self.assertTrue(capabilities['moment_classification'])
        self.assertTrue(capabilities['content_matching'])
        self.assertTrue(capabilities['script_generation'])
        self.assertTrue(capabilities['fact_integration'])

if __name__ == '__main__':
    unittest.main()