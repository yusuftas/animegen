"""
Content Matching Engine
Finds related moments, patterns, and connections across anime content
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
import sqlite3
from pathlib import Path
import json
import hashlib
from collections import defaultdict

try:
    from ..scene_analysis.scene_extractor import Scene
    from ..scene_analysis.interest_detector import InterestScore
    from .moment_classifier import MomentClassification, MomentType
    from .anime_knowledge_base import AnimeInfo
    from ..utils.logger import LoggerMixin
except ImportError:
    from scene_analysis.scene_extractor import Scene
    from scene_analysis.interest_detector import InterestScore
    from content_intelligence.moment_classifier import MomentClassification, MomentType
    from content_intelligence.anime_knowledge_base import AnimeInfo
    from utils.logger import LoggerMixin

@dataclass
class ContentMatch:
    scene_id: str
    match_type: str  # 'visual_similarity', 'thematic_similarity', 'pattern_match', 'reference'
    similarity_score: float
    anime_info: Dict[str, Any]
    scene_info: Dict[str, Any]
    explanation: str
    metadata: Dict[str, Any]

@dataclass
class SceneEmbedding:
    scene_id: str
    anime_id: int
    episode: int
    features: np.ndarray
    moment_type: MomentType
    tags: List[str]
    metadata: Dict[str, Any]

class ContentMatcher(LoggerMixin):
    """Finds relationships and patterns between anime content"""
    
    def __init__(self, db_path: str = "data/anime_db/content_matches.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        self.feature_cache = {}
        self.similarity_threshold = 0.7
        
        self.logger.info("Content Matcher initialized")
    
    def _init_database(self):
        """Initialize database for storing scene embeddings and matches"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS scene_embeddings (
                    scene_id TEXT PRIMARY KEY,
                    anime_id INTEGER,
                    episode INTEGER,
                    start_time REAL,
                    end_time REAL,
                    moment_type TEXT,
                    features BLOB,
                    tags TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS content_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scene_id_1 TEXT,
                    scene_id_2 TEXT,
                    match_type TEXT,
                    similarity_score REAL,
                    explanation TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pattern_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT,
                    pattern_type TEXT,
                    features BLOB,
                    description TEXT,
                    examples TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_anime_id ON scene_embeddings(anime_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_moment_type ON scene_embeddings(moment_type)')
    
    def store_scene_embedding(self, scene: Scene, anime_id: int, episode: int,
                            classification: MomentClassification, 
                            interest_score: InterestScore,
                            anime_info: Optional[AnimeInfo] = None) -> str:
        """Store scene embedding for future matching"""
        
        # Generate unique scene ID
        scene_id = self._generate_scene_id(anime_id, episode, scene.start_time)
        
        # Create feature vector
        features = self._create_feature_vector(scene, classification, interest_score, anime_info)
        
        # Generate tags
        tags = self._generate_scene_tags(classification, anime_info)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO scene_embeddings 
                (scene_id, anime_id, episode, start_time, end_time, moment_type, features, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                scene_id,
                anime_id,
                episode,
                scene.start_time,
                scene.end_time,
                classification.primary_type.value,
                features.tobytes(),
                json.dumps(tags),
                json.dumps({
                    'duration': scene.duration,
                    'confidence': classification.confidence,
                    'interest_score': interest_score.total_score
                })
            ))
        
        return scene_id
    
    def find_related_moments(self, scene: Scene, classification: MomentClassification,
                           interest_score: InterestScore, anime_info: Optional[AnimeInfo] = None,
                           limit: int = 5) -> List[ContentMatch]:
        """Find scenes related to the current scene"""
        
        # Create feature vector for current scene
        query_features = self._create_feature_vector(scene, classification, interest_score, anime_info)
        
        matches = []
        
        # Find similar scenes by feature similarity
        visual_matches = self._find_visual_similarities(query_features, limit)
        matches.extend(visual_matches)
        
        # Find thematic similarities
        thematic_matches = self._find_thematic_similarities(classification, anime_info, limit)
        matches.extend(thematic_matches)
        
        # Find pattern matches
        pattern_matches = self._find_pattern_matches(classification, limit)
        matches.extend(pattern_matches)
        
        # Remove duplicates and sort by score
        unique_matches = self._deduplicate_matches(matches)
        return sorted(unique_matches, key=lambda x: x.similarity_score, reverse=True)[:limit]
    
    def find_anime_references(self, anime_info: AnimeInfo, scene_context: Dict[str, Any]) -> List[ContentMatch]:
        """Find references to other anime in the current scene"""
        references = []
        
        # Genre-based references
        genre_refs = self._find_genre_references(anime_info)
        references.extend(genre_refs)
        
        # Studio references
        studio_refs = self._find_studio_references(anime_info)
        references.extend(studio_refs)
        
        # Character archetype references
        character_refs = self._find_character_references(anime_info, scene_context)
        references.extend(character_refs)
        
        return references
    
    def detect_common_patterns(self, scene_batch: List[Tuple[Scene, MomentClassification, InterestScore]]) -> Dict[str, List[str]]:
        """Detect common patterns across a batch of scenes"""
        patterns = defaultdict(list)
        
        # Group by moment type
        type_groups = defaultdict(list)
        for i, (scene, classification, _) in enumerate(scene_batch):
            type_groups[classification.primary_type].append(i)
        
        # Analyze patterns within each type
        for moment_type, scene_indices in type_groups.items():
            if len(scene_indices) >= 3:  # Need at least 3 scenes
                type_patterns = self._analyze_moment_type_patterns(
                    [scene_batch[i] for i in scene_indices], moment_type
                )
                patterns[f"{moment_type.value}_patterns"] = type_patterns
        
        # Find temporal patterns
        temporal_patterns = self._find_temporal_patterns(scene_batch)
        patterns["temporal_patterns"] = temporal_patterns
        
        # Find duration patterns
        duration_patterns = self._find_duration_patterns(scene_batch)
        patterns["duration_patterns"] = duration_patterns
        
        return dict(patterns)
    
    def _create_feature_vector(self, scene: Scene, classification: MomentClassification,
                             interest_score: InterestScore, anime_info: Optional[AnimeInfo] = None) -> np.ndarray:
        """Create feature vector for scene embedding"""
        features = []
        
        # Basic scene features
        features.extend([
            scene.duration / 30.0,  # Normalized duration
            interest_score.motion_score,
            interest_score.face_score,
            interest_score.color_variance_score,
            interest_score.composition_score,
            interest_score.audio_peak_score,
            classification.confidence
        ])
        
        # Moment type one-hot encoding
        moment_types = list(MomentType)
        type_encoding = [0.0] * len(moment_types)
        if classification.primary_type in moment_types:
            type_encoding[moment_types.index(classification.primary_type)] = 1.0
        features.extend(type_encoding)
        
        # Additional classification features
        features.extend([
            classification.features.get('edge_density', 0.0),
            classification.features.get('brightness_variance', 0.0),
            classification.features.get('color_temperature', 0.5),
            classification.features.get('contrast_level', 0.0)
        ])
        
        # Anime context features (if available)
        if anime_info:
            features.extend([
                len(anime_info.genres) / 10.0,  # Normalized genre count
                anime_info.score / 10.0 if anime_info.score else 0.5,
                anime_info.popularity / 100000.0 if anime_info.popularity else 0.5,
                len(anime_info.characters) / 20.0  # Normalized character count
            ])
        else:
            features.extend([0.0, 0.5, 0.5, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def _generate_scene_tags(self, classification: MomentClassification, 
                           anime_info: Optional[AnimeInfo] = None) -> List[str]:
        """Generate tags for scene indexing"""
        tags = []
        
        # Add moment type tags
        tags.append(classification.primary_type.value)
        for secondary_type, _ in classification.secondary_types:
            tags.append(secondary_type.value)
        
        # Add confidence level tags
        if classification.confidence > 0.8:
            tags.append("high_confidence")
        elif classification.confidence > 0.5:
            tags.append("medium_confidence")
        else:
            tags.append("low_confidence")
        
        # Add anime context tags
        if anime_info:
            tags.extend([genre.lower().replace(' ', '_') for genre in anime_info.genres])
            tags.extend([theme.lower().replace(' ', '_') for theme in anime_info.themes])
            if anime_info.studios:
                tags.extend([studio.lower().replace(' ', '_') for studio in anime_info.studios])
        
        return tags
    
    def _generate_scene_id(self, anime_id: int, episode: int, start_time: float) -> str:
        """Generate unique scene identifier"""
        identifier = f"{anime_id}_{episode}_{start_time:.2f}"
        return hashlib.md5(identifier.encode()).hexdigest()
    
    def _find_visual_similarities(self, query_features: np.ndarray, limit: int) -> List[ContentMatch]:
        """Find visually similar scenes using feature vectors"""
        matches = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT scene_id, anime_id, episode, start_time, end_time, 
                       moment_type, features, metadata 
                FROM scene_embeddings
                ORDER BY created_at DESC
                LIMIT 1000
            ''')
            
            for row in cursor.fetchall():
                stored_features = np.frombuffer(row[6], dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_features, stored_features)
                
                if similarity > self.similarity_threshold:
                    match = ContentMatch(
                        scene_id=row[0],
                        match_type="visual_similarity",
                        similarity_score=similarity,
                        anime_info={'anime_id': row[1], 'episode': row[2]},
                        scene_info={
                            'start_time': row[3],
                            'end_time': row[4],
                            'moment_type': row[5]
                        },
                        explanation=f"Visually similar {row[5]} scene with {similarity:.1%} similarity",
                        metadata=json.loads(row[7])
                    )
                    matches.append(match)
        
        return sorted(matches, key=lambda x: x.similarity_score, reverse=True)[:limit]
    
    def _find_thematic_similarities(self, classification: MomentClassification,
                                  anime_info: Optional[AnimeInfo], limit: int) -> List[ContentMatch]:
        """Find thematically similar scenes"""
        matches = []
        
        if not anime_info:
            return matches
        
        with sqlite3.connect(self.db_path) as conn:
            # Find scenes with similar moment types
            cursor = conn.execute('''
                SELECT scene_id, anime_id, episode, start_time, end_time,
                       moment_type, tags, metadata
                FROM scene_embeddings
                WHERE moment_type = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (classification.primary_type.value, limit * 2))
            
            for row in cursor.fetchall():
                tags = json.loads(row[6])
                metadata = json.loads(row[7])
                
                # Calculate thematic similarity based on tags
                similarity = self._calculate_tag_similarity(
                    self._generate_scene_tags(classification, anime_info),
                    tags
                )
                
                if similarity > 0.4:  # Lower threshold for thematic similarity
                    match = ContentMatch(
                        scene_id=row[0],
                        match_type="thematic_similarity",
                        similarity_score=similarity,
                        anime_info={'anime_id': row[1], 'episode': row[2]},
                        scene_info={
                            'start_time': row[3],
                            'end_time': row[4],
                            'moment_type': row[5]
                        },
                        explanation=f"Thematically similar {row[5]} scene",
                        metadata=metadata
                    )
                    matches.append(match)
        
        return matches[:limit]
    
    def _find_pattern_matches(self, classification: MomentClassification, limit: int) -> List[ContentMatch]:
        """Find scenes matching known patterns"""
        # This would implement pattern matching logic
        # For now, return empty list as patterns need to be learned over time
        return []
    
    def _find_genre_references(self, anime_info: AnimeInfo) -> List[ContentMatch]:
        """Find references to similar genres"""
        references = []
        
        # Common genre patterns and their typical references
        genre_patterns = {
            'Shounen': ['power-up scenes', 'friendship themes', 'tournament arcs'],
            'Shoujo': ['romantic moments', 'character development', 'emotional scenes'],
            'Mecha': ['transformation sequences', 'robot battles', 'pilot dedication'],
            'Slice of Life': ['daily activities', 'character interactions', 'peaceful moments']
        }
        
        for genre in anime_info.genres:
            if genre in genre_patterns:
                for pattern in genre_patterns[genre]:
                    reference = ContentMatch(
                        scene_id="genre_ref",
                        match_type="genre_reference",
                        similarity_score=0.6,
                        anime_info={'genre': genre},
                        scene_info={'pattern': pattern},
                        explanation=f"Common {genre} anime pattern: {pattern}",
                        metadata={'type': 'genre_pattern'}
                    )
                    references.append(reference)
        
        return references
    
    def _find_studio_references(self, anime_info: AnimeInfo) -> List[ContentMatch]:
        """Find references to studio styles"""
        references = []
        
        # Studio signature styles
        studio_styles = {
            'Studio Ghibli': 'fluid animation and natural backgrounds',
            'Madhouse': 'detailed action sequences and character expressions',
            'Bones': 'dynamic fight choreography',
            'KyoAni': 'exceptional character animation and lighting',
            'Trigger': 'over-the-top action and vibrant colors'
        }
        
        for studio in anime_info.studios:
            if studio in studio_styles:
                reference = ContentMatch(
                    scene_id="studio_ref",
                    match_type="studio_reference",
                    similarity_score=0.5,
                    anime_info={'studio': studio},
                    scene_info={'style': studio_styles[studio]},
                    explanation=f"Signature {studio} animation style",
                    metadata={'type': 'studio_style'}
                )
                references.append(reference)
        
        return references
    
    def _find_character_references(self, anime_info: AnimeInfo, scene_context: Dict[str, Any]) -> List[ContentMatch]:
        """Find character archetype references"""
        # This would analyze character types and find similar archetypes
        # Implementation would depend on character detection and classification
        return []
    
    def _analyze_moment_type_patterns(self, scenes: List[Tuple[Scene, MomentClassification, InterestScore]], 
                                    moment_type: MomentType) -> List[str]:
        """Analyze patterns within a specific moment type"""
        patterns = []
        
        durations = [scene[0].duration for scene in scenes]
        avg_duration = np.mean(durations)
        
        if avg_duration > 10:
            patterns.append(f"Long {moment_type.value} scenes (avg: {avg_duration:.1f}s)")
        elif avg_duration < 3:
            patterns.append(f"Short {moment_type.value} scenes (avg: {avg_duration:.1f}s)")
        
        # Interest score patterns
        interest_scores = [scene[2].total_score for scene in scenes]
        if np.mean(interest_scores) > 0.7:
            patterns.append(f"High-interest {moment_type.value} scenes")
        
        return patterns
    
    def _find_temporal_patterns(self, scene_batch: List[Tuple[Scene, MomentClassification, InterestScore]]) -> List[str]:
        """Find temporal patterns in scene sequences"""
        patterns = []
        
        if len(scene_batch) < 3:
            return patterns
        
        # Check for alternating patterns
        types = [scene[1].primary_type for scene in scene_batch]
        
        # Look for ABAB patterns
        if len(types) >= 4:
            if types[0] == types[2] and types[1] == types[3] and types[0] != types[1]:
                patterns.append(f"Alternating pattern: {types[0].value} → {types[1].value}")
        
        return patterns
    
    def _find_duration_patterns(self, scene_batch: List[Tuple[Scene, MomentClassification, InterestScore]]) -> List[str]:
        """Find duration-based patterns"""
        patterns = []
        
        durations = [scene[0].duration for scene in scene_batch]
        
        if len(durations) >= 3:
            # Check for increasing/decreasing patterns
            if all(durations[i] < durations[i+1] for i in range(len(durations)-1)):
                patterns.append("Progressively longer scenes")
            elif all(durations[i] > durations[i+1] for i in range(len(durations)-1)):
                patterns.append("Progressively shorter scenes")
        
        return patterns
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _calculate_tag_similarity(self, tags1: List[str], tags2: List[str]) -> float:
        """Calculate similarity based on tag overlap"""
        set1, set2 = set(tags1), set(tags2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _deduplicate_matches(self, matches: List[ContentMatch]) -> List[ContentMatch]:
        """Remove duplicate matches"""
        seen = set()
        unique_matches = []
        
        for match in matches:
            key = (match.scene_id, match.match_type)
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        return unique_matches
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get content matcher statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Scene embeddings count
            cursor = conn.execute('SELECT COUNT(*) FROM scene_embeddings')
            stats['total_scenes'] = cursor.fetchone()[0]
            
            # Moment type distribution
            cursor = conn.execute('''
                SELECT moment_type, COUNT(*) 
                FROM scene_embeddings 
                GROUP BY moment_type
            ''')
            stats['moment_type_distribution'] = dict(cursor.fetchall())
            
            # Matches count
            cursor = conn.execute('SELECT COUNT(*) FROM content_matches')
            stats['total_matches'] = cursor.fetchone()[0]
            
            return stats