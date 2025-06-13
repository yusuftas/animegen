"""
Anime Knowledge Base Module
Integrates with external APIs and maintains local cache of anime information
"""

import requests
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import time
import logging

try:
    from ..utils.logger import LoggerMixin
except ImportError:
    from utils.logger import LoggerMixin

@dataclass
class AnimeInfo:
    mal_id: Optional[int]
    title: str
    title_english: Optional[str]
    title_japanese: Optional[str]
    synopsis: str
    genres: List[str]
    themes: List[str]
    studios: List[str]
    year: Optional[int]
    season: Optional[str]
    episodes: Optional[int]
    score: Optional[float]
    popularity: Optional[int]
    characters: List[Dict[str, Any]]
    staff: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    external_links: List[Dict[str, Any]]
    cached_at: datetime

@dataclass
class CharacterInfo:
    mal_id: Optional[int]
    name: str
    name_kanji: Optional[str]
    about: str
    anime_appearances: List[str]
    voice_actors: List[Dict[str, Any]]

@dataclass
class AnimeTrivia:
    anime_id: int
    trivia_type: str  # 'production', 'cultural', 'reference', 'easter_egg'
    content: str
    source: str
    verified: bool
    relevance_score: float

class AnimeKnowledgeBase(LoggerMixin):
    """Central repository for anime information and trivia"""
    
    def __init__(self, db_path: str = "data/anime_db/anime_knowledge.db", cache_duration_days: int = 7):
        self.db_path = Path(db_path)
        self.cache_duration = timedelta(days=cache_duration_days)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AnimeShorts/1.0 (Educational Research Tool)'
        })
        
        # API endpoints
        self.jikan_base = "https://api.jikan.moe/v4"
        self.anilist_base = "https://graphql.anilist.co"
        
        # Rate limiting
        self.last_api_call = {}
        self.api_delay = {"jikan": 1.0, "anilist": 0.5}  # seconds between calls
        
        self._init_database()
        self.logger.info("Anime Knowledge Base initialized")
    
    def _init_database(self):
        """Initialize local SQLite database for caching"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS anime_info (
                    mal_id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    data TEXT NOT NULL,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS characters (
                    mal_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    data TEXT NOT NULL,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trivia (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    anime_id INTEGER,
                    trivia_type TEXT,
                    content TEXT,
                    source TEXT,
                    verified BOOLEAN,
                    relevance_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS search_cache (
                    query TEXT PRIMARY KEY,
                    results TEXT NOT NULL,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def search_anime(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for anime by title"""
        self.logger.info(f"Searching for anime: {query}")
        
        # Check cache first
        cached_results = self._get_cached_search(query)
        if cached_results:
            return cached_results[:limit]
        
        # Search using Jikan API (MyAnimeList)
        results = []
        try:
            results.extend(self._search_jikan(query, limit))
        except Exception as e:
            self.logger.warning(f"Jikan search failed: {e}")
        
        # Fallback to AniList if needed
        if not results:
            try:
                results.extend(self._search_anilist(query, limit))
            except Exception as e:
                self.logger.warning(f"AniList search failed: {e}")
        
        # Cache results
        if results:
            self._cache_search_results(query, results)
        
        return results[:limit]
    
    def get_anime_info(self, anime_identifier: str) -> Optional[AnimeInfo]:
        """Get comprehensive anime information"""
        # Try to parse as MAL ID first
        mal_id = None
        if anime_identifier.isdigit():
            mal_id = int(anime_identifier)
        
        # Check cache
        if mal_id:
            cached_info = self._get_cached_anime(mal_id)
            if cached_info:
                return cached_info
        
        # Search if not a direct ID
        if not mal_id:
            search_results = self.search_anime(anime_identifier, limit=1)
            if search_results:
                mal_id = search_results[0].get('mal_id')
        
        if not mal_id:
            self.logger.warning(f"Could not find MAL ID for: {anime_identifier}")
            return None
        
        # Fetch from API
        anime_info = self._fetch_anime_details(mal_id)
        if anime_info:
            self._cache_anime_info(anime_info)
        
        return anime_info
    
    def get_context_for_scene(self, anime_id: int, episode: int, timestamp: float) -> Dict[str, Any]:
        """Get relevant context for a specific scene"""
        anime_info = self.get_anime_info(str(anime_id))
        if not anime_info:
            return {}
        
        context = {
            'anime_title': anime_info.title,
            'genres': anime_info.genres,
            'themes': anime_info.themes,
            'episode_number': episode,
            'timestamp': timestamp,
            'characters': anime_info.characters[:5],  # Top 5 characters
            'trivia': self.get_relevant_trivia(anime_id, episode),
            'production_info': {
                'studios': anime_info.studios,
                'year': anime_info.year,
                'season': anime_info.season
            }
        }
        
        return context
    
    def get_relevant_trivia(self, anime_id: int, episode: int = None, limit: int = 3) -> List[AnimeTrivia]:
        """Get relevant trivia for anime/episode"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT * FROM trivia 
                WHERE anime_id = ? 
                ORDER BY relevance_score DESC, verified DESC 
                LIMIT ?
            '''
            cursor = conn.execute(query, (anime_id, limit))
            
            trivia_list = []
            for row in cursor.fetchall():
                trivia = AnimeTrivia(
                    anime_id=row[1],
                    trivia_type=row[2],
                    content=row[3],
                    source=row[4],
                    verified=bool(row[5]),
                    relevance_score=row[6]
                )
                trivia_list.append(trivia)
            
            return trivia_list
    
    def add_trivia(self, anime_id: int, trivia_type: str, content: str, 
                   source: str, verified: bool = False, relevance_score: float = 0.5):
        """Add trivia to the knowledge base"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO trivia (anime_id, trivia_type, content, source, verified, relevance_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (anime_id, trivia_type, content, source, verified, relevance_score))
    
    def _search_jikan(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using Jikan API"""
        self._rate_limit("jikan")
        
        url = f"{self.jikan_base}/anime"
        params = {'q': query, 'limit': limit, 'order_by': 'popularity'}
        
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for anime in data.get('data', []):
            result = {
                'mal_id': anime.get('mal_id'),
                'title': anime.get('title'),
                'title_english': anime.get('title_english'),
                'year': anime.get('year'),
                'score': anime.get('score'),
                'popularity': anime.get('popularity'),
                'source': 'jikan'
            }
            results.append(result)
        
        return results
    
    def _search_anilist(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using AniList GraphQL API"""
        self._rate_limit("anilist")
        
        graphql_query = '''
        query ($search: String, $perPage: Int) {
            Page(perPage: $perPage) {
                media(search: $search, type: ANIME) {
                    id
                    idMal
                    title {
                        romaji
                        english
                        native
                    }
                    startDate {
                        year
                    }
                    averageScore
                    popularity
                }
            }
        }
        '''
        
        variables = {'search': query, 'perPage': limit}
        
        response = self.session.post(
            self.anilist_base,
            json={'query': graphql_query, 'variables': variables},
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for anime in data.get('data', {}).get('Page', {}).get('media', []):
            result = {
                'mal_id': anime.get('idMal'),
                'anilist_id': anime.get('id'),
                'title': anime.get('title', {}).get('romaji'),
                'title_english': anime.get('title', {}).get('english'),
                'year': anime.get('startDate', {}).get('year'),
                'score': anime.get('averageScore'),
                'popularity': anime.get('popularity'),
                'source': 'anilist'
            }
            results.append(result)
        
        return results
    
    def _fetch_anime_details(self, mal_id: int) -> Optional[AnimeInfo]:
        """Fetch detailed anime information"""
        self._rate_limit("jikan")
        
        try:
            # Basic anime info
            url = f"{self.jikan_base}/anime/{mal_id}/full"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json().get('data', {})
            
            # Characters
            self._rate_limit("jikan")
            char_url = f"{self.jikan_base}/anime/{mal_id}/characters"
            char_response = self.session.get(char_url, timeout=10)
            characters = []
            if char_response.status_code == 200:
                char_data = char_response.json().get('data', [])
                characters = [
                    {
                        'name': char.get('character', {}).get('name'),
                        'role': char.get('role'),
                        'voice_actors': [
                            {
                                'name': va.get('person', {}).get('name'),
                                'language': va.get('language')
                            }
                            for va in char.get('voice_actors', [])
                        ]
                    }
                    for char in char_data[:10]  # Top 10 characters
                ]
            
            anime_info = AnimeInfo(
                mal_id=data.get('mal_id'),
                title=data.get('title', ''),
                title_english=data.get('title_english'),
                title_japanese=data.get('title_japanese'),
                synopsis=data.get('synopsis', ''),
                genres=[g.get('name') for g in data.get('genres', [])],
                themes=[t.get('name') for t in data.get('themes', [])],
                studios=[s.get('name') for s in data.get('studios', [])],
                year=data.get('year'),
                season=data.get('season'),
                episodes=data.get('episodes'),
                score=data.get('score'),
                popularity=data.get('popularity'),
                characters=characters,
                staff=[],  # Could be expanded
                relations=[],  # Could be expanded
                external_links=[],  # Could be expanded
                cached_at=datetime.now()
            )
            
            return anime_info
            
        except Exception as e:
            self.logger.error(f"Failed to fetch anime details for MAL ID {mal_id}: {e}")
            return None
    
    def _rate_limit(self, api: str):
        """Implement rate limiting for API calls"""
        now = time.time()
        last_call = self.last_api_call.get(api, 0)
        delay = self.api_delay.get(api, 1.0)
        
        time_since_last = now - last_call
        if time_since_last < delay:
            sleep_time = delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_api_call[api] = time.time()
    
    def _get_cached_search(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT results, cached_at FROM search_cache WHERE query = ?',
                (query,)
            )
            row = cursor.fetchone()
            
            if row:
                cached_at = datetime.fromisoformat(row[1])
                if datetime.now() - cached_at < self.cache_duration:
                    return json.loads(row[0])
        
        return None
    
    def _cache_search_results(self, query: str, results: List[Dict[str, Any]]):
        """Cache search results"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT OR REPLACE INTO search_cache (query, results) VALUES (?, ?)',
                (query, json.dumps(results))
            )
    
    def _get_cached_anime(self, mal_id: int) -> Optional[AnimeInfo]:
        """Get cached anime information"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT data, cached_at FROM anime_info WHERE mal_id = ?',
                (mal_id,)
            )
            row = cursor.fetchone()
            
            if row:
                cached_at = datetime.fromisoformat(row[1])
                if datetime.now() - cached_at < self.cache_duration:
                    data = json.loads(row[0])
                    data['cached_at'] = cached_at
                    return AnimeInfo(**data)
        
        return None
    
    def _cache_anime_info(self, anime_info: AnimeInfo):
        """Cache anime information"""
        with sqlite3.connect(self.db_path) as conn:
            data = asdict(anime_info)
            data['cached_at'] = anime_info.cached_at.isoformat()
            
            conn.execute(
                'INSERT OR REPLACE INTO anime_info (mal_id, title, data) VALUES (?, ?, ?)',
                (anime_info.mal_id, anime_info.title, json.dumps(data))
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Count cached anime
            cursor = conn.execute('SELECT COUNT(*) FROM anime_info')
            stats['cached_anime'] = cursor.fetchone()[0]
            
            # Count trivia entries
            cursor = conn.execute('SELECT COUNT(*) FROM trivia')
            stats['trivia_entries'] = cursor.fetchone()[0]
            
            # Count search cache
            cursor = conn.execute('SELECT COUNT(*) FROM search_cache')
            stats['cached_searches'] = cursor.fetchone()[0]
            
            return stats