"""
Fact Integration System
Intelligently weaves anime facts and trivia into generated scripts
"""

import re
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from ..content_intelligence.anime_knowledge_base import AnimeInfo, AnimeTrivia
from ..content_intelligence.moment_classifier import MomentClassification, MomentType
from ..scene_analysis.scene_extractor import Scene
from ..utils.logger import LoggerMixin

class FactRelevance(Enum):
    HIGH = "high"        # Directly related to current scene
    MEDIUM = "medium"    # Related to anime/episode
    LOW = "low"         # General anime knowledge
    CONTEXTUAL = "contextual"  # Provides useful context

@dataclass
class IntegratedFact:
    fact_text: str
    relevance: FactRelevance
    integration_method: str  # 'natural', 'transition', 'parenthetical', 'standalone'
    confidence: float
    source: str
    timing_suggestion: float  # When in script to place this fact

@dataclass
class FactIntegrationContext:
    anime_info: AnimeInfo
    scene: Scene
    moment_classification: MomentClassification
    script_style: str
    available_trivia: List[AnimeTrivia]
    scene_context: Dict[str, Any]

class FactIntegrator(LoggerMixin):
    """Integrates facts naturally into scripts"""
    
    def __init__(self, fact_templates_path: str = None):
        self.integration_templates = self._load_integration_templates(fact_templates_path)
        self.relevance_weights = self._setup_relevance_weights()
        self.style_preferences = self._setup_style_preferences()
        
        self.logger.info("Fact Integrator initialized")
    
    def integrate_facts(self, context: FactIntegrationContext, 
                       target_fact_count: int = 2) -> List[IntegratedFact]:
        """Intelligently integrate facts into script context"""
        
        # Score and rank available facts
        scored_facts = self._score_fact_relevance(context)
        
        # Select best facts for integration
        selected_facts = self._select_facts_for_integration(
            scored_facts, context, target_fact_count
        )
        
        # Determine integration methods for each fact
        integrated_facts = []
        for fact, relevance_score in selected_facts:
            integrated_fact = self._create_integrated_fact(fact, context, relevance_score)
            integrated_facts.append(integrated_fact)
        
        # Optimize timing and placement
        integrated_facts = self._optimize_fact_placement(integrated_facts, context)
        
        return integrated_facts
    
    def weave_facts_into_script(self, script_segments: List[Dict[str, Any]], 
                               integrated_facts: List[IntegratedFact]) -> List[Dict[str, Any]]:
        """Weave integrated facts into existing script segments"""
        
        enhanced_segments = script_segments.copy()
        
        for fact in integrated_facts:
            if fact.integration_method == 'natural':
                enhanced_segments = self._integrate_naturally(enhanced_segments, fact)
            elif fact.integration_method == 'transition':
                enhanced_segments = self._integrate_as_transition(enhanced_segments, fact)
            elif fact.integration_method == 'standalone':
                enhanced_segments = self._add_standalone_fact(enhanced_segments, fact)
            elif fact.integration_method == 'parenthetical':
                enhanced_segments = self._integrate_parenthetically(enhanced_segments, fact)
        
        return enhanced_segments
    
    def _score_fact_relevance(self, context: FactIntegrationContext) -> List[Tuple[AnimeTrivia, float]]:
        """Score facts based on relevance to current context"""
        scored_facts = []
        
        for trivia in context.available_trivia:
            score = 0.0
            
            # Content relevance scoring
            score += self._score_content_relevance(trivia, context)
            
            # Moment type relevance
            score += self._score_moment_relevance(trivia, context.moment_classification)
            
            # Style compatibility
            score += self._score_style_compatibility(trivia, context.script_style)
            
            # Timing appropriateness
            score += self._score_timing_appropriateness(trivia, context.scene)
            
            # Source credibility
            score += self._score_source_credibility(trivia)
            
            scored_facts.append((trivia, score))
        
        # Sort by score
        return sorted(scored_facts, key=lambda x: x[1], reverse=True)
    
    def _score_content_relevance(self, trivia: AnimeTrivia, context: FactIntegrationContext) -> float:
        """Score how relevant the trivia content is to the current scene"""
        score = 0.0
        
        # Check if trivia type matches scene context
        relevance_map = {
            'production': 0.8,     # Always interesting
            'cultural': 0.6,       # Good for context
            'reference': 0.7,      # Great for engagement
            'easter_egg': 0.5,     # Fun but not always relevant
            'voice_acting': 0.4,   # Scene dependent
            'animation': 0.9       # Highly relevant for visual content
        }
        
        score += relevance_map.get(trivia.trivia_type, 0.3)
        
        # Check content keywords against scene context
        content_lower = trivia.content.lower()
        scene_keywords = self._extract_scene_keywords(context)
        
        keyword_matches = sum(1 for keyword in scene_keywords if keyword in content_lower)
        score += min(keyword_matches * 0.2, 0.6)  # Max 0.6 bonus for keyword matches
        
        return score
    
    def _score_moment_relevance(self, trivia: AnimeTrivia, classification: MomentClassification) -> float:
        """Score relevance based on moment type"""
        moment_type = classification.primary_type
        
        # Trivia type preferences for different moment types
        preferences = {
            MomentType.ACTION_SEQUENCE: {
                'animation': 0.9,
                'production': 0.7,
                'reference': 0.6
            },
            MomentType.EMOTIONAL_MOMENT: {
                'cultural': 0.8,
                'voice_acting': 0.7,
                'production': 0.6
            },
            MomentType.TRANSFORMATION_SCENE: {
                'animation': 0.9,
                'production': 0.8,
                'reference': 0.7
            },
            MomentType.COMEDY_SCENE: {
                'easter_egg': 0.8,
                'reference': 0.7,
                'voice_acting': 0.6
            }
        }
        
        moment_prefs = preferences.get(moment_type, {})
        return moment_prefs.get(trivia.trivia_type, 0.3)
    
    def _score_style_compatibility(self, trivia: AnimeTrivia, script_style: str) -> float:
        """Score how well trivia fits the script style"""
        style_preferences = {
            'analytical': {
                'production': 0.9,
                'animation': 0.8,
                'cultural': 0.6
            },
            'trivia_focused': {
                'production': 0.9,
                'easter_egg': 0.9,
                'reference': 0.8,
                'voice_acting': 0.7
            },
            'enthusiastic': {
                'easter_egg': 0.8,
                'reference': 0.7,
                'animation': 0.6
            },
            'educational': {
                'cultural': 0.9,
                'production': 0.8,
                'animation': 0.7
            }
        }
        
        style_prefs = style_preferences.get(script_style, {})
        return style_prefs.get(trivia.trivia_type, 0.4)
    
    def _score_timing_appropriateness(self, trivia: AnimeTrivia, scene: Scene) -> float:
        """Score timing appropriateness for the scene duration"""
        # Longer facts work better in longer scenes
        fact_length = len(trivia.content.split())
        
        if scene.duration < 5:  # Short scene
            return 0.8 if fact_length < 15 else 0.3
        elif scene.duration < 10:  # Medium scene
            return 0.7 if fact_length < 25 else 0.4
        else:  # Long scene
            return 0.6  # Any length fact can work
    
    def _score_source_credibility(self, trivia: AnimeTrivia) -> float:
        """Score based on source credibility"""
        if trivia.verified:
            return 0.3
        
        credible_sources = ['official', 'interview', 'production', 'creator', 'studio']
        source_lower = trivia.source.lower()
        
        if any(term in source_lower for term in credible_sources):
            return 0.2
        
        return 0.1
    
    def _select_facts_for_integration(self, scored_facts: List[Tuple[AnimeTrivia, float]], 
                                    context: FactIntegrationContext, 
                                    target_count: int) -> List[Tuple[AnimeTrivia, float]]:
        """Select the best facts for integration avoiding redundancy"""
        
        if not scored_facts:
            return []
        
        selected = []
        used_types = set()
        
        for fact, score in scored_facts:
            if len(selected) >= target_count:
                break
            
            # Avoid too many facts of the same type
            if fact.trivia_type in used_types and len(selected) > 0:
                continue
            
            # Minimum score threshold
            if score < 0.4:
                continue
            
            selected.append((fact, score))
            used_types.add(fact.trivia_type)
        
        return selected
    
    def _create_integrated_fact(self, trivia: AnimeTrivia, context: FactIntegrationContext, 
                              relevance_score: float) -> IntegratedFact:
        """Create an integrated fact with appropriate formatting"""
        
        # Determine relevance level
        if relevance_score > 0.8:
            relevance = FactRelevance.HIGH
        elif relevance_score > 0.6:
            relevance = FactRelevance.MEDIUM
        elif relevance_score > 0.4:
            relevance = FactRelevance.LOW
        else:
            relevance = FactRelevance.CONTEXTUAL
        
        # Select integration method based on style and relevance
        integration_method = self._select_integration_method(context.script_style, relevance, trivia)
        
        # Format the fact text
        formatted_text = self._format_fact_text(trivia, integration_method, context.script_style)
        
        # Suggest timing
        timing = self._suggest_fact_timing(trivia, context.scene, integration_method)
        
        return IntegratedFact(
            fact_text=formatted_text,
            relevance=relevance,
            integration_method=integration_method,
            confidence=relevance_score,
            source=trivia.source,
            timing_suggestion=timing
        )
    
    def _select_integration_method(self, script_style: str, relevance: FactRelevance, 
                                 trivia: AnimeTrivia) -> str:
        """Select the best integration method"""
        
        # Style-based preferences
        style_methods = {
            'trivia_focused': ['standalone', 'natural', 'transition'],
            'analytical': ['natural', 'parenthetical', 'transition'],
            'enthusiastic': ['natural', 'standalone', 'parenthetical'],
            'educational': ['natural', 'transition', 'standalone']
        }
        
        preferred_methods = style_methods.get(script_style, ['natural', 'transition'])
        
        # Relevance-based adjustments
        if relevance == FactRelevance.HIGH:
            return 'natural'  # High relevance facts should be naturally integrated
        elif relevance == FactRelevance.LOW:
            return 'standalone'  # Low relevance facts work as standalone mentions
        
        return random.choice(preferred_methods)
    
    def _format_fact_text(self, trivia: AnimeTrivia, integration_method: str, style: str) -> str:
        """Format fact text based on integration method and style"""
        
        base_text = trivia.content
        
        if integration_method == 'standalone':
            starters = self.integration_templates['standalone_starters'][style]
            starter = random.choice(starters)
            return f"{starter} {base_text}"
        
        elif integration_method == 'transition':
            transitions = self.integration_templates['transition_phrases'][style]
            transition = random.choice(transitions)
            return f"{transition} {base_text}"
        
        elif integration_method == 'parenthetical':
            return f"({base_text})"
        
        else:  # natural
            return base_text
    
    def _suggest_fact_timing(self, trivia: AnimeTrivia, scene: Scene, integration_method: str) -> float:
        """Suggest when in the script to place this fact"""
        
        if integration_method == 'standalone':
            # Standalone facts work well in the middle or end
            return random.uniform(scene.duration * 0.4, scene.duration * 0.8)
        
        elif integration_method == 'transition':
            # Transition facts work well between main points
            return random.uniform(scene.duration * 0.3, scene.duration * 0.6)
        
        elif integration_method == 'parenthetical':
            # Parenthetical facts can go anywhere
            return random.uniform(scene.duration * 0.2, scene.duration * 0.9)
        
        else:  # natural
            # Natural integration depends on content flow
            return random.uniform(scene.duration * 0.2, scene.duration * 0.7)
    
    def _integrate_naturally(self, segments: List[Dict[str, Any]], fact: IntegratedFact) -> List[Dict[str, Any]]:
        """Integrate fact naturally into existing content"""
        
        # Find the best segment to enhance
        target_segment_idx = self._find_best_integration_point(segments, fact)
        
        if target_segment_idx is not None:
            original_text = segments[target_segment_idx]['text']
            enhanced_text = f"{original_text} {fact.fact_text}"
            segments[target_segment_idx]['text'] = enhanced_text
        
        return segments
    
    def _integrate_as_transition(self, segments: List[Dict[str, Any]], fact: IntegratedFact) -> List[Dict[str, Any]]:
        """Add fact as a transition between segments"""
        
        # Find good transition point
        insert_idx = len(segments) // 2  # Middle by default
        
        new_segment = {
            'text': fact.fact_text,
            'timing': fact.timing_suggestion,
            'emphasis': 'normal',
            'type': 'fact_transition'
        }
        
        segments.insert(insert_idx, new_segment)
        return segments
    
    def _add_standalone_fact(self, segments: List[Dict[str, Any]], fact: IntegratedFact) -> List[Dict[str, Any]]:
        """Add fact as a standalone segment"""
        
        new_segment = {
            'text': fact.fact_text,
            'timing': fact.timing_suggestion,
            'emphasis': 'excited' if fact.relevance == FactRelevance.HIGH else 'normal',
            'type': 'standalone_fact'
        }
        
        segments.append(new_segment)
        return segments
    
    def _integrate_parenthetically(self, segments: List[Dict[str, Any]], fact: IntegratedFact) -> List[Dict[str, Any]]:
        """Integrate fact parenthetically into existing content"""
        
        target_idx = self._find_best_integration_point(segments, fact)
        
        if target_idx is not None:
            original_text = segments[target_idx]['text']
            # Insert parenthetical at a natural break
            sentences = original_text.split('. ')
            if len(sentences) > 1:
                insert_point = len(sentences) // 2
                sentences.insert(insert_point, fact.fact_text)
                segments[target_idx]['text'] = '. '.join(sentences)
        
        return segments
    
    def _find_best_integration_point(self, segments: List[Dict[str, Any]], fact: IntegratedFact) -> Optional[int]:
        """Find the best segment to integrate a fact into"""
        
        if not segments:
            return None
        
        # Look for segments around the suggested timing
        target_timing = fact.timing_suggestion
        
        best_idx = 0
        min_time_diff = float('inf')
        
        for i, segment in enumerate(segments):
            time_diff = abs(segment.get('timing', 0) - target_timing)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                best_idx = i
        
        return best_idx
    
    def _extract_scene_keywords(self, context: FactIntegrationContext) -> List[str]:
        """Extract relevant keywords from scene context"""
        keywords = []
        
        # Add moment type as keyword
        keywords.append(context.moment_classification.primary_type.value)
        
        # Add anime info keywords
        if context.anime_info:
            keywords.extend([genre.lower() for genre in context.anime_info.genres])
            keywords.extend([theme.lower() for theme in context.anime_info.themes])
        
        # Add scene context keywords
        for key, value in context.scene_context.items():
            if isinstance(value, str):
                keywords.append(value.lower())
            elif isinstance(value, list):
                keywords.extend([str(v).lower() for v in value])
        
        return keywords
    
    def _optimize_fact_placement(self, facts: List[IntegratedFact], 
                                context: FactIntegrationContext) -> List[IntegratedFact]:
        """Optimize the timing and order of facts"""
        
        # Sort facts by suggested timing
        facts.sort(key=lambda f: f.timing_suggestion)
        
        # Ensure minimum spacing between facts
        min_spacing = 3.0  # 3 seconds minimum between facts
        
        for i in range(1, len(facts)):
            prev_timing = facts[i-1].timing_suggestion
            current_timing = facts[i].timing_suggestion
            
            if current_timing - prev_timing < min_spacing:
                facts[i].timing_suggestion = prev_timing + min_spacing
        
        return facts
    
    def _load_integration_templates(self, templates_path: Optional[str]) -> Dict[str, Any]:
        """Load or create integration templates"""
        
        if templates_path and Path(templates_path).exists():
            with open(templates_path, 'r') as f:
                return json.load(f)
        
        # Default templates
        return {
            'standalone_starters': {
                'trivia_focused': [
                    "Fun fact:", "Did you know:", "Here's something cool:",
                    "Trivia time:", "Cool detail:", "Interesting fact:"
                ],
                'analytical': [
                    "It's worth noting that", "Additionally,", "Interestingly,",
                    "From a production standpoint,", "Technically speaking,"
                ],
                'enthusiastic': [
                    "Oh, and get this!", "This is so cool -", "Amazing fact:",
                    "You'll love this -", "This blew my mind:"
                ],
                'educational': [
                    "For context,", "To understand this better,", "Historically,",
                    "From an educational perspective,", "To explain this:"
                ]
            },
            'transition_phrases': {
                'trivia_focused': [
                    "Speaking of trivia,", "Another fun fact -", "On that note,",
                    "Here's more cool info:", "Additional trivia:"
                ],
                'analytical': [
                    "This relates to", "Building on that,", "In connection with this,",
                    "Furthermore,", "This also demonstrates"
                ],
                'enthusiastic': [
                    "And here's something even cooler!", "But wait, there's more!",
                    "This gets better!", "Oh, and this is amazing too!"
                ],
                'educational': [
                    "This connects to", "In the broader context,", "This also teaches us",
                    "Related to this concept,", "This principle also applies to"
                ]
            }
        }
    
    def _setup_relevance_weights(self) -> Dict[str, float]:
        """Setup weights for relevance scoring"""
        return {
            'content_relevance': 0.3,
            'moment_relevance': 0.25,
            'style_compatibility': 0.2,
            'timing_appropriateness': 0.15,
            'source_credibility': 0.1
        }
    
    def _setup_style_preferences(self) -> Dict[str, List[str]]:
        """Setup style preferences for different integration methods"""
        return {
            'analytical': ['natural', 'parenthetical', 'transition'],
            'trivia_focused': ['standalone', 'natural', 'transition'],
            'enthusiastic': ['natural', 'standalone'],
            'mysterious': ['parenthetical', 'natural'],
            'comparison': ['transition', 'natural'],
            'educational': ['natural', 'transition', 'standalone'],
            'comedic': ['parenthetical', 'standalone'],
            'dramatic': ['natural', 'transition']
        }
    
    def get_integration_statistics(self, integrated_facts: List[IntegratedFact]) -> Dict[str, Any]:
        """Get statistics about fact integration"""
        if not integrated_facts:
            return {}
        
        stats = {
            'total_facts': len(integrated_facts),
            'relevance_distribution': {},
            'method_distribution': {},
            'average_confidence': sum(f.confidence for f in integrated_facts) / len(integrated_facts),
            'timing_spread': {
                'min': min(f.timing_suggestion for f in integrated_facts),
                'max': max(f.timing_suggestion for f in integrated_facts),
                'average': sum(f.timing_suggestion for f in integrated_facts) / len(integrated_facts)
            }
        }
        
        # Count relevance levels
        for fact in integrated_facts:
            rel_level = fact.relevance.value
            stats['relevance_distribution'][rel_level] = stats['relevance_distribution'].get(rel_level, 0) + 1
        
        # Count integration methods
        for fact in integrated_facts:
            method = fact.integration_method
            stats['method_distribution'][method] = stats['method_distribution'].get(method, 0) + 1
        
        return stats