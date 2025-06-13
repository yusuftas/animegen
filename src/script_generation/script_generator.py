"""
Script Generation System
Generates dynamic, engaging scripts for anime shorts with multiple styles
"""

import random
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

try:
    from ..scene_analysis.scene_extractor import Scene
    from ..scene_analysis.interest_detector import InterestScore
    from ..content_intelligence.moment_classifier import MomentClassification, MomentType
    from ..content_intelligence.anime_knowledge_base import AnimeInfo, AnimeTrivia
    from ..content_intelligence.content_matcher import ContentMatch
    from ..utils.logger import LoggerMixin
except ImportError:
    from scene_analysis.scene_extractor import Scene
    from scene_analysis.interest_detector import InterestScore
    from content_intelligence.moment_classifier import MomentClassification, MomentType
    from content_intelligence.anime_knowledge_base import AnimeInfo, AnimeTrivia
    from content_intelligence.content_matcher import ContentMatch
    from utils.logger import LoggerMixin

class ScriptStyle(Enum):
    ANALYTICAL = "analytical"
    TRIVIA_FOCUSED = "trivia_focused"
    ENTHUSIASTIC = "enthusiastic"
    MYSTERIOUS = "mysterious"
    COMPARISON = "comparison"
    EDUCATIONAL = "educational"
    COMEDIC = "comedic"
    DRAMATIC = "dramatic"

@dataclass
class ScriptSegment:
    text: str
    timing: float  # When to say this relative to scene start
    emphasis: str  # 'normal', 'excited', 'whisper', 'dramatic'
    pause_after: float = 0.0  # Pause duration after this segment

@dataclass
class GeneratedScript:
    segments: List[ScriptSegment]
    style: ScriptStyle
    total_duration: float
    word_count: int
    metadata: Dict[str, Any]

class ScriptGenerator(LoggerMixin):
    """Generates dynamic scripts for anime shorts"""
    
    def __init__(self, styles_config_path: str = None):
        self.styles_database = self._load_styles_database(styles_config_path)
        self.fact_templates = self._load_fact_templates()
        self.transition_phrases = self._load_transition_phrases()
        self.hook_generators = self._create_hook_generators()
        
        self.logger.info("Script Generator initialized")
    
    def generate_script(self, scene: Scene, classification: MomentClassification,
                       interest_score: InterestScore, anime_info: AnimeInfo,
                       content_matches: List[ContentMatch] = None,
                       trivia: List[AnimeTrivia] = None,
                       style: ScriptStyle = None,
                       target_duration: float = None) -> GeneratedScript:
        """Generate a complete script for a scene"""
        
        # Auto-select style if not provided
        if style is None:
            style = self._select_optimal_style(classification, anime_info)
        
        # Calculate target duration
        if target_duration is None:
            target_duration = min(scene.duration * 0.8, 25.0)  # 80% of scene or max 25s
        
        # Generate script segments
        segments = []
        
        # 1. Opening hook (0-3 seconds)
        hook = self._generate_opening_hook(style, classification, anime_info)
        segments.append(ScriptSegment(hook, 0.0, 'normal', 0.5))
        
        # 2. Main content (3-20 seconds)
        main_content = self._generate_main_content(
            style, scene, classification, interest_score, anime_info, content_matches, trivia
        )
        segments.extend(main_content)
        
        # 3. Closing/CTA (20-25 seconds)
        closing = self._generate_closing(style, classification)
        segments.append(ScriptSegment(closing, target_duration - 3.0, 'normal', 0.0))
        
        # Adjust timing to fit target duration
        segments = self._adjust_timing(segments, target_duration)
        
        # Calculate metadata
        full_text = " ".join([seg.text for seg in segments])
        word_count = len(full_text.split())
        
        script = GeneratedScript(
            segments=segments,
            style=style,
            total_duration=target_duration,
            word_count=word_count,
            metadata={
                'scene_type': classification.primary_type.value,
                'anime_title': anime_info.title,
                'confidence': classification.confidence,
                'has_trivia': len(trivia) > 0 if trivia else False,
                'has_matches': len(content_matches) > 0 if content_matches else False
            }
        )
        
        return script
    
    def _select_optimal_style(self, classification: MomentClassification, anime_info: AnimeInfo) -> ScriptStyle:
        """Select the best style for the given scene and anime"""
        
        # Style preferences based on moment type
        type_preferences = {
            MomentType.ACTION_SEQUENCE: [ScriptStyle.ENTHUSIASTIC, ScriptStyle.ANALYTICAL],
            MomentType.EMOTIONAL_MOMENT: [ScriptStyle.DRAMATIC, ScriptStyle.ANALYTICAL],
            MomentType.COMEDY_SCENE: [ScriptStyle.COMEDIC, ScriptStyle.ENTHUSIASTIC],
            MomentType.FIGHT_SCENE: [ScriptStyle.ENTHUSIASTIC, ScriptStyle.COMPARISON],
            MomentType.TRANSFORMATION_SCENE: [ScriptStyle.ENTHUSIASTIC, ScriptStyle.TRIVIA_FOCUSED],
            MomentType.PLOT_REVELATION: [ScriptStyle.MYSTERIOUS, ScriptStyle.ANALYTICAL],
            MomentType.CHARACTER_INTRODUCTION: [ScriptStyle.EDUCATIONAL, ScriptStyle.COMPARISON]
        }
        
        preferred_styles = type_preferences.get(classification.primary_type, [ScriptStyle.ANALYTICAL])
        
        # Consider anime genres
        if anime_info.genres:
            genre_modifiers = {
                'Comedy': ScriptStyle.COMEDIC,
                'Drama': ScriptStyle.DRAMATIC,
                'Action': ScriptStyle.ENTHUSIASTIC,
                'Mystery': ScriptStyle.MYSTERIOUS,
                'Educational': ScriptStyle.EDUCATIONAL
            }
            
            for genre in anime_info.genres:
                if genre in genre_modifiers:
                    preferred_styles.insert(0, genre_modifiers[genre])
        
        # Add some randomness to avoid repetition
        if random.random() < 0.3:  # 30% chance of random style
            return random.choice(list(ScriptStyle))
        
        return preferred_styles[0]
    
    def _generate_opening_hook(self, style: ScriptStyle, classification: MomentClassification, 
                              anime_info: AnimeInfo) -> str:
        """Generate an engaging opening hook"""
        
        hooks = self.styles_database[style.value]["opening_hooks"]
        
        # Select appropriate hook based on moment type
        moment_type = classification.primary_type.value
        
        # Context-aware hook selection
        context_hooks = {
            'action_sequence': [
                f"This {anime_info.title} fight scene is incredible because...",
                "Watch this amazing action sequence unfold...",
                "Here's why this battle scene stands out..."
            ],
            'emotional_moment': [
                f"This heartbreaking moment in {anime_info.title} hits different...",
                "Notice the subtle details in this emotional scene...",
                "This character development moment is masterfully crafted..."
            ],
            'transformation_scene': [
                f"This {anime_info.title} transformation is iconic because...",
                "The animation here is absolutely stunning...",
                "Watch how the animators handled this power-up..."
            ]
        }
        
        # Try to get context-specific hook first
        if moment_type in context_hooks:
            available_hooks = context_hooks[moment_type] + hooks
        else:
            available_hooks = hooks
        
        selected_hook = random.choice(available_hooks)
        
        # Personalize with anime title if placeholder exists
        return selected_hook.replace("{anime_title}", anime_info.title)
    
    def _generate_main_content(self, style: ScriptStyle, scene: Scene,
                              classification: MomentClassification, interest_score: InterestScore,
                              anime_info: AnimeInfo, content_matches: List[ContentMatch] = None,
                              trivia: List[AnimeTrivia] = None) -> List[ScriptSegment]:
        """Generate the main content segments"""
        
        segments = []
        current_time = 3.0  # Start after hook
        
        # 1. Scene analysis segment
        analysis = self._generate_scene_analysis(style, classification, interest_score, anime_info)
        segments.append(ScriptSegment(analysis, current_time, 'normal', 0.8))
        current_time += 5.0
        
        # 2. Add trivia if available
        if trivia and len(trivia) > 0:
            trivia_text = self._generate_trivia_segment(style, trivia[0], anime_info)
            segments.append(ScriptSegment(trivia_text, current_time, 'excited', 0.5))
            current_time += 4.0
        
        # 3. Add comparison if matches available
        if content_matches and len(content_matches) > 0:
            comparison = self._generate_comparison_segment(style, content_matches[0], anime_info)
            segments.append(ScriptSegment(comparison, current_time, 'normal', 0.3))
            current_time += 4.0
        
        # 4. Technical insight
        technical = self._generate_technical_insight(style, classification, anime_info)
        segments.append(ScriptSegment(technical, current_time, 'analytical', 0.5))
        
        return segments
    
    def _generate_scene_analysis(self, style: ScriptStyle, classification: MomentClassification,
                                interest_score: InterestScore, anime_info: AnimeInfo) -> str:
        """Generate analysis of the current scene"""
        
        moment_type = classification.primary_type.value
        confidence = classification.confidence
        
        analysis_templates = {
            ScriptStyle.ANALYTICAL: [
                f"The {moment_type} here showcases exceptional {self._get_standout_feature(interest_score)}.",
                f"Notice how the {moment_type} builds tension through {self._get_technique_description(interest_score)}.",
                f"This {moment_type} demonstrates {anime_info.studios[0] if anime_info.studios else 'the studio'}'s signature style."
            ],
            ScriptStyle.ENTHUSIASTIC: [
                f"This {moment_type} is absolutely incredible!",
                f"The way this {moment_type} unfolds is just *chef's kiss*!",
                f"Can we talk about how amazing this {moment_type} is?"
            ],
            ScriptStyle.TRIVIA_FOCUSED: [
                f"Fun fact about this {moment_type}...",
                f"Here's something most fans don't know about this scene...",
                f"The story behind this {moment_type} is fascinating..."
            ]
        }
        
        templates = analysis_templates.get(style, analysis_templates[ScriptStyle.ANALYTICAL])
        return random.choice(templates)
    
    def _generate_trivia_segment(self, style: ScriptStyle, trivia: AnimeTrivia, 
                               anime_info: AnimeInfo) -> str:
        """Generate trivia segment"""
        
        trivia_intros = [
            "Did you know",
            "Here's a cool fact",
            "Fun trivia",
            "Most fans don't realize",
            "Interestingly"
        ]
        
        intro = random.choice(trivia_intros)
        
        if style == ScriptStyle.ENTHUSIASTIC:
            return f"{intro} - {trivia.content}! How cool is that?"
        elif style == ScriptStyle.MYSTERIOUS:
            return f"{intro}... {trivia.content}. Makes you think, doesn't it?"
        else:
            return f"{intro}: {trivia.content}."
    
    def _generate_comparison_segment(self, style: ScriptStyle, match: ContentMatch, 
                                   anime_info: AnimeInfo) -> str:
        """Generate comparison with similar content"""
        
        comparison_templates = [
            f"This reminds me of {match.explanation}",
            f"Similar to {match.explanation}, but with a unique twist",
            f"You'll also see this pattern in {match.explanation}"
        ]
        
        return random.choice(comparison_templates)
    
    def _generate_technical_insight(self, style: ScriptStyle, classification: MomentClassification,
                                  anime_info: AnimeInfo) -> str:
        """Generate technical or production insight"""
        
        technical_aspects = [
            "animation technique",
            "directorial choice",
            "color palette",
            "sound design",
            "character expression",
            "camera work"
        ]
        
        aspect = random.choice(technical_aspects)
        
        if style == ScriptStyle.ANALYTICAL:
            return f"The {aspect} here is particularly noteworthy."
        elif style == ScriptStyle.EDUCATIONAL:
            return f"This demonstrates excellent {aspect} in anime production."
        else:
            return f"Notice the masterful {aspect} in this scene."
    
    def _generate_closing(self, style: ScriptStyle, classification: MomentClassification) -> str:
        """Generate closing segment with call-to-action"""
        
        closings = {
            ScriptStyle.ANALYTICAL: [
                "What do you think about this scene?",
                "Let me know your analysis in the comments!",
                "Drop your thoughts below!"
            ],
            ScriptStyle.ENTHUSIASTIC: [
                "Wasn't that amazing? Let me know what you think!",
                "That was incredible! What's your favorite part?",
                "So epic! Share your thoughts!"
            ],
            ScriptStyle.TRIVIA_FOCUSED: [
                "Know any other cool facts? Share them below!",
                "What other trivia should I cover?",
                "Drop more fun facts in the comments!"
            ]
        }
        
        style_closings = closings.get(style, closings[ScriptStyle.ANALYTICAL])
        return random.choice(style_closings)
    
    def _get_standout_feature(self, interest_score: InterestScore) -> str:
        """Identify the standout feature of the scene"""
        scores = {
            'motion': interest_score.motion_score,
            'character work': interest_score.face_score,
            'visual composition': interest_score.composition_score,
            'color work': interest_score.color_variance_score
        }
        
        best_feature = max(scores.keys(), key=lambda k: scores[k])
        return best_feature
    
    def _get_technique_description(self, interest_score: InterestScore) -> str:
        """Get description of the primary technique used"""
        techniques = [
            "dynamic camera movement",
            "masterful character animation",
            "brilliant color contrast",
            "strategic pacing",
            "emotional composition"
        ]
        
        return random.choice(techniques)
    
    def _adjust_timing(self, segments: List[ScriptSegment], target_duration: float) -> List[ScriptSegment]:
        """Adjust segment timing to fit target duration"""
        if not segments:
            return segments
        
        # Calculate current span
        last_segment_time = max(seg.timing for seg in segments)
        
        if last_segment_time > target_duration:
            # Compress timing
            scale_factor = (target_duration - 3.0) / last_segment_time
            for segment in segments[1:]:  # Keep first segment at 0.0
                segment.timing *= scale_factor
        
        return segments
    
    def _load_styles_database(self, config_path: Optional[str]) -> Dict[str, Dict[str, List[str]]]:
        """Load or create styles database"""
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default styles database
        return {
            "analytical": {
                "opening_hooks": [
                    "Here's something most fans missed in this scene...",
                    "The animation technique here is fascinating because...",
                    "Notice how the director chose to...",
                    "This scene demonstrates masterful...",
                    "The subtle details in this moment reveal..."
                ],
                "transition_phrases": [
                    "But that's not all...",
                    "Even more interesting is...",
                    "This connects to...",
                    "What's really clever here is...",
                    "Building on that..."
                ]
            },
            "trivia_focused": {
                "opening_hooks": [
                    "Did you know that in the manga...",
                    "Fun fact: The voice actor actually...",
                    "This scene took 3 months to animate because...",
                    "Here's a cool behind-the-scenes fact...",
                    "Most fans don't realize that..."
                ],
                "transition_phrases": [
                    "Another cool fact...",
                    "Speaking of trivia...",
                    "Here's another detail...",
                    "On that note...",
                    "Additionally..."
                ]
            },
            "enthusiastic": {
                "opening_hooks": [
                    "This scene is absolutely incredible!",
                    "Can we talk about how amazing this is?",
                    "I'm obsessed with this moment because...",
                    "This is why I love anime!",
                    "This scene gives me chills every time!"
                ],
                "transition_phrases": [
                    "And it gets even better!",
                    "But wait, there's more!",
                    "This is so good!",
                    "I can't even!",
                    "It's just perfect!"
                ]
            },
            "mysterious": {
                "opening_hooks": [
                    "There's a hidden meaning in this scene...",
                    "This moment holds a secret...",
                    "What if I told you this scene...",
                    "The true purpose of this scene is...",
                    "Hidden in plain sight is..."
                ],
                "transition_phrases": [
                    "But the mystery deepens...",
                    "That's not the only secret...",
                    "Things aren't what they seem...",
                    "The plot thickens...",
                    "There's more beneath the surface..."
                ]
            },
            "comparison": {
                "opening_hooks": [
                    "This scene reminds me of...",
                    "Unlike other anime that...",
                    "Compared to the manga version...",
                    "This is similar to... but different because...",
                    "While most anime do this, {anime_title}..."
                ],
                "transition_phrases": [
                    "In contrast...",
                    "Similarly...",
                    "On the other hand...",
                    "This differs because...",
                    "Like we saw before..."
                ]
            }
        }
    
    def _load_fact_templates(self) -> List[str]:
        """Load fact integration templates"""
        return [
            "Did you know {fact}?",
            "Fun fact: {fact}.",
            "Here's something interesting: {fact}.",
            "Most fans don't realize {fact}.",
            "Cool trivia: {fact}."
        ]
    
    def _load_transition_phrases(self) -> List[str]:
        """Load transition phrases for natural flow"""
        return [
            "Speaking of which,",
            "On that note,",
            "This reminds me,",
            "Additionally,",
            "What's more,",
            "Furthermore,",
            "Not only that,",
            "Beyond that,"
        ]
    
    def _create_hook_generators(self) -> Dict[str, List[str]]:
        """Create hook generators for different scenarios"""
        return {
            'high_motion': [
                "This action sequence is insane!",
                "The animation here is next level!",
                "Watch this incredible fight unfold!"
            ],
            'emotional': [
                "This moment hits different...",
                "Prepare for the feels...",
                "This scene is heartbreaking..."
            ],
            'technical': [
                "The animation technique here is brilliant...",
                "Notice the masterful direction...",
                "The art style in this scene..."
            ]
        }
    
    def generate_multiple_variations(self, scene: Scene, classification: MomentClassification,
                                   interest_score: InterestScore, anime_info: AnimeInfo,
                                   count: int = 3) -> List[GeneratedScript]:
        """Generate multiple script variations for A/B testing"""
        
        variations = []
        styles = list(ScriptStyle)
        
        for i in range(count):
            # Use different styles or add randomness
            if i < len(styles):
                style = styles[i]
            else:
                style = random.choice(styles)
            
            script = self.generate_script(scene, classification, interest_score, 
                                        anime_info, style=style)
            variations.append(script)
        
        return variations