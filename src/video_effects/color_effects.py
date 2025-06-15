"""
Color Effects Engine for Anime Video Processing

Provides dynamic color grading, lighting effects, chromatic aberration,
and other color-based visual enhancements for anime content.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging


# MoviePy 1.x imports (fallback)
from moviepy.editor import VideoFileClip, CompositeVideoClip
from moviepy.video.fx import speedx


logger = logging.getLogger(__name__)


class ColorEffectsEngine:
    """Engine for applying color and lighting effects to video clips."""
    
    def __init__(self):
        """Initialize the color effects engine."""
        self.color_profiles = {}
        self.load_default_profiles()
    
    def load_default_profiles(self):
        """Load default color profiles for different anime styles."""
        self.color_profiles = {
            'vibrant': {
                'saturation_boost': 1.3,
                'contrast_limit': 3.0,
                'brightness_boost': 1.1
            },
            'sunset': {
                'color_matrix': np.array([
                    [1.2, 0.1, 0.0],
                    [0.1, 1.0, 0.0],
                    [0.0, 0.0, 0.8]
                ])
            },
            'moonlight': {
                'color_matrix': np.array([
                    [0.8, 0.0, 0.1],
                    [0.0, 0.9, 0.1],
                    [0.1, 0.1, 1.3]
                ])
            },
            'dramatic': {
                'saturation_boost': 1.5,
                'contrast_limit': 4.0,
                'gamma': 0.8
            }
        }
    
    def anime_color_grade(self, frame: np.ndarray, style: str = "vibrant") -> np.ndarray:
        """
        Apply anime-style color grading to a frame.
        
        Args:
            frame: Input frame as numpy array
            style: Color grading style ("vibrant", "sunset", "moonlight", "dramatic")
            
        Returns:
            Color-graded frame
        """
        try:
            if style == "vibrant":
                # Enhance colors using LAB color space
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
                profile = self.color_profiles['vibrant']
                clahe = cv2.createCLAHE(clipLimit=profile['contrast_limit'], tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # Enhance A and B channels (color information)
                a = cv2.multiply(a, profile['saturation_boost'])
                b = cv2.multiply(b, profile['saturation_boost'])
                
                # Merge channels and convert back
                enhanced = cv2.merge([l, a, b])
                result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                
                # Apply brightness boost
                result = cv2.multiply(result, profile['brightness_boost'])
                result = np.clip(result, 0, 255).astype(np.uint8)
                
                return result
                
            elif style in ["sunset", "moonlight"]:
                # Apply color matrix transformation
                profile = self.color_profiles[style]
                frame_float = frame.astype(np.float32) / 255.0
                color_matrix = profile['color_matrix']
                
                # Apply color transformation
                result_float = np.zeros_like(frame_float)
                for i in range(3):
                    result_float[:, :, i] = np.clip(
                        frame_float[:, :, 0] * color_matrix[i, 0] +
                        frame_float[:, :, 1] * color_matrix[i, 1] +
                        frame_float[:, :, 2] * color_matrix[i, 2], 0, 1
                    )
                
                return (result_float * 255).astype(np.uint8)
                
            elif style == "dramatic":
                # High contrast, high saturation dramatic look
                profile = self.color_profiles['dramatic']
                
                # Convert to HSV for saturation control
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                
                # Boost saturation
                s = cv2.multiply(s, profile['saturation_boost'])
                s = np.clip(s, 0, 255)
                
                # Apply contrast to value channel
                clahe = cv2.createCLAHE(clipLimit=profile['contrast_limit'], tileGridSize=(8, 8))
                v = clahe.apply(v)
                
                # Merge and convert back
                enhanced_hsv = cv2.merge([h, s, v])
                result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
                
                # Apply gamma correction
                gamma = profile['gamma']
                result = np.power(result / 255.0, gamma) * 255
                result = np.clip(result, 0, 255).astype(np.uint8)
                
                return result
                
        except Exception as e:
            logger.error(f"Error applying color grading: {e}")
            return frame
    
    def apply_color_grading(self, clip: VideoFileClip, style: str = "vibrant") -> VideoFileClip:
        """
        Apply color grading to an entire video clip.
        
        Args:
            clip: Input video clip
            style: Color grading style
            
        Returns:
            Color-graded video clip
        """
        try:
            result = clip.fl_image(lambda img: self.anime_color_grade(img, style))
            logger.info(f"Applied {style} color grading to clip")
            return result
            
        except Exception as e:
            logger.error(f"Error applying color grading to clip: {e}")
            return clip
    
    def chromatic_aberration_effect(self, frame: np.ndarray, intensity: int = 5) -> np.ndarray:
        """
        Apply chromatic aberration effect for retro/glitch aesthetics.
        
        Args:
            frame: Input frame
            intensity: Intensity of the aberration effect
            
        Returns:
            Frame with chromatic aberration
        """
        try:
            h, w = frame.shape[:2]
            
            # Split color channels
            b, g, r = cv2.split(frame)
            
            # Create offset matrices
            offset_r = np.float32([[1, 0, intensity], [0, 1, 0]])
            offset_b = np.float32([[1, 0, -intensity], [0, 1, 0]])
            
            # Apply offsets to red and blue channels
            r_shifted = cv2.warpAffine(r, offset_r, (w, h), borderMode=cv2.BORDER_REFLECT)
            b_shifted = cv2.warpAffine(b, offset_b, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # Recombine channels
            result = cv2.merge([b_shifted, g, r_shifted])
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying chromatic aberration: {e}")
            return frame
    
    def vintage_vhs_effect(self, frame: np.ndarray) -> np.ndarray:
        """
        Create VHS-style vintage effect with noise and scan lines.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with VHS-style effect
        """
        try:
            # Add random noise
            noise = np.random.randint(0, 25, frame.shape, dtype=np.uint8)
            noisy_frame = cv2.add(frame, noise)
            
            # Add scan lines
            h, w = frame.shape[:2]
            for y in range(0, h, 4):
                noisy_frame[y:y+1, :] = (noisy_frame[y:y+1, :] * 0.8).astype(np.uint8)
            
            # Apply chromatic aberration
            aberrated = self.chromatic_aberration_effect(noisy_frame, 3)
            
            # Slight blur for VHS quality degradation
            result = cv2.GaussianBlur(aberrated, (3, 3), 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying VHS effect: {e}")
            return frame
    
    def selective_color_highlight(self, frame: np.ndarray, target_color: Tuple[int, int, int],
                                tolerance: int = 30, boost_factor: float = 1.5) -> np.ndarray:
        """
        Selectively highlight specific colors while desaturating others.
        
        Args:
            frame: Input frame
            target_color: RGB color to highlight
            tolerance: Color tolerance for selection
            boost_factor: How much to boost the selected color
            
        Returns:
            Frame with selective color highlighting
        """
        try:
            # Convert to HSV for better color selection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Convert target color to HSV
            target_bgr = np.uint8([[target_color[::-1]]])  # Convert RGB to BGR
            target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]
            
            # Create mask for target color
            lower_bound = np.array([max(0, target_hsv[0] - tolerance//2), 50, 50])
            upper_bound = np.array([min(179, target_hsv[0] + tolerance//2), 255, 255])
            
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Create desaturated version of the frame
            desaturated = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            desaturated = cv2.cvtColor(desaturated, cv2.COLOR_GRAY2BGR)
            
            # Boost the selected color areas
            boosted_frame = frame.copy()
            hsv_boosted = cv2.cvtColor(boosted_frame, cv2.COLOR_BGR2HSV)
            hsv_boosted[mask > 0, 1] = np.clip(hsv_boosted[mask > 0, 1] * boost_factor, 0, 255)
            boosted_frame = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)
            
            # Combine desaturated background with boosted color areas
            mask_3d = cv2.merge([mask, mask, mask]) / 255.0
            result = desaturated * (1 - mask_3d) + boosted_frame * mask_3d
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error applying selective color highlight: {e}")
            return frame
    
    def dynamic_lighting_effect(self, frame: np.ndarray, light_source: Tuple[int, int],
                               intensity: float = 1.0, falloff: float = 2.0) -> np.ndarray:
        """
        Apply dynamic lighting effect with a specific light source.
        
        Args:
            frame: Input frame
            light_source: (x, y) position of light source
            intensity: Intensity of the light
            falloff: How quickly light falls off with distance
            
        Returns:
            Frame with dynamic lighting
        """
        try:
            h, w = frame.shape[:2]
            
            # Create distance map from light source
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            distances = np.sqrt((x_coords - light_source[0])**2 + (y_coords - light_source[1])**2)
            
            # Create lighting mask (inverse distance with falloff)
            max_distance = np.sqrt(w**2 + h**2)
            normalized_distances = distances / max_distance
            lighting_mask = intensity * (1.0 / (1.0 + falloff * normalized_distances))
            
            # Apply lighting to each channel
            result = frame.copy().astype(np.float32)
            for i in range(3):
                result[:, :, i] = result[:, :, i] * lighting_mask
            
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying dynamic lighting: {e}")
            return frame
    
    def color_temperature_shift(self, frame: np.ndarray, temperature: int = 6500) -> np.ndarray:
        """
        Shift color temperature of the frame.
        
        Args:
            frame: Input frame
            temperature: Color temperature in Kelvin (2000-10000)
            
        Returns:
            Frame with adjusted color temperature
        """
        try:
            # Clamp temperature to reasonable range
            temperature = max(2000, min(10000, temperature))
            
            # Calculate RGB multipliers based on temperature
            if temperature <= 6600:
                # Warm colors (red dominant)
                r_mult = 1.0
                g_mult = 0.39 * np.log(temperature / 100) - 0.63
                if temperature <= 2000:
                    b_mult = 0.0
                else:
                    b_mult = 0.543 * np.log(temperature / 100 - 10) - 1.19
            else:
                # Cool colors (blue dominant)
                r_mult = 1.292 * ((temperature / 100) ** -0.1332) - 0.213
                g_mult = 1.292 * ((temperature / 100) ** -0.0755) - 0.213
                b_mult = 1.0
            
            # Normalize multipliers
            multipliers = np.array([b_mult, g_mult, r_mult])  # BGR order
            multipliers = np.clip(multipliers, 0.3, 2.0)
            
            # Apply color temperature shift
            result = frame.astype(np.float32)
            for i in range(3):
                result[:, :, i] = result[:, :, i] * multipliers[i]
            
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying color temperature shift: {e}")
            return frame
    
    def create_vignette_effect(self, frame: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Add vignette effect to frame.
        
        Args:
            frame: Input frame
            strength: Strength of vignette effect (0.0 to 1.0)
            
        Returns:
            Frame with vignette effect
        """
        try:
            h, w = frame.shape[:2]
            
            # Create vignette mask
            center_x, center_y = w // 2, h // 2
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            
            # Calculate distance from center
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            # Create vignette mask
            normalized_distances = distances / max_distance
            vignette_mask = 1.0 - (strength * normalized_distances**2)
            vignette_mask = np.clip(vignette_mask, 0.0, 1.0)
            
            # Apply vignette
            result = frame.astype(np.float32)
            for i in range(3):
                result[:, :, i] = result[:, :, i] * vignette_mask
            
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating vignette effect: {e}")
            return frame
    
    def bloom_effect(self, frame: np.ndarray, threshold: int = 200, blur_size: int = 15) -> np.ndarray:
        """
        Add bloom effect to bright areas of the frame.
        
        Args:
            frame: Input frame
            threshold: Brightness threshold for bloom
            blur_size: Size of bloom blur
            
        Returns:
            Frame with bloom effect
        """
        try:
            # Convert to grayscale for brightness detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Create mask for bright areas
            _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Create bloom overlay
            bloom_overlay = cv2.bitwise_and(frame, frame, mask=bright_mask)
            
            # Apply strong gaussian blur for bloom effect
            bloom_blur = cv2.GaussianBlur(bloom_overlay, (blur_size, blur_size), 0)
            
            # Blend with original frame
            result = cv2.addWeighted(frame, 1.0, bloom_blur, 0.3, 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying bloom effect: {e}")
            return frame
    
    def apply_multiple_color_effects(self, clip: VideoFileClip, effects_config: Dict[str, Any]) -> VideoFileClip:
        """
        Apply multiple color effects to a video clip.
        
        Args:
            clip: Input video clip
            effects_config: Dictionary of effects to apply
            
        Returns:
            Video clip with color effects applied
        """
        try:
            def apply_effects(frame):
                result = frame
                
                if 'color_grade' in effects_config:
                    style = effects_config['color_grade'].get('style', 'vibrant')
                    result = self.anime_color_grade(result, style)
                
                if 'chromatic_aberration' in effects_config:
                    intensity = effects_config['chromatic_aberration'].get('intensity', 5)
                    result = self.chromatic_aberration_effect(result, intensity)
                
                if 'vintage_vhs' in effects_config and effects_config['vintage_vhs']:
                    result = self.vintage_vhs_effect(result)
                
                if 'selective_color' in effects_config:
                    config = effects_config['selective_color']
                    result = self.selective_color_highlight(result, **config)
                
                if 'color_temperature' in effects_config:
                    temp = effects_config['color_temperature'].get('temperature', 6500)
                    result = self.color_temperature_shift(result, temp)
                
                if 'vignette' in effects_config:
                    strength = effects_config['vignette'].get('strength', 0.5)
                    result = self.create_vignette_effect(result, strength)
                
                if 'bloom' in effects_config:
                    config = effects_config['bloom']
                    result = self.bloom_effect(result, **config)
                
                return result
            
            result_clip = clip.fl_image(apply_effects)
            logger.info(f"Applied {len(effects_config)} color effects to clip")
            return result_clip
            
        except Exception as e:
            logger.error(f"Error applying multiple color effects: {e}")
            return clip
    
    def get_color_effect_presets(self) -> Dict[str, Any]:
        """
        Get predefined color effect presets for different moods.
        
        Returns:
            Dictionary of preset configurations
        """
        return {
            'action_scene': {
                'color_grade': {'style': 'dramatic'},
                'selective_color': {'target_color': (255, 0, 0), 'tolerance': 40},
                'vignette': {'strength': 0.3}
            },
            'emotional_scene': {
                'color_grade': {'style': 'sunset'},
                'bloom': {'threshold': 180, 'blur_size': 21},
                'color_temperature': {'temperature': 3200}
            },
            'night_scene': {
                'color_grade': {'style': 'moonlight'},
                'vignette': {'strength': 0.6},
                'color_temperature': {'temperature': 8000}
            },
            'retro_style': {
                'vintage_vhs': True,
                'chromatic_aberration': {'intensity': 8},
                'color_temperature': {'temperature': 4500}
            },
            'fantasy_magical': {
                'color_grade': {'style': 'vibrant'},
                'selective_color': {'target_color': (255, 0, 255), 'tolerance': 50},
                'bloom': {'threshold': 160, 'blur_size': 25}
            }
        }