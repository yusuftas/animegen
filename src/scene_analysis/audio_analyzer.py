"""
Audio Analysis Module
Analyzes audio tracks to detect peaks, events, and interesting moments
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available, using fallback audio analysis")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub not available, audio analysis will be limited")

@dataclass
class AudioPeak:
    timestamp: float
    intensity: float
    peak_type: str  # 'volume', 'onset', 'spectral'
    confidence: float
    metadata: Dict[str, Any]

class AudioAnalyzer:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
    def analyze_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Comprehensive audio analysis of a file"""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if LIBROSA_AVAILABLE:
            return self._analyze_with_librosa(audio_path)
        elif PYDUB_AVAILABLE:
            return self._analyze_with_pydub(audio_path)
        else:
            self.logger.error("No audio analysis libraries available")
            return {'peaks': [], 'metadata': {'method': 'none'}}
    
    def _analyze_with_librosa(self, audio_path: str) -> Dict[str, Any]:
        """Advanced audio analysis using librosa"""
        self.logger.info(f"Analyzing audio with librosa: {audio_path}")
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Detect onsets
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr, 
                units='time',
                hop_length=512,
                backtrack=True
            )
            
            # Detect tempo and beats
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
            
            # Analyze spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # RMS energy for volume analysis
            rms = librosa.feature.rms(y=y)[0]
            
            # Find volume peaks
            volume_peaks = self._find_volume_peaks(rms, sr)
            
            # Find spectral peaks (sudden changes in frequency content)
            spectral_peaks = self._find_spectral_peaks(spectral_centroids, sr)
            
            # Combine all peaks
            all_peaks = []
            
            # Add onset peaks
            for onset_time in onset_frames:
                peak = AudioPeak(
                    timestamp=float(onset_time),
                    intensity=self._get_intensity_at_time(rms, onset_time, sr),
                    peak_type='onset',
                    confidence=0.8,
                    metadata={'method': 'librosa_onset'}
                )
                all_peaks.append(peak)
            
            # Add volume peaks
            all_peaks.extend(volume_peaks)
            
            # Add spectral peaks
            all_peaks.extend(spectral_peaks)
            
            # Sort by timestamp
            all_peaks.sort(key=lambda x: x.timestamp)
            
            return {
                'peaks': all_peaks,
                'tempo': float(tempo),
                'beats': beats.tolist(),
                'duration': len(y) / sr,
                'metadata': {
                    'method': 'librosa',
                    'sample_rate': sr,
                    'total_peaks': len(all_peaks)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Librosa analysis failed: {e}")
            return self._analyze_with_pydub(audio_path)
    
    def _analyze_with_pydub(self, audio_path: str) -> Dict[str, Any]:
        """Basic audio analysis using pydub"""
        self.logger.info(f"Analyzing audio with pydub: {audio_path}")
        
        try:
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to mono and get raw data
            audio = audio.set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            
            # Normalize
            if len(samples) > 0:
                samples = samples / np.max(np.abs(samples))
            
            # Simple volume-based peak detection
            window_size = int(len(samples) * 0.1)  # 10% of audio length
            volume_peaks = []
            
            for i in range(0, len(samples) - window_size, window_size // 2):
                window = samples[i:i + window_size]
                volume = np.sqrt(np.mean(window**2))  # RMS
                
                # Simple threshold-based peak detection
                if volume > 0.3:  # Threshold
                    timestamp = i / audio.frame_rate
                    peak = AudioPeak(
                        timestamp=timestamp,
                        intensity=float(volume),
                        peak_type='volume',
                        confidence=0.6,
                        metadata={'method': 'pydub_simple'}
                    )
                    volume_peaks.append(peak)
            
            return {
                'peaks': volume_peaks,
                'duration': len(audio) / 1000.0,  # pydub uses milliseconds
                'metadata': {
                    'method': 'pydub',
                    'frame_rate': audio.frame_rate,
                    'total_peaks': len(volume_peaks)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Pydub analysis failed: {e}")
            return {'peaks': [], 'metadata': {'method': 'failed'}}
    
    def _find_volume_peaks(self, rms: np.ndarray, sr: int) -> List[AudioPeak]:
        """Find volume peaks in RMS energy"""
        # Convert RMS to time-based array
        hop_length = 512
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Find peaks using simple threshold and local maxima
        threshold = np.percentile(rms, 75)  # Top 25% of volume
        peaks = []
        
        for i in range(1, len(rms) - 1):
            if (rms[i] > threshold and 
                rms[i] > rms[i-1] and 
                rms[i] > rms[i+1]):
                
                peak = AudioPeak(
                    timestamp=float(times[i]),
                    intensity=float(rms[i]),
                    peak_type='volume',
                    confidence=0.7,
                    metadata={'method': 'rms_peaks'}
                )
                peaks.append(peak)
        
        return peaks
    
    def _find_spectral_peaks(self, spectral_centroids: np.ndarray, sr: int) -> List[AudioPeak]:
        """Find peaks in spectral content changes"""
        if not LIBROSA_AVAILABLE:
            return []
        
        hop_length = 512
        times = librosa.frames_to_time(np.arange(len(spectral_centroids)), sr=sr, hop_length=hop_length)
        
        # Find sudden changes in spectral centroid
        spectral_diff = np.abs(np.diff(spectral_centroids))
        threshold = np.percentile(spectral_diff, 80)  # Top 20% of changes
        
        peaks = []
        for i in range(len(spectral_diff)):
            if spectral_diff[i] > threshold:
                peak = AudioPeak(
                    timestamp=float(times[i]),
                    intensity=float(spectral_diff[i]),
                    peak_type='spectral',
                    confidence=0.6,
                    metadata={'method': 'spectral_centroid_diff'}
                )
                peaks.append(peak)
        
        return peaks
    
    def _get_intensity_at_time(self, rms: np.ndarray, time: float, sr: int) -> float:
        """Get RMS intensity at a specific time"""
        hop_length = 512
        frame_index = int(time * sr / hop_length)
        
        if 0 <= frame_index < len(rms):
            return float(rms[frame_index])
        return 0.0
    
    def extract_audio_from_video(self, video_path: str, output_path: str = None) -> str:
        """Extract audio track from video file"""
        if not PYDUB_AVAILABLE:
            raise ImportError("pydub is required for audio extraction")
        
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if output_path is None:
            output_path = str(Path(video_path).with_suffix('.wav'))
        
        try:
            # Extract audio using pydub
            video = AudioSegment.from_file(video_path)
            video.export(output_path, format="wav")
            
            self.logger.info(f"Audio extracted to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            raise
    
    def get_peaks_in_timerange(self, peaks: List[AudioPeak], start_time: float, end_time: float) -> List[AudioPeak]:
        """Filter peaks within a specific time range"""
        return [peak for peak in peaks if start_time <= peak.timestamp <= end_time]
    
    def calculate_audio_activity_score(self, peaks: List[AudioPeak], duration: float) -> float:
        """Calculate overall audio activity score"""
        if duration <= 0 or not peaks:
            return 0.0
        
        # Score based on peak density and intensity
        peak_density = len(peaks) / duration
        avg_intensity = np.mean([peak.intensity for peak in peaks])
        
        # Normalize scores
        density_score = min(peak_density / 2.0, 1.0)  # 2+ peaks per second = max
        intensity_score = min(avg_intensity, 1.0)
        
        return (density_score + intensity_score) / 2