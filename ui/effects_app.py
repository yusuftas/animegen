#!/usr/bin/env python3
"""
Video Effects UI Application

A Flask-based web interface for selecting video clips and applying effects.
Provides real-time preview and export functionality.
"""

import os
import sys
import json
import threading
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import logging

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# MoviePy 1.x imports (fallback)
from moviepy.editor import VideoFileClip, CompositeVideoClip
from moviepy.video.fx import speedx

try:
    from video_effects import (
        MotionEffectsEngine,
        AnimeEffectsLibrary,
        ColorEffectsEngine,
        TextEffectsEngine,
        AudioSyncEngine,
        TransitionEngine
    )
    EFFECTS_AVAILABLE = True and VideoFileClip is not None
except ImportError as e:
    logging.warning(f"Effects modules not available: {e}")
    EFFECTS_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'anime_effects_ui_secret_key_2024'

# Configuration
UPLOAD_FOLDER = Path('ui/uploads')
OUTPUT_FOLDER = Path('ui/outputs')
CLIPS_FOLDER = Path('extracted_clips')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Create directories
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# Global variables for processing status
processing_status = {}
effects_engines = {}

if EFFECTS_AVAILABLE:
    # Initialize effect engines
    effects_engines = {
        'motion': MotionEffectsEngine(),
        'anime': AnimeEffectsLibrary(),
        'color': ColorEffectsEngine(),
        'text': TextEffectsEngine(),
        'audio_sync': AudioSyncEngine(),
        'transitions': TransitionEngine()
    }

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_available_clips():
    """Get list of available video clips."""
    clips = []
    
    # Check extracted clips folder
    if CLIPS_FOLDER.exists():
        for clip_file in CLIPS_FOLDER.glob('*.mp4'):
            clips.append({
                'name': clip_file.name,
                'path': str(clip_file),
                'source': 'extracted'
            })
    
    # Check uploads folder
    for clip_file in UPLOAD_FOLDER.glob('*'):
        if allowed_file(clip_file.name):
            clips.append({
                'name': clip_file.name,
                'path': str(clip_file),
                'source': 'uploaded'
            })
    
    return clips

def get_effects_config():
    """Get available effects configuration."""
    if not EFFECTS_AVAILABLE:
        return {}
    
    return {
        'motion_effects': {
            'speed_ramp': {
                'name': 'Speed Ramping',
                'description': 'Dynamic speed changes for dramatic effect',
                'parameters': {
                    'speed_points': {
                        'type': 'list',
                        'default': '[(0, 1.0), (2, 0.3), (4, 2.0), (6, 1.0)]',
                        'description': 'List of (time, speed) points'
                    }
                }
            },
            'zoom_punch': {
                'name': 'Zoom Punch',
                'description': 'Rapid zoom effect synchronized with impacts',
                'parameters': {
                    'zoom_time': {'type': 'float', 'default': 2.0, 'min': 0, 'max': 10},
                    'zoom_factor': {'type': 'float', 'default': 1.5, 'min': 1.0, 'max': 3.0},
                    'duration': {'type': 'float', 'default': 0.2, 'min': 0.1, 'max': 1.0}
                }
            },
            'camera_shake': {
                'name': 'Camera Shake',
                'description': 'Simulate camera movement for impact',
                'parameters': {
                    'shake_intensity': {'type': 'int', 'default': 10, 'min': 1, 'max': 30},
                    'shake_duration': {'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 5.0}
                }
            }
        },
        'anime_effects': {
            'speed_lines': {
                'name': 'Speed Lines',
                'description': 'Anime-style motion lines',
                'parameters': {
                    'direction': {
                        'type': 'select',
                        'options': ['right', 'left', 'radial', 'diagonal'],
                        'default': 'right'
                    },
                    'start_time': {'type': 'float', 'default': 0.0, 'min': 0, 'max': 10},
                    'duration': {'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 5.0},
                    'intensity': {'type': 'float', 'default': 0.8, 'min': 0.1, 'max': 1.0}
                }
            },
            'impact_frames': {
                'name': 'Impact Frames',
                'description': 'High-contrast impact effects',
                'parameters': {
                    'impact_times': {
                        'type': 'list',
                        'default': '[2.0, 4.0, 6.0]',
                        'description': 'Times to add impact frames'
                    },
                    'style': {
                        'type': 'select',
                        'options': ['manga', 'energy', 'flash'],
                        'default': 'manga'
                    },
                    'duration': {'type': 'float', 'default': 0.1, 'min': 0.05, 'max': 0.5}
                }
            },
            'energy_aura': {
                'name': 'Energy Aura',
                'description': 'Pulsing energy effect around characters',
                'parameters': {
                    'start_time': {'type': 'float', 'default': 1.0, 'min': 0, 'max': 10},
                    'duration': {'type': 'float', 'default': 3.0, 'min': 0.5, 'max': 10},
                    'intensity': {'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 2.0},
                    'pulse_rate': {'type': 'float', 'default': 6.0, 'min': 1.0, 'max': 20.0}
                }
            }
        },
        'color_effects': {
            'color_grade': {
                'name': 'Color Grading',
                'description': 'Apply anime-style color grading',
                'parameters': {
                    'style': {
                        'type': 'select',
                        'options': ['vibrant', 'sunset', 'moonlight', 'dramatic'],
                        'default': 'vibrant'
                    }
                }
            },
            'chromatic_aberration': {
                'name': 'Chromatic Aberration',
                'description': 'Retro RGB separation effect',
                'parameters': {
                    'intensity': {'type': 'int', 'default': 5, 'min': 1, 'max': 20}
                }
            },
            'bloom': {
                'name': 'Bloom Effect',
                'description': 'Glow effect on bright areas',
                'parameters': {
                    'threshold': {'type': 'int', 'default': 200, 'min': 100, 'max': 255},
                    'blur_size': {'type': 'int', 'default': 15, 'min': 5, 'max': 50}
                }
            }
        },
        'text_effects': {
            'animated_text': {
                'name': 'Animated Text',
                'description': 'Add animated text overlays',
                'parameters': {
                    'text': {'type': 'string', 'default': 'ANIME TEXT!'},
                    'duration': {'type': 'float', 'default': 3.0, 'min': 0.5, 'max': 10},
                    'animation': {
                        'type': 'select',
                        'options': ['slide_in', 'typewriter', 'bounce', 'fade_in', 'zoom_in', 'glitch'],
                        'default': 'slide_in'
                    },
                    'fontsize': {'type': 'int', 'default': 50, 'min': 20, 'max': 100}
                }
            },
            'sound_effect_text': {
                'name': 'Sound Effect Text',
                'description': 'Add impact text like "BOOM!", "SLASH!"',
                'parameters': {
                    'text': {'type': 'string', 'default': 'BOOM!'},
                    'position_x': {'type': 'int', 'default': 200, 'min': 0, 'max': 1920},
                    'position_y': {'type': 'int', 'default': 300, 'min': 0, 'max': 1080},
                    'style': {
                        'type': 'select',
                        'options': ['impact', 'explosive', 'electric'],
                        'default': 'impact'
                    }
                }
            }
        }
    }

@app.route('/')
def index():
    """Main page with effects UI."""
    clips = get_available_clips()
    effects_config = get_effects_config()
    
    return render_template('effects_ui.html', 
                         clips=clips, 
                         effects_config=effects_config,
                         effects_available=EFFECTS_AVAILABLE)

@app.route('/api/clips')
def api_clips():
    """API endpoint to get available clips."""
    return jsonify(get_available_clips())

@app.route('/api/effects')
def api_effects():
    """API endpoint to get effects configuration."""
    return jsonify(get_effects_config())

@app.route('/api/apply_effects', methods=['POST'])
def api_apply_effects():
    """API endpoint to apply effects to a video clip."""
    if not EFFECTS_AVAILABLE:
        return jsonify({'error': 'Effects not available'}), 500
    
    try:
        data = request.json
        clip_path = data.get('clip_path')
        effects = data.get('effects', {})
        output_name = data.get('output_name', 'processed_clip.mp4')
        
        if not clip_path or not Path(clip_path).exists():
            return jsonify({'error': 'Invalid clip path'}), 400
        
        # Generate unique job ID
        job_id = f"job_{int(time.time())}"
        processing_status[job_id] = {
            'status': 'starting',
            'progress': 0,
            'message': 'Initializing...'
        }
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_video_effects,
            args=(job_id, clip_path, effects, output_name)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'job_id': job_id, 'status': 'started'})
        
    except Exception as e:
        logging.error(f"Error in apply_effects: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>')
def api_status(job_id):
    """Get processing status for a job."""
    status = processing_status.get(job_id, {'status': 'not_found'})
    return jsonify(status)

@app.route('/api/download/<filename>')
def api_download(filename):
    """Download processed video file."""
    try:
        return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload a video file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = UPLOAD_FOLDER / filename
        file.save(file_path)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'path': str(file_path)
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

def process_video_effects(job_id, clip_path, effects, output_name):
    """Process video effects in background thread."""
    try:
        processing_status[job_id] = {
            'status': 'loading',
            'progress': 10,
            'message': 'Loading video clip...'
        }
        
        # Load video clip
        clip = VideoFileClip(clip_path)
        
        processing_status[job_id] = {
            'status': 'processing',
            'progress': 20,
            'message': 'Applying effects...'
        }
        
        # Apply effects based on configuration
        total_effects = len(effects)
        current_effect = 0
        
        for effect_category, effect_list in effects.items():
            if effect_category == 'motion_effects':
                for effect_name, params in effect_list.items():
                    clip = apply_motion_effect(clip, effect_name, params)
                    current_effect += 1
                    progress = 20 + (current_effect / total_effects) * 60
                    processing_status[job_id]['progress'] = progress
                    processing_status[job_id]['message'] = f'Applied {effect_name}'
            
            elif effect_category == 'anime_effects':
                for effect_name, params in effect_list.items():
                    clip = apply_anime_effect(clip, effect_name, params)
                    current_effect += 1
                    progress = 20 + (current_effect / total_effects) * 60
                    processing_status[job_id]['progress'] = progress
                    processing_status[job_id]['message'] = f'Applied {effect_name}'
            
            elif effect_category == 'color_effects':
                for effect_name, params in effect_list.items():
                    clip = apply_color_effect(clip, effect_name, params)
                    current_effect += 1
                    progress = 20 + (current_effect / total_effects) * 60
                    processing_status[job_id]['progress'] = progress
                    processing_status[job_id]['message'] = f'Applied {effect_name}'
            
            elif effect_category == 'text_effects':
                text_clips = []
                for effect_name, params in effect_list.items():
                    text_clip = apply_text_effect(effect_name, params)
                    if text_clip:
                        text_clips.append(text_clip)
                
                if text_clips:
                    clip = CompositeVideoClip([clip] + text_clips)
                
                current_effect += len(effect_list)
                progress = 20 + (current_effect / total_effects) * 60
                processing_status[job_id]['progress'] = progress
                processing_status[job_id]['message'] = 'Applied text effects'
        
        processing_status[job_id] = {
            'status': 'exporting',
            'progress': 80,
            'message': 'Exporting video...'
        }
        
        # Export video
        output_path = OUTPUT_FOLDER / output_name
        clip.write_videofile(str(output_path), codec='libx264', audio_codec='aac')
        
        processing_status[job_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Processing completed!',
            'output_file': output_name
        }
        
        # Clean up
        clip.close()
        
    except Exception as e:
        logging.error(f"Error processing video effects: {e}")
        processing_status[job_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }

def apply_motion_effect(clip, effect_name, params):
    """Apply motion effect to clip."""
    motion_fx = effects_engines['motion']
    
    if effect_name == 'speed_ramp':
        speed_points = eval(params.get('speed_points', '[(0, 1.0), (2, 0.3), (4, 2.0)]'))
        return motion_fx.speed_ramp_effect(clip, speed_points)
    
    elif effect_name == 'zoom_punch':
        return motion_fx.zoom_punch_effect(
            clip,
            zoom_time=float(params.get('zoom_time', 2.0)),
            zoom_factor=float(params.get('zoom_factor', 1.5)),
            duration=float(params.get('duration', 0.2))
        )
    
    elif effect_name == 'camera_shake':
        return motion_fx.camera_shake_effect(
            clip,
            shake_intensity=int(params.get('shake_intensity', 10)),
            shake_duration=float(params.get('shake_duration', 1.0))
        )
    
    return clip

def apply_anime_effect(clip, effect_name, params):
    """Apply anime effect to clip."""
    anime_fx = effects_engines['anime']
    
    if effect_name == 'speed_lines':
        return anime_fx.speed_lines_clip(
            clip,
            direction=params.get('direction', 'right'),
            start_time=float(params.get('start_time', 0.0)),
            duration=float(params.get('duration', 1.0)),
            intensity=float(params.get('intensity', 0.8))
        )
    
    elif effect_name == 'impact_frames':
        impact_times = eval(params.get('impact_times', '[2.0, 4.0]'))
        return anime_fx.add_impact_frames(
            clip,
            impact_times=impact_times,
            style=params.get('style', 'manga'),
            duration=float(params.get('duration', 0.1))
        )
    
    elif effect_name == 'energy_aura':
        return anime_fx.energy_aura_effect(
            clip,
            start_time=float(params.get('start_time', 1.0)),
            duration=float(params.get('duration', 3.0)),
            intensity=float(params.get('intensity', 1.0)),
            pulse_rate=float(params.get('pulse_rate', 6.0))
        )
    
    return clip

def apply_color_effect(clip, effect_name, params):
    """Apply color effect to clip."""
    color_fx = effects_engines['color']
    
    if effect_name == 'color_grade':
        return color_fx.apply_color_grading(clip, params.get('style', 'vibrant'))
    
    elif effect_name == 'chromatic_aberration':
        intensity = int(params.get('intensity', 5))
        return clip.fl_image(lambda img: color_fx.chromatic_aberration_effect(img, intensity))
    
    elif effect_name == 'bloom':
        threshold = int(params.get('threshold', 200))
        blur_size = int(params.get('blur_size', 15))
        return clip.fl_image(lambda img: color_fx.bloom_effect(img, threshold, blur_size))
    
    return clip

def apply_text_effect(effect_name, params):
    """Apply text effect and return text clip."""
    text_fx = effects_engines['text']
    
    if effect_name == 'animated_text':
        return text_fx.create_animated_text(
            text=params.get('text', 'ANIME TEXT!'),
            duration=float(params.get('duration', 3.0)),
            animation=params.get('animation', 'slide_in'),
            fontsize=int(params.get('fontsize', 50))
        )
    
    elif effect_name == 'sound_effect_text':
        position = (
            int(params.get('position_x', 200)),
            int(params.get('position_y', 300))
        )
        return text_fx.sound_effect_text(
            text=params.get('text', 'BOOM!'),
            position=position,
            style=params.get('style', 'impact')
        )
    
    return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Starting Video Effects UI...")
    print("Access the interface at: http://localhost:5000")
    
    if not EFFECTS_AVAILABLE:
        print("WARNING: Effects modules not available. UI will have limited functionality.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)