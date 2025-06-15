# Anime Video Effects UI

A comprehensive web-based interface for applying professional video effects to anime clips.

## Features

### 🎬 Video Clip Management
- **Clip Selection**: Browse and select from extracted clips or uploaded videos
- **Drag & Drop Upload**: Easy file upload with drag and drop support
- **Multiple Sources**: Support for both extracted clips and custom uploads

### ✨ Effects Categories

#### Motion Effects
- **Speed Ramping**: Dynamic speed changes for dramatic timing
- **Zoom Punch**: Rapid zoom effects synchronized with impacts
- **Camera Shake**: Simulate camera movement for action sequences
- **Motion Blur**: Fast movement blur effects
- **Freeze Frame**: Dramatic pause effects

#### Anime-Specific Effects
- **Speed Lines**: Motion lines in multiple directions (radial, horizontal, diagonal)
- **Impact Frames**: High-contrast manga-style impact effects
- **Energy Aura**: Pulsing energy effects around characters
- **Action Lines**: Radiating lines for power-up scenes
- **Screen Tones**: Manga-style dot and line patterns

#### Color & Lighting Effects
- **Color Grading**: Multiple anime styles (vibrant, sunset, moonlight, dramatic)
- **Chromatic Aberration**: Retro RGB separation effects
- **Bloom Effects**: Glow effects on bright areas
- **Vignette**: Dramatic edge darkening
- **VHS Vintage**: Retro tape effects

#### Text & Typography
- **Animated Text**: 6 animation types (slide, typewriter, bounce, fade, zoom, glitch)
- **Sound Effect Text**: Impact text like "BOOM!", "SLASH!"
- **Character Introductions**: Professional character name displays
- **Technique Names**: Attack/move name overlays

#### Audio Synchronization
- **Beat Detection**: Automatic audio beat detection
- **Visual Sync**: Effects timed to music beats
- **Energy Mapping**: Effect intensity based on audio energy
- **Multi-Beat Effects**: Complex synchronized effect combinations

#### Transitions
- **Iris Transitions**: Classic anime-style circular reveals
- **Swipe Transitions**: Directional wipe effects
- **Spiral Transitions**: Rotating reveal patterns
- **Slice Transitions**: Multiple segment reveals
- **Pixelate Transitions**: Digital transformation effects

## Quick Start

### 1. Launch the UI
```bash
python launch_ui.py
```

### 2. Access the Interface
Open your browser to `http://localhost:5000`

### 3. Select a Video Clip
- Choose from extracted clips in the left panel
- Or upload a new video file using drag & drop

### 4. Configure Effects
- Expand effect categories to see available options
- Check the effects you want to apply
- Adjust parameters for each effect

### 5. Process Video
- Enter an output filename
- Click "Apply Effects" to start processing
- Monitor progress in real-time
- Download the processed video when complete

## Interface Walkthrough

### Main Panels

#### 📁 Clip Selection Panel
- **Clip Grid**: Visual selection of available video clips
- **Upload Area**: Drag and drop zone for new videos
- **Source Indicators**: Shows whether clips are extracted or uploaded

#### ✨ Effects Selection Panel
- **Collapsible Categories**: Organized effect groupings
- **Effect Cards**: Individual effect selection with descriptions
- **Parameter Controls**: Customizable settings for each effect
- **Real-time Preview**: Parameter validation and suggestions

#### 🎬 Process Panel
- **Output Naming**: Custom filename for processed videos
- **Process Button**: Start effect application
- **Progress Tracking**: Real-time processing status
- **Download Links**: Access to completed videos

### Controls

#### Effect Parameters
Each effect has customizable parameters:
- **Sliders**: For numerical ranges (intensity, duration, etc.)
- **Dropdowns**: For predefined options (styles, directions)
- **Text Fields**: For custom values (text content, timestamps)
- **Lists**: For multiple values (impact times, speed points)

#### Processing Controls
- **Apply Effects**: Start video processing
- **Reset**: Clear all selections and start over
- **Preview Settings**: View current configuration
- **Progress Bar**: Visual processing progress

## File Organization

```
ui/
├── effects_app.py          # Main Flask application
├── templates/
│   └── effects_ui.html     # Web interface template
├── static/
│   └── styles.css          # Additional styling
├── uploads/                # User uploaded videos
└── outputs/                # Processed video files
```

## API Endpoints

### GET `/`
Main UI interface

### GET `/api/clips`
Get list of available video clips

### GET `/api/effects`
Get effects configuration and parameters

### POST `/api/apply_effects`
Apply effects to selected clip
```json
{
    "clip_path": "/path/to/clip.mp4",
    "effects": {
        "motion_effects": {
            "zoom_punch": {
                "zoom_time": 2.0,
                "zoom_factor": 1.5
            }
        }
    },
    "output_name": "processed_clip.mp4"
}
```

### GET `/api/status/<job_id>`
Get processing status for a job

### GET `/api/download/<filename>`
Download processed video file

### POST `/upload`
Upload new video file

## Effect Presets

The UI includes several preset configurations:

### Action Scene
- Speed ramping with slow-motion impacts
- Zoom punches at key moments
- Camera shake for intensity
- Vibrant color grading

### Emotional Moment
- Soft dissolve transitions
- Sunset color grading
- Energy aura effects
- Gentle text animations

### Power-Up Scene
- Radial speed lines
- Energy aura with high intensity
- Action lines radiating from center
- Dramatic color enhancement

### Retro Style
- VHS vintage effects
- Chromatic aberration
- Warm color temperature
- Glitch text animations

## Performance Tips

### For Best Results:
1. **Clip Length**: Keep clips under 30 seconds for faster processing
2. **Effect Combinations**: Start with 2-3 effects, add more gradually
3. **File Formats**: MP4 works best for compatibility
4. **Resolution**: 1080p recommended for quality vs. speed balance

### Processing Time Estimates:
- **Simple Effects** (1-2 effects): 1-2 minutes for 10-second clip
- **Complex Effects** (3-5 effects): 3-5 minutes for 10-second clip
- **Text Effects**: Add minimal processing time
- **Audio Sync**: May add 30-60 seconds for beat detection

## Troubleshooting

### Common Issues:

#### "Effects Not Available"
- Install missing dependencies: `pip install -r requirements.txt`
- Ensure all video effects modules are properly installed

#### Slow Processing
- Reduce clip length
- Use fewer simultaneous effects
- Check system resources (CPU/Memory)

#### Upload Failures
- Check file format (MP4, AVI, MOV, MKV supported)
- Ensure file size is reasonable (<500MB recommended)
- Verify disk space for uploads

#### Browser Issues
- Try refreshing the page
- Clear browser cache
- Use Chrome or Firefox for best compatibility

### Getting Help
- Check the console output for detailed error messages
- Review the application logs for processing issues
- Ensure all file paths are accessible

## Advanced Usage

### Custom Effect Combinations
Create sophisticated effects by combining multiple categories:

```json
{
    "motion_effects": {
        "speed_ramp": {"speed_points": "[(0, 1.0), (2, 0.3), (4, 2.0)]"},
        "zoom_punch": {"zoom_time": 2.5, "zoom_factor": 1.8}
    },
    "anime_effects": {
        "speed_lines": {"direction": "radial", "intensity": 0.9},
        "impact_frames": {"impact_times": "[2.0, 4.0]", "style": "energy"}
    },
    "color_effects": {
        "color_grade": {"style": "dramatic"},
        "bloom": {"threshold": 180, "blur_size": 20}
    }
}
```

### Batch Processing
While the UI processes one video at a time, you can:
1. Process multiple clips sequentially
2. Use different effect combinations for variety
3. Organize outputs with descriptive filenames

### Quality Control
- Preview effects with short test clips first
- Adjust parameters based on content type
- Use consistent naming for organized output management

This UI provides a powerful, user-friendly way to create professional anime video effects without command-line complexity!