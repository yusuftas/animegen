# 🎬 Anime Effects Studio - Standalone UI

A comprehensive video effects editor specifically designed for anime content processing. This standalone application provides a modern, intuitive interface for applying sophisticated visual effects to video clips with real-time preview capabilities.

## ✨ Features

### 🎭 Effects Library
- **Motion Effects**: Speed ramping, zoom punch, camera shake, motion blur
- **Anime Effects**: Speed lines, impact frames, energy aura, character glow
- **Color Effects**: Color grading, chromatic aberration, bloom, VHS effects
- **Text Effects**: Animated text overlays, sound effect text, subtitles
- **Audio Sync**: Beat-synchronized effects, flash, zoom, color pulse
- **Transitions**: Iris, swipe, dissolve, spiral, and custom transitions

### 🎬 Pipeline Editor
- **Drag-and-drop** effect ordering
- **Real-time parameter** adjustment
- **Effect enabling/disabling** with visual feedback
- **Save/Load** effect pipelines
- **Preset combinations** for common scenarios

### 📺 Live Preview
- **Real-time video preview** with applied effects
- **Timeline scrubbing** and playback controls
- **Performance-optimized** preview generation
- **Export settings** with quality controls
- **Batch processing** capabilities

### ⚡ Quick Presets
- **Action Scene**: Speed lines + zoom punch + camera shake + dramatic grading
- **Power Up**: Energy aura + bloom + animated text + beat effects
- **Emotional Moment**: Sunset grading + bloom + fade text
- **Speed Boost**: Speed lines + speed ramp + chromatic aberration
- **Impact Hit**: Zoom punch + camera shake + impact frames

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (for video processing)
- Git

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd animegen/ui_standalone
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\\Scripts\\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install FFmpeg

#### Windows:
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to `C:\\ffmpeg`
3. Add `C:\\ffmpeg\\bin` to your PATH

#### macOS:
```bash
brew install ffmpeg
```

#### Linux:
```bash
sudo apt update
sudo apt install ffmpeg
```

### Step 5: Run Application
```bash
python main_ui.py
```

## 📖 Usage Guide

### Basic Workflow

1. **Load Video**
   - Click "Choose File" to load your video
   - Supported formats: MP4, AVI, MOV, MKV, WMV, FLV

2. **Add Effects**
   - Browse the Effects Library on the left
   - Click effects to add them to your pipeline
   - Use search to find specific effects quickly

3. **Configure Effects**
   - Click the ⚙️ button on any effect to configure parameters
   - Drag effects up/down to reorder
   - Toggle effects on/off with the ✓/✗ button

4. **Preview Results**
   - Watch real-time preview in the right panel
   - Use playback controls to scrub through video
   - Apply quick presets for common effect combinations

5. **Export Video**
   - Configure quality settings (720p - 4K)
   - Choose frame rate (24/30/60 fps)
   - Select format and codec
   - Click "RENDER FINAL VIDEO"

### Advanced Features

#### Effect Categories

**Motion Effects**
- *Speed Ramp*: Dynamic speed changes for dramatic emphasis
- *Zoom Punch*: Rapid zoom-in synchronized with impact moments
- *Camera Shake*: Simulate camera movement for excitement

**Anime Effects**
- *Speed Lines*: Classic anime motion lines in multiple directions
- *Impact Frames*: High-contrast frames for dramatic moments
- *Energy Aura*: Pulsing energy effects around characters

**Color Effects**
- *Color Grade*: Anime-style color enhancement (vibrant/sunset/dramatic)
- *Chromatic Aberration*: RGB separation for retro effects
- *Bloom*: Glow effects on bright areas

**Text Effects**
- *Animated Text*: Dynamic text with multiple animation styles
- *Sound FX Text*: Impact text like "BOOM!", "SLASH!"
- *Subtitles*: Styled subtitle overlays

#### Keyboard Shortcuts
- `Ctrl+O`: Load video file
- `Ctrl+S`: Save effects pipeline
- `Ctrl+E`: Export video
- `Space`: Play/pause preview
- `Delete`: Delete selected effect

#### Pipeline Management
- **Save Pipeline**: Save your effect combination for reuse
- **Load Pipeline**: Load previously saved effect combinations
- **Preset System**: Quick access to common effect combinations
- **Effect Reordering**: Drag effects to change processing order

## ⚙️ Configuration

### Performance Settings

For better performance on lower-end systems:

1. **Reduce Preview Quality**
   - Set preview quality to HD instead of Full HD
   - Lower preview frame rate in settings

2. **Effect Optimization**
   - Disable real-time preview for complex effects
   - Use batch processing for multiple files

3. **Memory Management**
   - Close unused applications
   - Increase virtual memory if needed

### Quality Settings

#### Export Presets:
- **Web/Social Media**: 720p, 30fps, H.264
- **High Quality**: 1080p, 60fps, H.264
- **Professional**: 4K, 60fps, H.265
- **Archive**: 1440p, 30fps, H.264

## 🎯 Effect Parameters

### Motion Effects
- **Speed Ramp**: Time/speed point pairs for dynamic pacing
- **Zoom Punch**: Zoom factor (1.0-3.0), duration (0.1-1.0s)
- **Camera Shake**: Intensity (1-30 pixels), duration (0.1-5.0s)

### Anime Effects
- **Speed Lines**: Direction (right/left/radial), intensity (0.1-1.0)
- **Impact Frame**: Style (manga/energy/flash), duration (0.1-1.0s)
- **Energy Aura**: Intensity (0.1-2.0), pulse rate (1-15 Hz)

### Color Effects
- **Color Grade**: Style selection, intensity (0.0-2.0)
- **Chromatic Aberration**: Intensity (1-20 pixels)
- **Bloom**: Threshold (100-255), blur size (5-50 pixels)

### Text Effects
- **Animated Text**: Text content, animation type, font size (12-120)
- **Position**: Center, top, bottom, left, right, custom coordinates
- **Timing**: Start time, duration, fade in/out

## 🐛 Troubleshooting

### Common Issues

**"MoviePy not found" Error**
```bash
pip install moviepy
```

**"FFmpeg not found" Error**
- Ensure FFmpeg is installed and in PATH
- Restart terminal/command prompt after installation

**"Permission denied" Error**
- Run as administrator (Windows) or with sudo (Linux/macOS)
- Check file permissions for video files

**Poor Performance**
- Reduce preview quality in settings
- Close other applications
- Use SSD storage for temporary files

**Video Not Loading**
- Check if video format is supported
- Try converting to MP4 format
- Ensure video file is not corrupted

### Getting Help

1. Check the troubleshooting section above
2. Review error messages in the console
3. Ensure all dependencies are properly installed
4. Try with a different video file to isolate issues

## 🔧 Development

### Project Structure
```
ui_standalone/
├── main_ui.py              # Main application entry point
├── components/             # UI components
│   ├── effects_library.py  # Effects browser panel
│   ├── pipeline_editor.py  # Pipeline management panel
│   └── video_preview.py    # Preview and export panel
├── models/                 # Data models
│   ├── effect_models.py    # Individual effect definitions
│   └── effect_pipeline.py  # Pipeline management
├── utils/                  # Utility modules
│   ├── video_processor.py  # Video processing engine
│   └── preview_generator.py # Real-time preview generation
└── requirements.txt        # Dependencies
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Building from Source

```bash
# Clone repository
git clone <repository-url>
cd animegen/ui_standalone

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-qt

# Run tests
pytest

# Run application
python main_ui.py
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with CustomTkinter for modern UI components
- Video processing powered by MoviePy and OpenCV
- Audio analysis using LibROSA
- Effect algorithms inspired by professional anime production techniques

---

**Anime Effects Studio** - Transform your videos with professional anime-style effects! 🎬✨