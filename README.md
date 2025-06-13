# Anime YouTube Shorts Generator

An AI-powered automation system for generating engaging anime YouTube shorts with dynamic content, intelligent scene detection, and copyright-safe transformations.

## 🎯 Project Status

This project is currently in **Phase 1** of development (Foundation & Scene Analysis). The following features are implemented:

✅ **Completed:**
- Scene detection and extraction
- Interest scoring algorithms  
- Audio peak analysis
- Basic automation pipeline
- Configuration management
- Logging system

🚧 **In Development:**
- Script generation system
- Video editing and effects
- Text-to-speech integration
- YouTube API integration

## 🚀 Features

### Current Capabilities
- **Intelligent Scene Detection**: Automatically identifies interesting moments in anime videos
- **Interest Scoring**: Multi-factor algorithm analyzing motion, faces, composition, and audio
- **Audio Analysis**: Detects peaks, onsets, and significant audio events
- **Configurable Pipeline**: Flexible settings for different anime types and preferences
- **Batch Processing**: Process multiple episodes automatically

### Planned Features
- **Dynamic Script Generation**: Multiple commentary styles with anime knowledge integration
- **Advanced Video Editing**: Random effects, transitions, and anti-template systems
- **Multi-Voice TTS**: Professional voiceover with prosody variations
- **YouTube Integration**: Automated upload with SEO optimization
- **Analytics & Optimization**: ML-driven performance improvement

## 📋 Prerequisites

- Python 3.8 or higher
- FFmpeg (for video processing)
- At least 4GB RAM (8GB+ recommended)
- GPU support recommended for faster processing

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/animegen.git
   cd animegen
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies:**
   
   **Ubuntu/Debian:**
   ```bash
   sudo apt update
   sudo apt install ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0
   ```
   
   **macOS:**
   ```bash
   brew install ffmpeg
   ```
   
   **Windows:**
   - Download FFmpeg from https://ffmpeg.org/download.html
   - Add FFmpeg to your system PATH

4. **Verify installation:**
   ```bash
   python main.py --help
   ```

## 🎮 Usage

### Basic Usage

Process a single anime episode:

```bash
python main.py --video "path/to/anime_episode.mp4" --anime-name "Attack on Titan"
```

### Advanced Options

```bash
python main.py \
  --video "path/to/episode.mp4" \
  --anime-name "Your Anime Name" \
  --output-dir "./shorts_output" \
  --max-shorts 10 \
  --verbose
```

### Batch Processing

Process an entire anime series:

```python
from src.automation.pipeline import AnimeShortsPipeline

pipeline = AnimeShortsPipeline()
results = pipeline.batch_process_episodes(
    video_directory="./anime_episodes/",
    anime_name="My Hero Academia",
    output_dir="./output/"
)
```

## ⚙️ Configuration

The system uses YAML configuration files. Default configuration is created automatically at `config/config.yaml`.

### Key Configuration Sections:

```yaml
scene_analysis:
  threshold: 30.0          # Scene change sensitivity
  min_scene_length: 2.0    # Minimum scene duration (seconds)
  max_scene_length: 15.0   # Maximum scene duration (seconds)

interest_detection:
  motion_weight: 0.4       # Motion analysis weight
  audio_peak_weight: 0.2   # Audio events weight
  composition_weight: 0.2  # Visual composition weight

quality_control:
  min_engagement_score: 0.5  # Minimum interest score threshold
```

## 📁 Project Structure

```
animegen/
├── src/
│   ├── scene_analysis/      # Scene detection and interest scoring
│   ├── content_intelligence/ # Anime knowledge and classification
│   ├── script_generation/   # Dynamic script creation
│   ├── video_assembly/      # Video editing and effects
│   ├── audio_system/        # TTS and audio mixing
│   ├── automation/          # Main pipeline orchestration
│   ├── youtube_integration/ # Upload and optimization
│   ├── analytics/           # Performance tracking
│   └── utils/              # Configuration and logging
├── config/                 # Configuration files
├── data/                   # Anime database and assets
├── tests/                  # Unit and integration tests
└── main.py                # CLI entry point
```

## 🔧 Development

### Running Tests

```bash
python -m pytest tests/ -v
```

### Code Style

The project follows PEP 8 conventions. Use black for formatting:

```bash
pip install black
black src/ tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Performance

Current benchmarks on a typical anime episode (24 minutes):

- **Scene Detection**: ~30-60 seconds
- **Interest Scoring**: ~2-5 minutes  
- **Audio Analysis**: ~1-3 minutes
- **Total Processing**: ~5-10 minutes per episode

## 🎯 Roadmap

### Phase 1: Foundation ✅
- [x] Scene detection and extraction
- [x] Interest scoring algorithms
- [x] Audio analysis
- [x] Basic pipeline structure

### Phase 2: Content Intelligence (Weeks 3-4)
- [ ] Anime knowledge base integration
- [ ] Moment classification system
- [ ] Content relationship mapping

### Phase 3: Script Generation (Weeks 5-6)
- [ ] Template-free script generation
- [ ] Multiple commentary styles
- [ ] Fact integration system

### Phase 4: Video Assembly (Weeks 7-8)
- [ ] Dynamic video editing
- [ ] Visual effects library
- [ ] Anti-template mechanisms

### Phase 5: Full Automation (Weeks 9-12)
- [ ] TTS integration
- [ ] YouTube API
- [ ] Analytics and optimization

## ⚖️ Legal Considerations

This tool is designed to create transformative content that falls under fair use guidelines:

- **Commentary Focus**: Generated content adds substantial commentary and analysis
- **Limited Clips**: Uses short clips (typically 5-10 seconds) from longer works
- **Educational Purpose**: Aims to educate and inform about anime content
- **Transformation**: Adds original narration, effects, and context

**Important**: Users are responsible for ensuring compliance with copyright laws in their jurisdiction. Always verify fair use applicability for your specific content.

## 🐛 Troubleshooting

### Common Issues

1. **"scenedetect not found" Warning**
   - Install with: `pip install scenedetect[opencv]`

2. **Audio extraction fails**
   - Ensure FFmpeg is properly installed and in PATH
   - Check video file format compatibility

3. **Low interest scores**
   - Adjust thresholds in `config/config.yaml`
   - Try different anime types (action vs. slice-of-life)

4. **Memory errors**
   - Reduce video resolution before processing
   - Process shorter segments

### Getting Help

- 📖 Check the [Wiki](https://github.com/yourusername/animegen/wiki)
- 🐛 Report bugs in [Issues](https://github.com/yourusername/animegen/issues)
- 💬 Join discussions in [Discussions](https://github.com/yourusername/animegen/discussions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) for scene detection
- [librosa](https://github.com/librosa/librosa) for audio analysis
- [OpenCV](https://opencv.org/) for computer vision
- The anime community for inspiration and feedback

---

**Disclaimer**: This tool is for educational and research purposes. Users must ensure compliance with copyright laws and platform terms of service.