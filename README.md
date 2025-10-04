# 🎬 Auto-Generated Instagram Video System

A fully automated pipeline for creating human-like Instagram Reels with AI-generated content, voice, and visuals.

## 🚀 Features

- **AI Script Generation**: Uses LLMs to create engaging, conversational scripts
- **Natural Voice Synthesis**: Converts text to human-like speech
- **Avatar Animation**: Creates lip-synced talking head videos
- **Auto-Captioning**: Generates accurate subtitles
- **Visual Enhancement**: Adds B-roll, backgrounds, and effects
- **Instagram-Ready Export**: Outputs vertical videos optimized for Reels

## 📋 Architecture

```
TOPIC → Script Generation → Voice Synthesis → Avatar Creation → Video Composition → Export
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Script Generation | LLaMA 3 / Mixtral | Human-like dialogue creation |
| Text-to-Speech | Bark / XTTS v2 | Natural voice generation |
| Avatar | SadTalker / Wav2Lip | Lip-synced face animation |
| Subtitles | Whisper | Automatic captioning |
| Video Processing | FFmpeg | Composition and export |
| Orchestration | Python | Pipeline automation |

## 📁 Project Structure

```
├── src/
│   ├── prompt_generator/     # Topic and prompt generation
│   ├── script_writer/         # Script creation and styling
│   ├── voice_generator/       # TTS implementation
│   ├── avatar_generator/      # Face animation and lip-sync
│   ├── visual_composer/       # Background, B-roll, captions
│   └── video_exporter/        # Final video assembly
├── config/                    # Configuration files
├── data/
│   ├── avatars/              # Avatar images/videos
│   ├── backgrounds/          # Background assets
│   └── templates/            # Script templates
├── output/                   # Generated videos
└── requirements.txt          # Python dependencies
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/prachijecrc-debug/genAIVideoTry-1.git
cd genAIVideoTry-1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download models:
```bash
python scripts/download_models.py
```

## 🎯 Usage

### Quick Start
```python
from src.pipeline import VideoGenerator

generator = VideoGenerator()
video = generator.create_video(
    topic="How photosynthesis works",
    style="educational",
    duration=60  # seconds
)
video.export("output/photosynthesis_reel.mp4")
```

### Command Line
```bash
python main.py --topic "5 Amazing Space Facts" --style "exciting" --duration 30
```

## 📊 Pipeline Stages

### 1. Prompt Generation
- Generates structured topic outlines
- Defines tone and style parameters
- Creates section breakdowns

### 2. Script Writing
- Expands outlines into conversational dialogue
- Adds natural speech patterns and fillers
- Includes gesture and emotion cues

### 3. Voice Generation
- Converts script to natural speech
- Adds emotion and intonation
- Supports multiple voices and languages

### 4. Avatar Animation
- Creates realistic talking head videos
- Synchronizes lip movements with audio
- Supports custom avatars

### 5. Visual Composition
- Adds relevant B-roll footage
- Generates dynamic backgrounds
- Overlays captions and graphics

### 6. Export & Optimization
- Renders final video in Instagram format (1080x1920)
- Optimizes for mobile viewing
- Adds metadata and hashtags

## 🎨 Customization

### Custom Avatars
Place your avatar images in `data/avatars/` and configure in `config/avatars.yaml`

### Voice Profiles
Create custom voice profiles in `config/voices.yaml`

### Script Templates
Add custom templates in `data/templates/` for different content styles

## 📈 Performance

- Average generation time: 2-3 minutes per 60-second video
- GPU recommended for avatar generation
- Supports batch processing for multiple videos

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- LLaMA by Meta AI
- Bark by Suno AI
- SadTalker research team
- FFmpeg community

## 📧 Contact

For questions or support, please open an issue on GitHub.
