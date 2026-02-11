<div align="center">

# Media to Text

**AI-Powered Media Transcription Using OpenAI's Whisper**

[![Version](https://img.shields.io/badge/Version-1.3.0-red?logo=github&logoColor=white)](https://github.com/Zer0Wav3s/media-to-text/releases)
[![OpenAI](https://img.shields.io/badge/OpenAI-Whisper-green?logo=openai&logoColor=white)](https://openai.com/research/whisper)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-required-orange?logo=ffmpeg&logoColor=white)](https://ffmpeg.org)

</div>

> **Disclaimer**: This is an experimental project provided as-is. The author assumes no responsibility for how this tool is used. By using this software, you agree that you are solely responsible for ensuring your use complies with applicable laws and terms of service. Use at your own risk.

---

A command-line tool that converts audio and video files to text using OpenAI's Whisper API. Automatically splits large files into chunks and transcribes them concurrently for maximum speed.

## Features

- **Async concurrent processing** with 5 parallel chunk uploads per file
- **PyAV audio extraction** with automatic FFmpeg fallback for unsupported formats
- **Optimized audio** (16kHz/16kbps/mono) for 20-50% faster API transcription
- **Smart chunking** that auto-splits files exceeding OpenAI's 25MB limit
- **Brew-style progress display** with real-time chunk status and ETA calculations
- **Smart terminal fallback** for CI/CD and non-ANSI environments
- **Automatic retry logic** with 3 attempts per chunk on failure
- **Transcript combiner** to merge multiple transcripts into a single file

## Supported Formats

| Video | Audio |
|-------|-------|
| `.mp4` `.mkv` `.webm` | `.mp3` `.wav` `.flac` |
| `.avi` `.mov` `.wmv` | `.aac` `.m4a` `.ogg` |
| `.flv` `.m4v` `.3gp` | `.opus` `.wma` `.aiff` `.amr` |

Unsupported formats (like `.caf`) are automatically converted to MP3 using FFmpeg.

## Installation

```bash
# Clone the repository
git clone https://github.com/Zer0Wav3s/media-to-text.git
cd media-to-text

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (required)
brew install ffmpeg        # macOS
sudo apt install ffmpeg    # Ubuntu/Debian
```

## Usage

```bash
python transcribe.py
```

1. **First run** creates a `.env` file — add your OpenAI API key: `OPENAI_API_KEY=your_key_here`
2. **Place media files** in the `media-files/` directory
3. **Run the script** — select files from the interactive menu
4. **Transcripts** are saved to the `transcripts/` directory

### Combining Transcripts

```bash
python combine_transcripts.py
```

Merges all transcripts in `transcripts/` into a single cleaned text file with timestamps and headers removed.

## How It Works

1. **Audio Extraction** — PyAV extracts audio at 16kHz/16kbps/mono (FFmpeg fallback for unsupported formats)
2. **Chunking** — Files over 25MB are split into gapless chunks using FFmpeg
3. **Parallel Transcription** — Up to 5 chunks sent to the Whisper API concurrently
4. **Assembly** — Chunk transcripts are combined in order and saved
5. **Cleanup** — Temporary files are removed automatically

Each file gets a fresh API client to prevent connection pool hangs. Ctrl+C restores cursor visibility and cleans up temp files.

## Configuration

Edit constants in `transcribe.py` to customize:

```python
MAX_FILE_SIZE = 25 * 1024 * 1024   # 25MB chunk limit
max_concurrent = 5                  # Parallel API calls per file
```

## Requirements

- Python 3.8+
- FFmpeg installed and available in PATH
- OpenAI API key
- Core dependencies:
  - `openai>=1.60.2`
  - `python-dotenv>=1.0.1`
  - `av>=13.0.0` (PyAV)
  - `aiofiles>=24.1.0`

## License

MIT
