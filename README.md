<div align="center">

# media-to-text

**AI-Powered Media Transcription Using OpenAI's Whisper**

[![Version](https://img.shields.io/badge/Version-1.1.0-red?logo=github&logoColor=white)](https://github.com/Zer0Wav3s/media-to-text/releases)
[![OpenAI](https://img.shields.io/badge/OpenAI-Whisper-green?logo=openai&logoColor=white)](https://openai.com/research/whisper)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-required-red?logo=ffmpeg&logoColor=white)](https://ffmpeg.org)

</div>

---

### Overview

A high-performance command-line tool that uses AI to automatically:
- Convert any audio/video to text with high accuracy
- **Auto-convert unsupported formats** (like .caf) to MP3 automatically
- **Optimized audio processing** (16kHz/16kbps/mono) for 20-50% faster API transcription
- Split large files into chunks under 25MB (OpenAI's limit)
- **Transcribe chunks in parallel** (5 concurrent) for maximum speed
- Process files sequentially with fresh API clients (prevents connection issues)
- **Brew-style progress display** with real-time chunk status and ETA calculations
- Handle errors gracefully with automatic retry logic and cleanup
- Combine multiple transcripts into a single file

**Key Features:**
- Async/concurrent chunk processing (5 simultaneous API calls per file)
- PyAV for fast audio extraction with FFmpeg fallback
- Optimized audio settings reduce API processing time by 20-50%
- Fresh API client per file prevents connection pool hangs
- Async file I/O operations throughout
- **Brew-style multi-line progress display** with ANSI terminal support
- **Smart terminal fallback** for CI/CD and non-ANSI environments

This is an open-source setup designed to be easily:
- Forked and modified for your specific needs
- Integrated into larger projects
- Extended with additional features
- Shared and improved by the community

Feel free to use, modify, and share this tool as you see fit!

### Quick Start

1. **Set Up Environment**
```bash
# Clone the repository
git clone https://github.com/Zer0Wav3s/media-to-text.git
cd media-to-text

# Install dependencies
pip install -r requirements.txt
```

2. **Add Your API Key**
```bash
# The script will create .env for you
# Just add your OpenAI API key:
OPENAI_API_KEY=your_key_here
```

3. **Run It**
```bash
# Transcribe media files
python transcribe.py

# Combine transcripts (optional)
python combine_transcripts.py
```

### Supported Formats

<div align="left">

| Video Formats | Audio Formats |
|:-------------:|:-------------:|
| `.mp4` `.mkv` | `.mp3` `.wav` |
| `.webm` `.avi` | `.flac` `.aac` |
| `.mov` `.wmv` | `.m4a` `.ogg` |
| `.flv` `.m4v` | `.opus` `.wma` |
| `.3gp` | `.aiff` `.amr` |

**Auto-Conversion:** Unsupported formats (like `.caf`) are automatically converted to MP3 using FFmpeg before transcription.

</div>

### Requirements

- Python 3.8+
- FFmpeg (for audio extraction and format conversion)
- OpenAI API key
- Required packages:
  ```
  openai>=1.60.2
  python-dotenv>=1.0.1
  av>=13.0.0 (PyAV - fast FFmpeg bindings)
  aiofiles>=24.1.0
  ```

### Pro Tips

**For Best Results:**
- Use clear audio with minimal background noise
- Ensure sufficient disk space for temporary files
- Monitor your OpenAI API usage/costs
- Expect 7-10 minutes per hour of audio (varies by server load)

**Performance Optimization:**
- **Chunks processed in parallel** (5 concurrent API calls per file)
- **Files processed sequentially** (one at a time with fresh API client)
- Audio optimized to 16kHz/16kbps/mono (20-50% faster API processing)
- Adjust concurrency in `AsyncTranscriber(max_concurrent=N)` if needed
- Higher concurrency = faster but may hit API rate limits
- Default settings optimized for OpenAI tier limits

**File Processing:**
- **Auto-conversion:** Unsupported formats automatically converted to MP3
- **Smart chunking:** Files >25MB split automatically (most optimized files stay under limit)
- **Parallel chunks:** Multiple chunks from same file processed simultaneously
- **Fresh client:** Each file gets new API client to prevent connection hangs
- **Progress tracking:** Brew-style multi-line display with chunk status and ETA

**Transcript Combination:**
- All transcripts are saved in the `transcripts/` directory
- Use `combine_transcripts.py` to merge multiple transcripts
- Files are read and processed concurrently for speed
- Combined file optimized for readability

### Project Structure
```
media-to-text/
├── transcribe.py         # Main transcription script (async)
├── combine_transcripts.py # Transcript combiner (async)
├── requirements.txt      # Dependencies
├── .env                 # API key
├── media-files/         # Input files
├── transcripts/         # Individual transcripts
└── temp/               # Processing files
```

### Changelog

#### Version 1.1.0 (2026-01-27)
**Brew-Style Progress Display & Async Architecture**
- **Brew-style multi-line progress display** with real-time chunk status updates
- **ANSI terminal support** with cursor control for in-place updates
- **Smart fallback mode** for CI/CD, NO_COLOR, and non-TTY environments
- **ETA calculations** with elapsed time and remaining time estimates
- **Async/concurrent architecture** with 5 parallel chunk processing per file
- **PyAV integration** replacing MoviePy for faster audio extraction
- **Optimized audio processing** (16kHz/16kbps/mono) for 20-50% faster API transcription
- **Auto-conversion** of unsupported formats (e.g., .caf) to MP3 using FFmpeg
- **Fresh API client per file** prevents connection pool hangs
- **Signal handling** (SIGINT/SIGTERM) restores cursor visibility on interrupt
- **Terminal capability detection** (TTY, NO_COLOR, Windows ctypes support)
- Removed YouTube support to simplify codebase
- Fixed parallel chunk processing for optimal speed
- Added CLAUDE.md for AI assistant context

#### Version 1.0.1 (2024-12-17)
- Added .gitignore file
- Minor documentation updates

#### Version 1.0.0 (2024-12-17)
- Initial release
- Basic transcription functionality with OpenAI Whisper
- Support for multiple audio/video formats
- Automatic chunk processing for files >25MB
- Transcript combination utility

### Roadmap

Check out our [Future Updates & Enhancements](FUTURE_UPDATES.md) document for planned features and improvements.

### Contributing

Found a bug or want to contribute? Feel free to:
- Open an issue
- Submit a pull request
- Suggest improvements

### License

MIT License - Use it, modify it, share it.

---

<div align="center">

**Made by [Zer0Wav3s](https://github.com/Zer0Wav3s)**

</div>
