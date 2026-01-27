# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**media-to-text** is a command-line tool that converts audio/video files to text using OpenAI's Whisper API. It processes local media files, automatically splitting large files into 25MB chunks to comply with API limits. Uses async concurrent processing for optimal performance.

## Core Architecture

### Two-Script System

| Script | Purpose | Lines |
|--------|---------|-------|
| `transcribe.py` | Main async transcription engine with PyAV media processing | ~1500 |
| `combine_transcripts.py` | Post-processing utility that merges transcripts into clean text | ~230 |

### Key Components

| Class | Purpose |
|-------|---------|
| `DependencyManager` | Package checks, SSL certificate configuration |
| `ProgressTracker` | Legacy simple progress updates (deprecated) |
| `EnvironmentSetup` | `.env` file creation and API key validation |
| `MediaProcessor` | PyAV audio extraction, FFmpeg format conversion |
| `AsyncTranscriber` | Concurrent Whisper API calls (5 parallel chunks) |
| `TranscriptionApp` | Main orchestrator coordinating the pipeline |

#### Progress Display System (New)

| Class | Purpose |
|-------|---------|
| `ChunkStatus` | Enum for chunk states (PENDING, PROCESSING, COMPLETED, FAILED, RETRYING) |
| `ChunkState` | Dataclass tracking individual chunk progress with timing |
| `FileProgressState` | Dataclass managing file-level progress with ETA calculations |
| `TerminalCapabilities` | ANSI terminal detection (TTY, NO_COLOR, Windows support) |
| `MultiLineProgressDisplay` | Brew-style multi-line ANSI progress with cursor control |
| `SimpleFallbackDisplay` | Non-ANSI fallback for incompatible terminals |

### Processing Flow

```
Local Media Discovery
  → Audio Extraction (PyAV primary, FFmpeg for unsupported formats)
    → Chunk Creation (<25MB each, gapless splitting)
      → Async Concurrent Transcription (5 parallel)
        → Transcript Assembly
          → Cleanup
```

## Development Commands

### Environment Setup

```bash
# Create and activate virtual environment (REQUIRED)
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Tool

```bash
# Must have virtual environment activated

# Main transcription (interactive menu)
python transcribe.py

# Combine multiple transcripts
python combine_transcripts.py
```

### Configuration

- Create `.env` file with: `OPENAI_API_KEY=your_key_here`
- Required directories (auto-created): `media-files/`, `transcripts/`, `temp/`

## Implementation Details

### Media Processing

| Strategy | Usage | Details |
|----------|-------|---------|
| **PyAV (Primary)** | Audio extraction | 16kHz, 16kbps, mono output |
| **FFmpeg (Fallback)** | Unsupported formats | Converts to MP3 before processing |

- `MediaProcessor.extract_audio()` handles both strategies
- Supported formats defined in `SUPPORTED_VIDEO_FORMATS` and `SUPPORTED_AUDIO_FORMATS` constants

### Async Transcription

- **AsyncOpenAI client**: Fresh instance per file (prevents connection hangs)
- **Concurrency**: 5 parallel chunk uploads per file
- **Retry logic**: 3 attempts per chunk with error recovery
- **Chunk splitting**: Gapless audio with `ffmpeg -ss START -to END`

### Chunk Management

- Files >25MB trigger automatic splitting
- Calculation: `math.ceil(total_size / MAX_FILE_SIZE)`
- **Critical**: If any chunk exceeds 25MB, entire file is rejected with cleanup

### Error Handling

- **Graceful degradation**: PyAV failure → FFmpeg fallback
- **Resource cleanup**: Context managers ensure temp file deletion
- **User feedback**: Color-coded status messages with Unicode symbols
- **Interrupt recovery**: Signal handler restores cursor on Ctrl+C

### Progress Display System

Brew-style multi-line progress display for concurrent chunk processing:

```
Processing: video_file.mp4 (File 1/3)
├─ Chunk 1/5   ████████████████████████████████     Completed    2.3MB  (45s)
├─ Chunk 2/5   ████████████████░░░░░░░░░░░░░░░░     Processing   2.5MB  (23s)
├─ Chunk 3/5   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     Pending      2.1MB
└─ Chunk 4/5   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     Pending      2.4MB
Overall: 25% (1/4 chunks) | Elapsed: 1m 08s | Est: 3m 24s remaining
```

**Architecture**:
- **State Management**: `asyncio.Lock` ensures thread-safe updates from concurrent chunks
- **ANSI Rendering**: Cursor control sequences (`\033[nA`, `\033[2K`) for in-place updates
- **Fallback Mode**: `SimpleFallbackDisplay` for non-ANSI terminals (CI, NO_COLOR)
- **Factory Pattern**: `create_progress_display()` auto-selects based on terminal capabilities
- **Progress Calculation**: Completion-based (pending=0%, processing=50%, completed=100%)

**Terminal Support**:
- macOS Terminal.app, iTerm2: Full ANSI support
- VS Code integrated terminal: Full ANSI support
- Windows Terminal: Full support (with ctypes enablement)
- cmd.exe, CI environments: Fallback mode (status changes only)

## Working with the Codebase

### Adding Features

1. Check `DependencyManager.REQUIRED_PACKAGES` if adding new dependencies
2. Add formats to `SUPPORTED_VIDEO_FORMATS` or `SUPPORTED_AUDIO_FORMATS`
3. Use `print_status()` for user feedback
4. Follow async patterns in `AsyncTranscriber` for API operations

### Testing Considerations

- Test video and audio inputs
- Test files above and below 25MB threshold
- Verify FFmpeg fallback for unsupported formats
- Check cleanup in error scenarios
- Test progress display: single file, multiple chunks, multiple files
- Test fallback display: set `NO_COLOR=1` environment variable
- Test interrupt handling: Ctrl+C should restore cursor visibility

### Transcript Processing

- `combine_transcripts.py` uses regex patterns to clean output
- Excludes: timestamps, headers, brackets, empty lines, separators
- Output is single continuous text block

## File Structure

```
media-to-text/
├── transcribe.py              # Main async transcription engine
├── combine_transcripts.py     # Transcript merger utility
├── requirements.txt           # Python dependencies
├── .env                       # API key (gitignored)
├── media-files/              # Input directory for local media
├── transcripts/              # Output directory
└── temp/                     # Temporary files (auto-cleaned)
```

## Dependencies

Core packages (see `requirements.txt`):

| Package | Purpose |
|---------|---------|
| `openai>=1.60.2` | Whisper API client |
| `python-dotenv>=1.0.1` | Environment variable management |
| `av>=13.0.0` | PyAV for media processing |
| `aiofiles>=24.1.0` | Async file I/O |

**External requirement**: FFmpeg must be installed separately and available in PATH.
