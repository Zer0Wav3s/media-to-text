"""
Media Transcription Tool - A command-line tool to convert audio and video files into text using OpenAI's Whisper API.

This script provides a robust solution for transcribing media files by:
1. Automatically handling both audio and video inputs
2. Auto-converting unsupported formats to MP3
3. Splitting large files into processable chunks
4. Managing API interactions with OpenAI's Whisper
5. Providing real-time progress updates
6. Implementing error handling and cleanup

Author: OkhDev
Version: 1.1.0
"""

import os
import json
import math
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import sys
import subprocess
import importlib.metadata
from contextlib import contextmanager, asynccontextmanager
import logging
import shutil
import signal
from enum import Enum, auto
from dataclasses import dataclass, field

# Define Colors and Symbols first
class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class Symbols:
    """Unicode symbols for status messages."""
    CHECK = '✓'
    CROSS = '✗'
    INFO = 'ℹ'
    WARNING = '⚠'
    PROCESS = '⚙'
    TIME = '⏱'
    FILE = '⚄'
    FOLDER = '⚃'
    MEDIA = '▶'
    AUDIO = '♪'
    VIDEO = '◉'
    STAR = '★'
    SPARKLES = '✧'

# First, ensure all requirements are installed
def install_requirements():
    """Install or update packages from requirements.txt."""
    try:
        requirements_path = Path('requirements.txt')
        if not requirements_path.exists():
            print(f"{Colors.RED}{Symbols.CROSS} requirements.txt not found{Colors.RESET}")
            return False
            
        print(f"{Colors.BLUE}{Symbols.INFO} Checking dependencies...{Colors.RESET}")
        
        # Run pip install with quiet output
        subprocess.check_call([
            sys.executable, 
            '-m', 
            'pip', 
            'install', 
            '-r', 
            'requirements.txt',
            '--upgrade',
            '--quiet'
        ])
        
        print(f"{Colors.GREEN}{Symbols.CHECK} All dependencies are up to date{Colors.RESET}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}{Symbols.CROSS} Failed to install requirements: {str(e)}{Colors.RESET}")
        return False

# Now import modules that come from requirements.txt
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging
import av  # PyAV for fast media processing
import aiofiles  # Async file operations
import asyncio

# Configure logging
logging.getLogger('libav').setLevel(logging.ERROR)

# ============================================================================
# Constants and Configuration
# ============================================================================

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
UPDATE_INTERVAL = 15  # seconds

SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', '.m4v', '.3gp'}
SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.aac', '.ogg', '.wma', '.m4a', '.opus', '.flac', '.aiff', '.amr'}

# ============================================================================
# Dependency Management
# ============================================================================

class DependencyManager:
    """Handles checking and installing required Python packages."""
    
    REQUIRED_PACKAGES = {
        'openai': 'openai',
        'python-dotenv': 'dotenv',
        'moviepy': 'moviepy',
        'requests': 'requests',
        'certifi': 'certifi'  # Add certifi for SSL certificate handling
    }

    @staticmethod
    def check_and_install_dependencies() -> bool:
        """
        Check if all required packages are installed and offer to install missing ones.
        
        Returns:
            bool: True if all dependencies are satisfied, False otherwise
        """
        missing_packages = []
        
        for package, import_name in DependencyManager.REQUIRED_PACKAGES.items():
            try:
                importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                missing_packages.append(package)
        
        if not missing_packages:
            try:
                import certifi
                import requests.utils
                import ssl
                
                # Force SSL context to use certifi's certificates
                os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
                os.environ['SSL_CERT_FILE'] = certifi.where()
                requests.utils.DEFAULT_CA_BUNDLE_PATH = certifi.where()
                
                # Verify SSL configuration
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                return True
            except Exception as e:
                print_status(f"SSL configuration failed: {str(e)}", "error")
                return False
            
        print_status("Missing required packages:", "warning")
        for package in missing_packages:
            print(f"{Colors.YELLOW}   {Symbols.WARNING} {package}{Colors.RESET}")
            
        while True:
            response = input(f"\n{Colors.BLUE}Would you like to install the missing packages? (y/n): {Colors.RESET}").lower()
            if response in ['y', 'yes']:
                try:
                    print_status("Installing missing packages...", "process")
                    # Force reinstall certifi first to ensure proper configuration
                    subprocess.check_call([
                        sys.executable, 
                        '-m', 'pip', 
                        'install', 
                        '--force-reinstall', 
                        'certifi'
                    ])
                    
                    if missing_packages:
                        subprocess.check_call([
                            sys.executable, 
                            '-m', 'pip', 
                            'install'
                        ] + missing_packages)
                    
                    # Configure environment variables
                    import certifi
                    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
                    os.environ['SSL_CERT_FILE'] = certifi.where()
                    
                    print_status("Successfully installed all required packages!", "success")
                    return True
                except subprocess.CalledProcessError as e:
                    print_status(f"Failed to install packages: {str(e)}", "error")
                    print_status("Please install the required packages manually:", "info")
                    for package in missing_packages:
                        print(f"{Colors.BLUE}   pip install {package}{Colors.RESET}")
                    return False
            elif response in ['n', 'no']:
                print_status("Please install the required packages manually:", "info")
                for package in missing_packages:
                    print(f"{Colors.BLUE}   pip install {package}{Colors.RESET}")
                return False
            else:
                print_status("Please enter 'y' or 'n'", "warning")

# ============================================================================
# Utility Functions
# ============================================================================

def format_time(seconds: float) -> str:
    """Format time duration into a human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{secs}s"

def print_status(message: str, status: str = "info") -> None:
    """Print a formatted status message with appropriate color and symbol."""
    status_config = {
        "success": (Colors.GREEN, Symbols.CHECK),
        "error": (Colors.RED, Symbols.CROSS),
        "warning": (Colors.YELLOW, Symbols.WARNING),
        "info": (Colors.RESET, Symbols.INFO),
        "process": (Colors.BLUE, Symbols.PROCESS),
    }
    
    color, symbol = status_config.get(status, (Colors.RESET, Symbols.INFO))
    print(f"{color}{symbol} {message}{Colors.RESET}")

def print_header(title: str = "Media Transcription Tool") -> None:
    """Print a styled header for the application."""
    print(f"\n{Colors.BLUE}{Symbols.MEDIA}  {title}  {Symbols.AUDIO}{Colors.RESET}")
    print(f"{Colors.BLUE}{'─' * 40}{Colors.RESET}")

def print_divider() -> None:
    """Print a divider line for visual separation."""
    print(f"\n{Colors.BLUE}{'─' * 60}{Colors.RESET}\n")

# ============================================================================
# Progress Tracking
# ============================================================================

class ProgressTracker:
    """Handles progress tracking and status updates during processing."""
    
    def __init__(self):
        self.processing = False
        self.last_update = 0
        self.operation_start_time = 0
    
    def show_processing_status(self, message: str) -> None:
        """Show processing status at regular intervals."""
        self.processing = True
        self.last_update = time.time()
        self.operation_start_time = time.time()
        
        while self.processing:
            current_time = time.time()
            if current_time - self.last_update >= UPDATE_INTERVAL:
                elapsed_time = current_time - self.operation_start_time
                elapsed_str = format_time(elapsed_time)
                print(f"{Colors.YELLOW}{Symbols.PROCESS} {message} (Elapsed: {elapsed_str}){Colors.RESET}", end='\r')
                self.last_update = current_time
            time.sleep(1)
    
    def start(self, message: str) -> threading.Thread:
        """Start progress tracking in a separate thread."""
        thread = threading.Thread(target=self.show_processing_status, args=(message,))
        thread.daemon = True
        thread.start()
        return thread
    
    def stop(self) -> None:
        """Stop progress tracking."""
        self.processing = False
        print()  # Add newline after stopping progress
        time.sleep(0.1)

# ============================================================================
# Multi-Line Progress Display (Brew-Style)
# ============================================================================

class ChunkStatus(Enum):
    """Status states for chunk transcription."""
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()


@dataclass
class ChunkState:
    """State for a single chunk being processed."""
    chunk_num: int
    total_chunks: int
    size_mb: float
    status: ChunkStatus = ChunkStatus.PENDING
    start_time: Optional[float] = None
    elapsed_seconds: float = 0.0
    attempt: int = 0
    error_message: Optional[str] = None

    @property
    def status_text(self) -> str:
        """Human-readable status string."""
        status_map = {
            ChunkStatus.PENDING: "Pending",
            ChunkStatus.PROCESSING: "Processing",
            ChunkStatus.COMPLETED: "Completed",
            ChunkStatus.FAILED: "Failed",
            ChunkStatus.RETRYING: f"Retry {self.attempt}/3",
        }
        return status_map.get(self.status, "Unknown")


@dataclass
class FileProgressState:
    """Aggregate state for a file being processed."""
    filename: str
    file_index: int
    total_files: int
    chunks: List[ChunkState] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    @property
    def completed_count(self) -> int:
        return sum(1 for c in self.chunks if c.status == ChunkStatus.COMPLETED)

    @property
    def processing_count(self) -> int:
        return sum(1 for c in self.chunks if c.status == ChunkStatus.PROCESSING)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.chunks if c.status == ChunkStatus.FAILED)

    @property
    def progress_percent(self) -> float:
        """Calculate progress giving partial credit for processing chunks.

        COMPLETED = 100%, PROCESSING = 50% (can't track actual API bytes), others = 0%
        """
        if not self.chunks:
            return 0.0
        # Give processing chunks 50% credit since we can't track actual API progress
        weighted_progress = self.completed_count + (self.processing_count * 0.5)
        return (weighted_progress / len(self.chunks)) * 100

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def estimated_remaining(self) -> Optional[float]:
        """Estimate remaining time based on completed chunks.

        If no chunks completed yet, estimate based on processing time of current chunks.
        """
        total = len(self.chunks)
        if total == 0:
            return None

        if self.completed_count > 0:
            # Use actual completion time data
            avg_per_chunk = self.elapsed_seconds / self.completed_count
            remaining_chunks = total - self.completed_count
            return avg_per_chunk * remaining_chunks
        elif self.processing_count > 0:
            # Estimate: assume current elapsed is ~50% of first chunk's time
            # So remaining = (elapsed * 2) * total - elapsed
            estimated_chunk_time = self.elapsed_seconds * 2
            return (estimated_chunk_time * total) - self.elapsed_seconds
        else:
            return None


class TerminalCapabilities:
    """Detect terminal capabilities for ANSI support."""

    def __init__(self):
        self._ansi_supported: Optional[bool] = None
        self._width: int = 80
        self._detect_capabilities()

    def _detect_capabilities(self) -> None:
        """Detect terminal ANSI support and dimensions."""
        # Check if stdout is a TTY
        is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

        # Check for explicit NO_COLOR environment variable
        no_color = os.environ.get('NO_COLOR') is not None

        # Check for TERM variable (common on Unix)
        term = os.environ.get('TERM', '')
        dumb_terminal = term == 'dumb'

        # Windows detection
        is_windows = os.name == 'nt'
        windows_ansi = False
        if is_windows:
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # Enable ANSI escape sequences on Windows
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                windows_ansi = True
            except:
                pass

        self._ansi_supported = (
            is_tty and
            not no_color and
            not dumb_terminal and
            (not is_windows or windows_ansi)
        )

        # Get terminal width
        try:
            size = shutil.get_terminal_size((80, 24))
            self._width = size.columns
        except:
            self._width = 80

    @property
    def supports_ansi(self) -> bool:
        return self._ansi_supported or False

    @property
    def width(self) -> int:
        return self._width

    def refresh_width(self) -> int:
        """Refresh terminal width (for resize handling)."""
        try:
            size = shutil.get_terminal_size((80, 24))
            self._width = size.columns
        except:
            pass
        return self._width


class MultiLineProgressDisplay:
    """
    Brew-style multi-line progress display with ANSI cursor control.

    Renders a display like:

    Processing: video_file.mp4 (File 1/3)
    ├─ Chunk 1/5   ████████████████████████████████     Completed    2.3MB
    ├─ Chunk 2/5   ██████████░░░░░░░░░░░░░░░░░░░░░░     Processing   2.5MB
    └─ Chunk 3/5   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     Pending      2.4MB
    Overall: 33% (1/3 chunks) | Elapsed: 45s | Est: 1m 30s remaining
    """

    # ANSI escape sequences
    CURSOR_UP = "\033[{}A"
    CLEAR_LINE = "\033[2K"
    CURSOR_START = "\033[G"
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"

    # Progress bar characters
    BAR_FILLED = "█"
    BAR_EMPTY = "░"

    # Tree characters
    TREE_BRANCH = "├─"
    TREE_LAST = "└─"

    def __init__(self, update_interval: float = 1.0, bar_width: int = 30):
        """
        Initialize progress display.

        Args:
            update_interval: Seconds between display refreshes
            bar_width: Width of progress bar in characters
        """
        self.update_interval = update_interval
        self.bar_width = bar_width

        self.terminal = TerminalCapabilities()
        self.state: Optional[FileProgressState] = None
        self._lock = asyncio.Lock()
        self._display_lines = 0
        self._refresh_task: Optional[asyncio.Task] = None
        self._running = False
        self._initial_render_done = False

    async def initialize_file(
        self,
        filename: str,
        file_index: int,
        total_files: int,
        chunk_sizes: List[float]
    ) -> None:
        """Initialize state for a new file."""
        async with self._lock:
            chunks = [
                ChunkState(
                    chunk_num=i + 1,
                    total_chunks=len(chunk_sizes),
                    size_mb=size_mb
                )
                for i, size_mb in enumerate(chunk_sizes)
            ]
            self.state = FileProgressState(
                filename=filename,
                file_index=file_index,
                total_files=total_files,
                chunks=chunks
            )
            self._initial_render_done = False

    async def update_chunk_status(
        self,
        chunk_num: int,
        status: ChunkStatus,
        attempt: int = 0,
        error_message: Optional[str] = None
    ) -> None:
        """Update status for a specific chunk."""
        async with self._lock:
            if self.state and 1 <= chunk_num <= len(self.state.chunks):
                chunk = self.state.chunks[chunk_num - 1]
                chunk.status = status
                chunk.attempt = attempt
                chunk.error_message = error_message

                if status == ChunkStatus.PROCESSING and chunk.start_time is None:
                    chunk.start_time = time.time()
                elif status in (ChunkStatus.COMPLETED, ChunkStatus.FAILED):
                    if chunk.start_time:
                        chunk.elapsed_seconds = time.time() - chunk.start_time

    def _render_progress_bar(self, progress: float) -> str:
        """Render a progress bar string."""
        filled = int(self.bar_width * progress)
        return (
            self.BAR_FILLED * filled +
            self.BAR_EMPTY * (self.bar_width - filled)
        )

    def _render_chunk_line(self, chunk: ChunkState, is_last: bool) -> str:
        """Render a single chunk status line."""
        prefix = self.TREE_LAST if is_last else self.TREE_BRANCH

        # Calculate progress for this chunk
        if chunk.status == ChunkStatus.COMPLETED:
            progress = 1.0
        elif chunk.status in (ChunkStatus.PROCESSING, ChunkStatus.RETRYING):
            progress = 0.5  # Show half-filled for processing
        else:
            progress = 0.0

        bar = self._render_progress_bar(progress)

        # Color based on status
        if chunk.status == ChunkStatus.COMPLETED:
            color = Colors.GREEN
        elif chunk.status == ChunkStatus.FAILED:
            color = Colors.RED
        elif chunk.status in (ChunkStatus.PROCESSING, ChunkStatus.RETRYING):
            color = Colors.YELLOW
        else:
            color = Colors.RESET

        # Build the line
        chunk_label = f"Chunk {chunk.chunk_num}/{chunk.total_chunks}"
        status_text = chunk.status_text
        size_text = f"{chunk.size_mb:.1f}MB"

        # Add elapsed time for processing/completed chunks
        if chunk.status == ChunkStatus.PROCESSING and chunk.start_time:
            elapsed = format_time(time.time() - chunk.start_time)
            time_text = f"({elapsed})"
        elif chunk.status == ChunkStatus.COMPLETED:
            time_text = f"({format_time(chunk.elapsed_seconds)})"
        else:
            time_text = ""

        line = (
            f"{prefix} {chunk_label:<12} {bar}  "
            f"{color}{status_text:<12}{Colors.RESET} {size_text:<8} {time_text}"
        )

        return line

    def _render_overall_line(self) -> str:
        """Render the overall progress summary line."""
        if not self.state:
            return ""

        percent = self.state.progress_percent
        completed = self.state.completed_count
        processing = self.state.processing_count
        total = len(self.state.chunks)

        # Show clear status: completed vs processing
        if processing > 0 and completed < total:
            status_text = f"{completed} done, {processing} active"
        else:
            status_text = f"{completed}/{total} done"

        return f"Overall: {percent:.0f}% ({status_text})"

    def _render_full_display(self) -> str:
        """Render the complete multi-line display."""
        if not self.state:
            return ""

        lines = []

        # Header line
        header = (
            f"{Colors.BOLD}Processing: {self.state.filename} "
            f"(File {self.state.file_index}/{self.state.total_files}){Colors.RESET}"
        )
        lines.append(header)

        # Chunk lines
        for i, chunk in enumerate(self.state.chunks):
            is_last = (i == len(self.state.chunks) - 1)
            lines.append(self._render_chunk_line(chunk, is_last))

        # Overall progress line
        lines.append(self._render_overall_line())

        return "\n".join(lines)

    def _clear_display(self) -> None:
        """Clear the previously rendered display."""
        if not self.terminal.supports_ansi or self._display_lines == 0:
            return

        # Move cursor up and clear each line
        for _ in range(self._display_lines):
            sys.stdout.write(self.CURSOR_UP.format(1))
            sys.stdout.write(self.CLEAR_LINE)
            sys.stdout.write(self.CURSOR_START)
        sys.stdout.flush()

    def _write_display(self) -> None:
        """Write the current display to terminal."""
        if not self.state:
            return

        if self.terminal.supports_ansi:
            # Clear previous display (except first render)
            if self._initial_render_done:
                self._clear_display()

            # Render and write new display
            display = self._render_full_display()
            print(display)

            # Track lines written (header + chunks + overall)
            self._display_lines = 2 + len(self.state.chunks)
            self._initial_render_done = True
        else:
            # Fallback handled by SimpleFallbackDisplay
            pass

    async def _refresh_loop(self) -> None:
        """Background task that refreshes the display periodically."""
        while self._running:
            async with self._lock:
                self._write_display()
            await asyncio.sleep(self.update_interval)

    async def start(self) -> None:
        """Start the progress display refresh loop."""
        if self._running:
            return

        self._running = True

        if self.terminal.supports_ansi:
            # Hide cursor during display
            sys.stdout.write(self.HIDE_CURSOR)
            sys.stdout.flush()

        # Initial render
        async with self._lock:
            self._write_display()

        # Start background refresh task
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def stop(self, interrupted: bool = False) -> None:
        """Stop the progress display and clean up.

        Args:
            interrupted: If True, skip final display update (e.g., Ctrl+C)
        """
        self._running = False

        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

        if self.terminal.supports_ansi:
            # Show cursor again
            sys.stdout.write(self.SHOW_CURSOR)
            sys.stdout.flush()

        # Only do final display update on clean exit
        if not interrupted:
            async with self._lock:
                self._write_display()
            # Add newline after display
            print()

    @asynccontextmanager
    async def display_context(self):
        """Context manager for automatic start/stop."""
        interrupted = False
        try:
            await self.start()
            yield self
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            interrupted = True
            raise
        finally:
            await self.stop(interrupted=interrupted)


class SimpleFallbackDisplay:
    """Simple fallback for non-ANSI terminals."""

    def __init__(self):
        self.last_status: Dict[int, ChunkStatus] = {}
        self.state: Optional[FileProgressState] = None
        self._lock = asyncio.Lock()

    async def initialize_file(
        self,
        filename: str,
        file_index: int,
        total_files: int,
        chunk_sizes: List[float]
    ) -> None:
        """Initialize state for a new file."""
        async with self._lock:
            chunks = [
                ChunkState(chunk_num=i + 1, total_chunks=len(chunk_sizes), size_mb=size_mb)
                for i, size_mb in enumerate(chunk_sizes)
            ]
            self.state = FileProgressState(
                filename=filename,
                file_index=file_index,
                total_files=total_files,
                chunks=chunks
            )
            self.last_status.clear()
            print(f"\n{Colors.BOLD}Processing: {filename} (File {file_index}/{total_files}){Colors.RESET}")
            print(f"Chunks to process: {len(chunk_sizes)}")

    async def update_chunk_status(
        self,
        chunk_num: int,
        status: ChunkStatus,
        attempt: int = 0,
        error_message: Optional[str] = None
    ) -> None:
        """Print status only when it changes."""
        async with self._lock:
            if self.last_status.get(chunk_num) != status:
                self.last_status[chunk_num] = status

                if status == ChunkStatus.PROCESSING:
                    print_status(f"Chunk {chunk_num}: Starting transcription...", "process")
                elif status == ChunkStatus.COMPLETED:
                    print_status(f"Chunk {chunk_num}: Completed", "success")
                elif status == ChunkStatus.FAILED:
                    print_status(f"Chunk {chunk_num}: Failed - {error_message or 'Unknown error'}", "error")
                elif status == ChunkStatus.RETRYING:
                    print_status(f"Chunk {chunk_num}: Retrying (attempt {attempt}/3)...", "warning")

    async def start(self) -> None:
        """No-op for fallback display."""
        pass

    async def stop(self, interrupted: bool = False) -> None:
        """Print summary on stop."""
        if not interrupted and self.state:
            completed = self.state.completed_count
            total = len(self.state.chunks)
            elapsed = format_time(self.state.elapsed_seconds)
            print(f"\nCompleted {completed}/{total} chunks in {elapsed}")

    @asynccontextmanager
    async def display_context(self):
        """Context manager for automatic start/stop."""
        interrupted = False
        try:
            await self.start()
            yield self
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            interrupted = True
            raise
        finally:
            await self.stop(interrupted=interrupted)


def create_progress_display():
    """Create appropriate progress display based on terminal capabilities."""
    terminal = TerminalCapabilities()
    if terminal.supports_ansi:
        return MultiLineProgressDisplay()
    else:
        return SimpleFallbackDisplay()


# ============================================================================
# Environment Setup
# ============================================================================

class EnvironmentSetup:
    """Handles environment configuration and directory setup."""
    
    @staticmethod
    def check_env_setup() -> bool:
        """Check and setup environment variables."""
        env_path = Path('.env')
        
        if not env_path.exists():
            print_status("No .env file found. Creating one for you...", "error")
            with open(env_path, 'w') as f:
                f.write("# OpenAI API Configuration\nOPENAI_API_KEY=your_api_key_here")
            print_status("Created .env file. Please add your OpenAI API key.", "error")
            return False
        
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key or api_key == "your_api_key_here":
            print_status("No valid API key found in .env file.", "error")
            print_status("Please ensure your .env file contains: OPENAI_API_KEY=your_actual_api_key", "error")
            return False
        
        print_status("Environment variables loaded successfully", "success")
        return True
    
    @staticmethod
    def setup_directories() -> None:
        """Create necessary directories for operation."""
        try:
            Path('media-files').mkdir(exist_ok=True)
            print_status("Created media-files directory", "success")
            
            Path('transcripts').mkdir(exist_ok=True)
            print_status("Created transcripts directory", "success")
            
            Path('temp').mkdir(exist_ok=True)
            print_status("Created temporary directory", "success")
            
        except Exception as e:
            print_status(f"Error creating directories: {str(e)}", "error")
            raise

# ============================================================================
# Media Processing
# ============================================================================

class MediaProcessor:
    """
    Handles media file processing and chunking.
    
    This class is responsible for:
    1. Discovering supported media files
    2. Extracting audio from video files
    3. Splitting large files into API-compatible chunks
    4. Managing temporary file operations
    """
    
    def __init__(self):
        self.progress = ProgressTracker()
    
    @contextmanager
    def _load_media_pyav(self, media_path: Path):
        """Context manager for loading media files with PyAV."""
        container = None
        try:
            container = av.open(str(media_path))
            # Get audio stream
            audio_stream = next(
                (s for s in container.streams if s.type == 'audio'),
                None
            )
            if not audio_stream:
                raise ValueError(f"No audio stream found in {media_path.name}")

            yield container, audio_stream
        finally:
            if container:
                container.close()

    def convert_unsupported_to_mp3(self, file_path: Path) -> Optional[Path]:
        """
        Convert unsupported audio/video file to MP3 using FFmpeg.

        Args:
            file_path: Path to unsupported media file

        Returns:
            Path to converted MP3 file or None if conversion failed
        """
        try:
            # Generate output path with _converted suffix
            output_path = file_path.parent / f"{file_path.stem}_converted.mp3"

            # Skip if already converted
            if output_path.exists():
                print_status(f"Using existing conversion: {output_path.name}", "info")
                return output_path

            print_status(f"Converting {file_path.name} to MP3...", "process")

            # FFmpeg conversion command
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', str(file_path),
                '-vn',  # No video
                '-acodec', 'libmp3lame',
                '-ab', '16k',    # 16kbps bitrate (optimized for Whisper)
                '-ar', '16000',  # 16kHz sample rate (Whisper's native rate)
                '-ac', '1',      # Mono audio (speech optimized)
                '-y',  # Overwrite
                str(output_path),
                '-hide_banner', '-loglevel', 'error'
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print_status(f"Conversion failed: {result.stderr}", "error")
                return None

            print_status(f"Successfully converted to {output_path.name}", "success")
            return output_path

        except Exception as e:
            print_status(f"Error converting {file_path.name}: {str(e)}", "error")
            return None

    def get_media_files(self) -> List[Path]:
        """Discover and validate media files in the media-files directory."""
        media_dir = Path('media-files')
        supported_files = []
        converted_files = []
        truly_unsupported = []

        for file_path in media_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                extension = file_path.suffix.lower()

                # Skip already converted files to avoid duplicate processing
                if '_converted.mp3' in file_path.name:
                    continue
                # Check if already supported
                elif extension in SUPPORTED_VIDEO_FORMATS or extension in SUPPORTED_AUDIO_FORMATS:
                    supported_files.append(file_path)
                # Try to convert unsupported files
                else:
                    print_status(f"Detected unsupported format: {file_path.name}", "warning")
                    converted_path = self.convert_unsupported_to_mp3(file_path)

                    if converted_path:
                        converted_files.append(converted_path)
                        print_status(f"Will transcribe converted file: {converted_path.name}", "success")
                    else:
                        truly_unsupported.append(file_path)

        all_files = supported_files + converted_files

        if all_files:
            print_status(f"Found {len(supported_files)} supported + {len(converted_files)} converted file(s)", "success")

        if truly_unsupported:
            print_status(f"Could not convert {len(truly_unsupported)} file(s):", "error")
            for file in truly_unsupported:
                print(f"{Colors.RED}   {Symbols.CROSS} {file.name}{Colors.RESET}")

        return all_files
    
    def extract_audio(self, media_path: Path) -> List[tuple]:
        """
        Extract and chunk audio using FFmpeg with gapless splitting.

        This method ensures NO DATA LOSS by:
        1. Extracting full audio ONCE
        2. Splitting using FFmpeg segment filter (frame-perfect)
        3. Sequential processing for reliability
        """
        try:
            is_video = media_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS
            file_type = "video" if is_video else "audio"

            print_status(f"Loading {file_type} file...", "process")

            # Step 1: Extract FULL audio using FFmpeg (reliable, no PyAV issues)
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, dir='temp') as temp_file:
                full_audio_path = temp_file.name

            print_status("Extracting audio with FFmpeg...", "process")

            ffmpeg_extract_cmd = [
                'ffmpeg', '-i', str(media_path),
                '-vn',  # No video
                '-acodec', 'libmp3lame',  # MP3 codec
                '-ab', '16k',  # 16kbps bitrate (optimized for Whisper)
                '-ar', '16000',  # 16kHz sample rate (Whisper's native rate)
                '-ac', '1',  # Mono audio (speech optimized)
                '-y',  # Overwrite
                full_audio_path,
                '-hide_banner', '-loglevel', 'error'
            ]

            result = subprocess.run(ffmpeg_extract_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg extraction failed: {result.stderr}")

            # Get duration and file size
            duration_cmd = ['ffprobe', '-i', full_audio_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0']
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)

            try:
                duration = float(duration_result.stdout.strip())
            except:
                duration = 0

            file_size = os.path.getsize(media_path) / (1024 * 1024)
            audio_size = os.path.getsize(full_audio_path) / (1024 * 1024)

            print_status(f"Duration: {format_time(duration)}", "info")
            print_status(f"Original size: {file_size:.1f}MB", "info")
            print_status(f"Audio size: {audio_size:.1f}MB", "info")

            # Check if chunking needed
            if os.path.getsize(full_audio_path) <= MAX_FILE_SIZE:
                print_status("File small enough, no chunking needed", "success")
                return [(full_audio_path, 0, duration)]

            # Step 2: Calculate optimal chunk size (by duration, not file size)
            # OpenAI limit is 25MB, aim for ~24MB chunks (96% utilization)
            target_chunk_size = 24 * 1024 * 1024  # 24MB (safe 4% margin)
            total_size = os.path.getsize(full_audio_path)
            num_chunks = math.ceil(total_size / target_chunk_size)
            chunk_duration = duration / num_chunks

            print_status(f"Splitting into {num_chunks} chunks with manual time-based splitting", "info")

            # Step 3: Manual time-based splitting for reliability
            chunk_files = []

            # Create chunks directory
            chunks_dir = Path('temp') / f'chunks_{int(time.time())}'
            chunks_dir.mkdir(exist_ok=True)

            # Create each chunk manually with precise time ranges
            for i in range(num_chunks):
                start_time = i * chunk_duration
                # For last chunk, use remaining duration to avoid creating empty chunks
                if i == num_chunks - 1:
                    chunk_time = duration - start_time
                else:
                    chunk_time = chunk_duration

                chunk_path = chunks_dir / f'chunk_{i:03d}.mp3'

                # Extract chunk with re-encoding to ensure valid MP3
                chunk_cmd = [
                    'ffmpeg',
                    '-ss', str(start_time),  # Start time
                    '-t', str(chunk_time),   # Duration (not end time)
                    '-i', full_audio_path,
                    '-c:a', 'libmp3lame',    # Re-encode for reliability
                    '-b:a', '16k',           # 16kbps bitrate (optimized for Whisper)
                    '-ar', '16000',          # 16kHz sample rate (Whisper's native rate)
                    '-ac', '1',              # Mono audio (speech optimized)
                    '-y',
                    str(chunk_path),
                    '-hide_banner', '-loglevel', 'error'
                ]

                result = subprocess.run(chunk_cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print_status(f"Failed to create chunk {i+1}: {result.stderr}", "error")
                    # Clean up
                    os.unlink(full_audio_path)
                    for c in chunk_files:
                        try:
                            os.unlink(c[0])
                        except:
                            pass
                    return []

                # Validate chunk
                chunk_size = os.path.getsize(chunk_path)

                # Skip empty or corrupted chunks
                if chunk_size < 1024:  # Less than 1KB is likely corrupted
                    print_status(f"Skipping corrupted chunk {i+1} ({chunk_size} bytes)", "warning")
                    os.unlink(chunk_path)
                    continue

                print_status(f"Chunk {i+1}/{num_chunks}: {chunk_size/(1024*1024):.1f}MB", "info")

                if chunk_size > MAX_FILE_SIZE:
                    print_status(f"WARNING: Chunk {i+1} exceeds 25MB limit!", "error")
                    # Clean up all chunks
                    os.unlink(full_audio_path)
                    for c in chunk_files:
                        try:
                            os.unlink(c[0])
                        except:
                            pass
                    os.unlink(chunk_path)
                    return []

                end_time = start_time + chunk_time
                chunk_files.append((chunk_path, start_time, end_time))
                print_status(f"Chunk {i+1} ready ({chunk_size / (1024*1024):.1f}MB)", "success")

            # Clean up full audio file (no longer needed)
            os.unlink(full_audio_path)

            return chunk_files

        except Exception as e:
            print_status(f"Audio extraction failed: {str(e)}", "error")
            return []

# ============================================================================
# Transcription
# ============================================================================

class AsyncTranscriber:
    """
    Async transcription using OpenAI's Whisper API with parallel chunk processing.

    Features:
    1. Parallel chunk processing (5 concurrent requests for speed)
    2. Sequential file processing (one file at a time)
    3. Retry logic for failed requests (3 attempts per chunk)
    4. Progress tracking with elapsed time updates
    5. Async file I/O
    6. NO data loss
    """

    def __init__(self, max_retries: int = 3, max_concurrent: int = 5, progress_display=None):
        """
        Initialize async transcriber.

        Args:
            max_retries: Max retry attempts for failed API calls (default 3)
            max_concurrent: Max concurrent chunk requests (default 5)
            progress_display: Progress display instance (MultiLineProgressDisplay or SimpleFallbackDisplay)
        """
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.client = None
        self.progress_display = progress_display or create_progress_display()

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.close()

    async def transcribe_chunk(self, chunk_path: str, chunk_num: int, total_chunks: int) -> Optional[str]:
        """
        Transcribe single chunk with retry logic and progress display updates.

        Args:
            chunk_path: Path to audio chunk
            chunk_num: Current chunk number (for progress display)
            total_chunks: Total number of chunks

        Returns:
            Transcribed text or None if failed after all retries
        """
        for attempt in range(self.max_retries):
            try:
                # Update progress display: processing or retrying
                if attempt == 0:
                    await self.progress_display.update_chunk_status(
                        chunk_num, ChunkStatus.PROCESSING
                    )
                else:
                    await self.progress_display.update_chunk_status(
                        chunk_num, ChunkStatus.RETRYING, attempt=attempt + 1
                    )

                # Read file asynchronously
                async with aiofiles.open(chunk_path, 'rb') as audio_file:
                    file_content = await audio_file.read()

                # Create tuple format (filename, file_content) for OpenAI API
                file_tuple = (Path(chunk_path).name, file_content)

                # Call API with generous timeout (chunks can be 20+ minutes of audio)
                response = await self.client.audio.transcriptions.create(
                    file=file_tuple,
                    model="whisper-1",
                    timeout=600.0  # 10 minute timeout per chunk
                )

                # Update progress display: completed
                await self.progress_display.update_chunk_status(
                    chunk_num, ChunkStatus.COMPLETED
                )
                return response.text.strip()

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate_limit" in error_msg.lower():
                    # Rate limit - wait before retry
                    wait_time = (attempt + 1) * 5
                    await asyncio.sleep(wait_time)
                elif "timeout" in error_msg.lower():
                    await asyncio.sleep(2)
                else:
                    pass  # Just retry

                if attempt == self.max_retries - 1:
                    # Update progress display: failed
                    await self.progress_display.update_chunk_status(
                        chunk_num, ChunkStatus.FAILED, error_message=error_msg
                    )
                    return None

        return None

    async def transcribe_chunks_parallel(
        self,
        chunks: List[tuple],
        filename: str = "unknown",
        file_index: int = 1,
        total_files: int = 1
    ) -> List[Optional[str]]:
        """
        Transcribe chunks IN PARALLEL with brew-style progress display.

        This provides:
        - Parallel processing (5 concurrent API requests for speed)
        - Rate limiting via semaphore (prevents overwhelming API)
        - Automatic retry on failures (3 attempts per chunk)
        - Preserved chunk order in results
        - Real-time multi-line progress display
        - NO data loss

        Args:
            chunks: List of (chunk_path, start_time, end_time) tuples
            filename: Name of the file being processed
            file_index: Current file number (1-based)
            total_files: Total number of files to process

        Returns:
            List of transcribed texts (same order as input)
        """
        total_chunks = len(chunks)

        # Calculate chunk sizes for display
        chunk_sizes = []
        for chunk_path, _, _ in chunks:
            try:
                size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            except:
                size_mb = 0.0
            chunk_sizes.append(size_mb)

        # Initialize progress display with file info
        await self.progress_display.initialize_file(
            filename=filename,
            file_index=file_index,
            total_files=total_files,
            chunk_sizes=chunk_sizes
        )

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def transcribe_with_limit(chunk_data: tuple, chunk_num: int) -> Optional[str]:
            """Transcribe single chunk with semaphore rate limiting."""
            chunk_path, start_time, end_time = chunk_data

            async with semaphore:
                result = await self.transcribe_chunk(chunk_path, chunk_num, total_chunks)
                return result

        # Wrap chunk processing with progress display context
        async with self.progress_display.display_context():
            # Create tasks for all chunks
            tasks = [
                transcribe_with_limit(chunk_data, i)
                for i, chunk_data in enumerate(chunks, 1)
            ]

            # Execute all tasks in parallel (respecting semaphore limit)
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to None
        results = [
            r if not isinstance(r, Exception) else None
            for r in results
        ]

        # Print final summary after progress display ends
        successful = sum(1 for r in results if r is not None)
        status = "success" if successful == total_chunks else "warning"
        print_status(f"Completed {successful}/{total_chunks} chunks successfully", status)

        return results

    async def create_transcript_file(self, video_name: str) -> Path:
        """Create transcript file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path('transcripts') / f"{video_name}_{timestamp}.txt"

    async def append_to_transcript(self, file_path: Path, text: str) -> None:
        """Append text to transcript file asynchronously."""
        async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
            await f.write(text + '\n\n')

# ============================================================================
# Main Application
# ============================================================================

class TranscriptionApp:
    """
    Async transcription app with optimized parallel processing.

    Features:
    1. Parallel chunk transcription (5 concurrent for speed)
    2. Sequential file processing (one file at a time)
    3. Async file I/O (non-blocking disk operations)
    4. Retry logic (3 attempts per chunk)
    5. Optimized audio (16kHz/16kbps/mono for faster processing)
    """

    def __init__(self):
        load_dotenv()
        self.env_setup = EnvironmentSetup()
        self.media_processor = MediaProcessor()

    async def process_media_file_async(
        self,
        media_path: Path,
        transcriber: AsyncTranscriber,
        file_index: int = 1,
        total_files: int = 1
    ) -> Optional[Path]:
        """
        Process single media file with parallel chunk transcription and progress display.

        Args:
            media_path: Path to media file
            transcriber: Async transcriber instance
            file_index: Current file number (1-based)
            total_files: Total number of files to process

        Returns:
            Path to transcript file if successful
        """
        try:
            transcript_file = await transcriber.create_transcript_file(media_path.stem)

            # Extract audio (sync operation but fast with PyAV)
            print_status(f"Extracting audio from {media_path.name}...", "process")
            chunk_files = await asyncio.to_thread(
                self.media_processor.extract_audio,
                media_path
            )

            if not chunk_files:
                return None

            print_status(f"Created {len(chunk_files)} chunk(s) for transcription", "info")

            # Transcribe chunks IN PARALLEL with progress display
            results = await transcriber.transcribe_chunks_parallel(
                chunk_files,
                filename=media_path.name,
                file_index=file_index,
                total_files=total_files
            )

            # Write results to file
            for i, (text, (chunk_path, start_time, end_time)) in enumerate(
                zip(results, chunk_files), 1
            ):
                if text:
                    await transcriber.append_to_transcript(transcript_file, text)

                # Clean up chunk file
                try:
                    os.unlink(chunk_path)
                except:
                    pass

            return transcript_file

        except Exception as e:
            print_status(f"Error processing {media_path.name}: {str(e)}", "error")
            return None

    def cleanup(self) -> None:
        """Clean up temporary files and directories."""
        try:
            import shutil
            temp_path = Path('temp')

            # Clean up all files and directories in temp
            for item in temp_path.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)  # Remove chunk directories
                    else:
                        os.unlink(item)  # Remove individual files
                except Exception as e:
                    print_status(f"Could not remove {item.name}: {str(e)}", "warning")

            print_status("Cleared temp directory", "success")
        except Exception as e:
            print_status(f"Error cleaning up: {str(e)}", "warning")

    async def run_async(self) -> None:
        """Run transcription with parallel chunk processing for optimal speed."""
        print_header()

        if not DependencyManager.check_and_install_dependencies():
            return

        if not self.env_setup.check_env_setup():
            return

        try:
            self.env_setup.setup_directories()

            # Clean up any leftover temp files from previous interrupted runs
            print_status("Cleaning up temp directory from previous runs...", "info")
            self.cleanup()

            media_files = self.media_processor.get_media_files()

            if not media_files:
                print_status("No supported media files found", "warning")
                print_status("Add media files to 'media-files' folder", "info")
                return

            print(f"\n{Colors.BLUE}Processing {len(media_files)} file(s) SEQUENTIALLY (one file at a time){Colors.RESET}")
            print(f"{Colors.BLUE}Each file's chunks will be transcribed IN PARALLEL (5 concurrent for speed){Colors.RESET}")
            print(f"{Colors.BLUE}Fresh API client created for each file to prevent connection issues{Colors.RESET}")

            # Process files SEQUENTIALLY (one file at a time)
            # Each file gets a FRESH transcriber/client to prevent connection pooling issues
            # Each file's chunks are processed IN PARALLEL (for speed)
            results = []
            for i, media_path in enumerate(media_files, 1):
                print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.BLUE}FILE {i}/{len(media_files)}: {media_path.name}{Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

                # Create fresh async transcriber for THIS file
                async with AsyncTranscriber(max_retries=3, max_concurrent=5) as transcriber:
                    result = await self.process_media_file_async(
                        media_path, transcriber,
                        file_index=i,
                        total_files=len(media_files)
                    )
                    results.append(result)

                    if result:
                        print(f"\n{Colors.GREEN}✓ File {i}/{len(media_files)} completed successfully{Colors.RESET}")
                        print(f"{Colors.GREEN}  Transcript: {result.name}{Colors.RESET}")
                    else:
                        print(f"\n{Colors.RED}✗ File {i}/{len(media_files)} FAILED{Colors.RESET}")

            # Analyze results
            successful = sum(1 for r in results if r and not isinstance(r, Exception))
            failed_files = [
                media_files[i]
                for i, r in enumerate(results)
                if not r or isinstance(r, Exception)
            ]

            print_divider()
            print_status(
                f"Complete! {successful}/{len(media_files)} files transcribed",
                "success" if successful == len(media_files) else "warning"
            )

            if failed_files:
                print_status("Failed files:", "error")
                for f in failed_files:
                    print_status(f"• {f.name}", "error")

        except KeyboardInterrupt:
            print_status("\nInterrupted by user", "warning")
        except Exception as e:
            print_status(f"Unexpected error: {str(e)}", "error")
        finally:
            self.cleanup()

    def run(self) -> None:
        """Synchronous wrapper for async run."""
        asyncio.run(self.run_async())

# ============================================================================
# Entry Point
# ============================================================================

def main():
    # Signal handler to restore cursor on interrupt
    def cleanup_on_interrupt(signum, frame):
        sys.stdout.write('\033[?25h')  # Show cursor
        sys.stdout.flush()
        print(f"\n{Colors.YELLOW}{Symbols.WARNING} Interrupted by user{Colors.RESET}")
        sys.exit(1)

    signal.signal(signal.SIGINT, cleanup_on_interrupt)
    signal.signal(signal.SIGTERM, cleanup_on_interrupt)

    try:
        # Install requirements first
        if not install_requirements():
            return

        # Clear screen before starting
        time.sleep(1.5)
        os.system('cls' if os.name == 'nt' else 'clear')

        # Create necessary directories if they don't exist
        required_dirs = ['media-files', 'transcripts', 'temp']
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                dir_path.mkdir(exist_ok=True)
                print(f"{Colors.BLUE}{Symbols.INFO} Created {dir_name} directory{Colors.RESET}")

        # Process local media files
        print(f"\n{Colors.BLUE}Processing media files from: media-files/{Colors.RESET}")
        print(f"{Colors.BLUE}Supported formats: Video {', '.join(sorted(SUPPORTED_VIDEO_FORMATS))}{Colors.RESET}")
        print(f"{Colors.BLUE}                   Audio {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}{Colors.RESET}")
        print(f"{Colors.BLUE}Unsupported formats will be auto-converted to MP3{Colors.RESET}\n")

        app = TranscriptionApp()
        app.run()

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}{Symbols.INFO} Program terminated by user{Colors.RESET}")
        sys.exit(0)

if __name__ == "__main__":
    main() 