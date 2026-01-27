"""
Combine Transcripts - A utility script to merge multiple transcript files into one.

This script provides functionality to:
1. Discover transcript files in a specified directory
2. Merge multiple transcripts concurrently into a single continuous text
3. Filter out unwanted text blocks and formatting
4. Handle errors gracefully with informative messages

Author: OkhDev
Version: 1.3 - Transcript combination utility (part of media-to-text v1.3)
"""

import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import re
import asyncio
import aiofiles

# ============================================================================
# Constants and Configuration
# ============================================================================

TRANSCRIPTS_DIR = Path("transcripts")
DEFAULT_OUTPUT = Path("transcripts.txt")

# Text patterns to exclude
EXCLUDE_PATTERNS = [
    r"={2,}",  # Separator lines with two or more equals signs
    r"#\s*Transcript from:.*$",  # Headers starting with "# Transcript from:"
    r"^\s*$",  # Empty lines
    r"\[.*?\]",  # Text in square brackets
    r"\(.*?\)",  # Text in parentheses
    r"\d{2}:\d{2}",  # Timestamp patterns
]

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

class Symbols:
    """Unicode symbols for status messages."""
    CHECK = '✓'
    CROSS = '✗'
    INFO = 'ℹ'
    WARNING = '⚠'

def print_status(message: str, status: str = "info") -> None:
    """Print a formatted status message."""
    status_config = {
        "success": (Colors.GREEN, Symbols.CHECK),
        "error": (Colors.RED, Symbols.CROSS),
        "warning": (Colors.YELLOW, Symbols.WARNING),
        "info": (Colors.BLUE, Symbols.INFO),
    }
    
    color, symbol = status_config.get(status, (Colors.RESET, Symbols.INFO))
    print(f"{color}{symbol} {message}{Colors.RESET}")

def get_output_filename() -> Path:
    """Return the fixed output filename."""
    return DEFAULT_OUTPUT

def clean_text(text: str) -> str:
    """
    Clean text by removing unwanted patterns and normalizing whitespace.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text with unwanted patterns removed and whitespace normalized
    """
    # Remove all excluded patterns
    cleaned = text
    for pattern in EXCLUDE_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
    
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def count_words(text: str) -> int:
    """Count the number of words in the text."""
    return len(text.split())

# ============================================================================
# Core Functionality
# ============================================================================

class TranscriptCombiner:
    """
    Handles the combination of multiple transcript files into a single continuous text.
    
    This class manages:
    1. File discovery and validation
    2. Content cleaning and merging
    3. Output file generation
    4. Error handling and user feedback
    """
    
    def __init__(self, input_dir: Path = TRANSCRIPTS_DIR, output_file: Optional[Path] = None):
        self.input_dir = input_dir
        self.output_file = output_file or get_output_filename()
    
    def get_transcript_files(self) -> List[Path]:
        """Discover and validate transcript files in the input directory."""
        if not self.input_dir.exists():
            print_status(f"Directory '{self.input_dir}' not found", "error")
            return []
        
        transcript_files = sorted(self.input_dir.glob("*.txt"))
        
        if transcript_files:
            print_status(f"Found {len(transcript_files)} transcript file(s)", "success")
        else:
            print_status("No transcript files found", "warning")
        
        return transcript_files
    
    async def _read_and_clean_file(self, transcript_file: Path) -> Optional[str]:
        """
        Read and clean a single transcript file asynchronously.

        Args:
            transcript_file (Path): Path to the transcript file

        Returns:
            Optional[str]: Cleaned content or None if error
        """
        try:
            async with aiofiles.open(transcript_file, 'r', encoding='utf-8') as infile:
                content = await infile.read()
                # Clean and add the content
                cleaned_content = clean_text(content)
                if cleaned_content:  # Only add non-empty content
                    print_status(f"Processed: {transcript_file.name}", "success")
                    return cleaned_content
                return None
        except Exception as e:
            print_status(f"Error processing {transcript_file.name}: {str(e)}", "error")
            return None

    async def combine_files_async(self, transcript_files: List[Path]) -> bool:
        """
        Merge multiple transcript files concurrently into a single continuous text document.

        Args:
            transcript_files (List[Path]): List of transcript files to combine

        Returns:
            bool: True if combination was successful, False if any errors occurred
        """
        if not transcript_files:
            return False

        try:
            print_status("Starting concurrent file combination...", "info")

            # Read all files concurrently (MAJOR SPEEDUP for many files!)
            tasks = [self._read_and_clean_file(f) for f in transcript_files]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out None values and exceptions
            combined_text = [
                result for result in results
                if result is not None and not isinstance(result, Exception)
            ]

            if not combined_text:
                print_status("No valid content found in transcript files", "warning")
                return False

            # Join all cleaned text with a space
            final_text = ' '.join(combined_text)

            # Write output file asynchronously
            async with aiofiles.open(self.output_file, 'w', encoding='utf-8') as outfile:
                await outfile.write(final_text)

            # Print success message and word count
            print_status(
                f"Successfully combined {len(transcript_files)} files into '{self.output_file}'",
                "success"
            )
            word_count = count_words(final_text)
            print_status(f"Total word count: {word_count:,} words", "info")

            return True

        except Exception as e:
            print_status(f"Error combining files: {str(e)}", "error")
            return False

    def combine_files(self, transcript_files: List[Path]) -> bool:
        """
        Synchronous wrapper for async combine_files_async.

        Args:
            transcript_files (List[Path]): List of transcript files to combine

        Returns:
            bool: True if combination was successful, False if any errors occurred
        """
        return asyncio.run(self.combine_files_async(transcript_files))

# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """Main entry point for the transcript combination utility."""
    try:
        combiner = TranscriptCombiner()
        transcript_files = combiner.get_transcript_files()
        
        if transcript_files:
            combiner.combine_files(transcript_files)
        else:
            print_status("Please add transcript files to the 'transcripts' directory", "info")
            
    except KeyboardInterrupt:
        print_status("\nOperation cancelled by user", "warning")
    except Exception as e:
        print_status(f"An unexpected error occurred: {str(e)}", "error")

if __name__ == "__main__":
    main() 