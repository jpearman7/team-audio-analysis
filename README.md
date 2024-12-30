# Audio Merger

A Python tool for merging multiple WAV audio files into single MP3 files while maintaining directory structure.

## Features

- Recursively processes directories of WAV files
- Maintains original folder structure in output
- Handles corrupted files gracefully
- Provides detailed logging
- Merges files in sequential order based on timestamps

## Requirements

- Python 3.8 or higher
- ffmpeg
- UV package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/audio-merger.git
cd audio-merger
```

2. Create and activate virtual environment using UV:
```bash
uv venv
source .venv/Scripts/activate  # Windows
# or
source .venv/bin/activate      # Unix
```

3. Install dependencies:
```bash
uv pip install pydub
```

## Usage

1. Place your WAV files in the `audio_dat` directory following the structure:
```
audio_dat/
└── DATE/
    └── SESSION/
        └── audio_files.wav
```

2. Run the script:
```bash
python audio_merge.py
```

3. Check the `audio_output` directory for merged MP3 files.

## Output Structure

```
audio_output/
└── DATE_merged/
    └── SESSION_merged.mp3
```

## Logging

The script creates an `audio_merger.log` file with detailed processing information and any errors encountered.
