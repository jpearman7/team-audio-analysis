import os
from pathlib import Path
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_merger.log'),
        logging.StreamHandler()
    ]
)

def create_output_directory(input_path, base_output_dir):
    """Create output directory with _merged suffix if it doesn't exist."""
    dir_name = input_path.name
    output_dir = base_output_dir / f"{dir_name}_merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def safe_load_audio(wav_file):
    """Safely load an audio file with error handling."""
    try:
        return AudioSegment.from_file(str(wav_file), format="wav")
    except (CouldntDecodeError, OSError) as e:
        logging.error(f"Failed to load {wav_file}: {str(e)}")
        return None

def merge_audio_files(input_dir, output_dir):
    """Merge all WAV files in a directory into a single MP3."""
    # Get all WAV files and sort them by name
    wav_files = sorted([f for f in input_dir.glob("*.wav")])
    
    if not wav_files:
        logging.warning(f"No WAV files found in {input_dir}")
        return False

    logging.info(f"Processing {input_dir.name}...")
    
    # Keep track of successfully loaded audio segments
    audio_segments = []
    
    # Try to load each audio file
    for wav_file in wav_files:
        logging.info(f"Loading {wav_file.name}")
        audio = safe_load_audio(wav_file)
        if audio is not None:
            audio_segments.append(audio)
        
    if not audio_segments:
        logging.error(f"No valid audio files found in {input_dir}")
        return False
        
    try:
        # Combine all valid audio segments
        combined = audio_segments[0]
        for segment in audio_segments[1:]:
            combined += segment
            
        # Export as MP3
        output_file = output_dir / f"{input_dir.name}_merged.mp3"
        combined.export(str(output_file), format="mp3")
        logging.info(f"Successfully created {output_file}")
        return True
        
    except Exception as e:
        logging.error(f"Error merging audio files in {input_dir}: {str(e)}")
        return False

def process_directory(input_dir, output_base):
    """Process a directory and its subdirectories."""
    try:
        # Create parallel output directory
        output_dir = create_output_directory(input_dir, output_base)
        
        # Process all subdirectories
        for subdir in input_dir.iterdir():
            if subdir.is_dir():
                # If directory contains WAV files, merge them
                if any(f.suffix.lower() == '.wav' for f in subdir.iterdir()):
                    merge_audio_files(subdir, output_dir)
                # Otherwise, process it as a container directory
                else:
                    process_directory(subdir, output_dir)
    except Exception as e:
        logging.error(f"Error processing directory {input_dir}: {str(e)}")

def main():
    # Get the script's directory
    script_dir = Path(__file__).parent.resolve()
    
    # Define base directories using absolute paths
    audio_dat = script_dir / "audio_dat"
    audio_output = script_dir / "audio_output"
    # audio_dat = Path("E:/11-8-2024_audio_backup")
    # audio_output = Path("E:/merged_audio")
    
    logging.info("Starting audio merger process")
    logging.info(f"Input directory: {audio_dat}")
    logging.info(f"Output directory: {audio_output}")
    
    # Create main output directory if it doesn't exist
    audio_output.mkdir(exist_ok=True)
    
    # Process each date directory in audio_dat
    try:
        for date_dir in audio_dat.iterdir():
            if date_dir.is_dir():
                logging.info(f"\nProcessing date directory: {date_dir.name}")
                process_directory(date_dir, audio_output)
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
    
    logging.info("Audio merger process completed")

if __name__ == "__main__":
    main()