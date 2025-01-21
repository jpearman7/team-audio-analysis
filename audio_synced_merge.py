import os
from datetime import datetime
from pydub import AudioSegment
import logging
from typing import Dict, List, Tuple
import wave

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        filename='audio_merger.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def is_valid_wav(file_path: str) -> bool:
    """Check if a WAV file is valid by attempting to open it."""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            return True
    except Exception as e:
        logging.warning(f"Invalid WAV file {file_path}: {str(e)}")
        return False

def parse_timestamp(filename: str) -> datetime:
    """Extract timestamp from filename."""
    # Extract the timestamp part: "11-10-2023-14-52-03"
    timestamp_str = filename.split('.')[0]
    return datetime.strptime(timestamp_str, '%m-%d-%Y-%H-%M-%S')

def find_latest_start_time_for_date(date_path: str) -> Tuple[datetime, str]:
    """
    Find the latest initial timestamp across session folders within a specific date folder.
    Returns the timestamp and the folder it was found in.
    """
    latest_time = None
    latest_folder = None
    
    # Iterate through session folders (121, 122, etc.)
    for session_dir in os.listdir(date_path):
        session_path = os.path.join(date_path, session_dir)
        if not os.path.isdir(session_path):
            continue
            
        # Get valid WAV files in the session folder
        wav_files = [f for f in os.listdir(session_path) 
                    if f.endswith('.wav') and is_valid_wav(os.path.join(session_path, f))]
        if not wav_files:
            continue
            
        # Get chronologically first valid file
        first_file = min(wav_files)
        timestamp = parse_timestamp(first_file)
        
        if latest_time is None or timestamp > latest_time:
            latest_time = timestamp
            latest_folder = session_dir

    return latest_time, latest_folder

def calculate_trim_duration(start_file: str, sync_time: datetime) -> float:
    """Calculate how many milliseconds to trim from the start of the audio."""
    file_time = parse_timestamp(start_file)
    time_diff = (sync_time - file_time).total_seconds() * 1000
    return max(0, time_diff)  # Ensure we don't get negative duration

def merge_and_sync_audio(input_dir: str, output_dir: str) -> None:
    """
    Merge and synchronize audio files from each session folder,
    trimming the start based on the latest initial timestamp within each date.
    """
    setup_logging()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each date directory
    for date_dir in os.listdir(input_dir):
        date_path = os.path.join(input_dir, date_dir)
        if not os.path.isdir(date_path):
            continue
            
        # Find the latest start time for this specific date
        sync_time, sync_folder = find_latest_start_time_for_date(date_path)
        if not sync_time:
            logging.warning(f"No valid WAV files found in date folder {date_dir}")
            continue
        
        logging.info(f"Using sync time {sync_time} from folder {sync_folder} for date {date_dir}")
        print(f"Date {date_dir}: Synchronizing audio to start time: {sync_time} (from folder {sync_folder})")
        
        # Create corresponding output date directory
        output_date_dir = os.path.join(output_dir, f"{date_dir}_merged")
        if not os.path.exists(output_date_dir):
            os.makedirs(output_date_dir)
        
        # Process each session directory
        for session_dir in os.listdir(date_path):
            session_path = os.path.join(date_path, session_dir)
            if not os.path.isdir(session_path):
                continue
                
            try:
                # Get valid WAV files in the session directory
                wav_files = [f for f in os.listdir(session_path) 
                           if f.endswith('.wav') and is_valid_wav(os.path.join(session_path, f))]
                if not wav_files:
                    logging.warning(f"No valid WAV files found in session {session_dir}")
                    continue
                    
                # Sort files chronologically
                wav_files.sort()
                
                # Calculate trim duration based on first file
                trim_duration = calculate_trim_duration(wav_files[0], sync_time)
                
                # Merge files
                merged_audio = None
                for wav_file in wav_files:
                    try:
                        file_path = os.path.join(session_path, wav_file)
                        audio = AudioSegment.from_wav(file_path)
                        
                        if merged_audio is None:
                            merged_audio = audio
                        else:
                            merged_audio += audio
                            
                        logging.info(f"Successfully merged {wav_file}")
                        
                    except Exception as e:
                        logging.error(f"Error processing file {wav_file}: {str(e)}")
                        continue
                
                if merged_audio is not None:
                    # Trim the start of the merged audio
                    if trim_duration > 0:
                        merged_audio = merged_audio[trim_duration:]
                    
                    # Save the merged and trimmed audio
                    output_file = os.path.join(output_date_dir, f"{session_dir}_merged.mp3")
                    merged_audio.export(output_file, format="mp3")
                    
                    logging.info(f"Successfully processed session {session_dir}")
                    print(f"Processed session {session_dir} (trimmed {trim_duration:.2f}ms)")
                    
            except Exception as e:
                logging.error(f"Error processing session {session_dir}: {str(e)}")
                print(f"Error processing session {session_dir}: {str(e)}")

def main():
    input_dir = "audio_dat"  # Base directory containing date folders
    # input_dir = "E:/11-8-2024_audio_backup"
    output_dir = "audio_output"  # Directory for merged output
    # output_dir = "E:/merged_synced_audio"  # Directory for merged output
    
    try:
        merge_and_sync_audio(input_dir, output_dir)
        print("Audio merging and synchronization complete!")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()