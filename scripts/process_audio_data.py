import os
import json
from pathlib import Path
from pydub import AudioSegment
import re

def extract_age_gender(line):
    # Pattern to match age and gender (e.g., "01M-BL2")
    pattern = r'(\d{2})([MF])-.*'
    match = re.match(pattern, line.strip())
    if match:
        age = match.group(1)
        gender = match.group(2)
        return age, gender
    return None, None

def process_text_file(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 3:
        raise ValueError(f"Invalid text file format in {txt_path}")
    
    words = lines[0].strip().split()
    age, gender = extract_age_gender(lines[2])
    
    if age is None or gender is None:
        raise ValueError(f"Could not extract age and gender from {txt_path}")
    
    return {
        "words": words,
        "Age": age,
        "Gender": gender
    }

def convert_wav_to_mp3(wav_path, output_path):
    audio = AudioSegment.from_wav(wav_path)
    audio.export(output_path, format="mp3")

def process_directory(input_path, output_path, speech_condition):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process all .wav files
    for wav_file in input_path.rglob("*.wav"):
        # Get corresponding txt file
        txt_file = wav_file.with_suffix('.txt')
        if not txt_file.exists():
            print(f"Warning: No corresponding txt file found for {wav_file}")
            continue
        
        try:
            # Process text file
            text_data = process_text_file(txt_file)
            text_data["answer"] = speech_condition
            
            # Convert audio file
            relative_path = wav_file.relative_to(input_path)
            mp3_output_path = output_path / relative_path.with_suffix('.mp3')
            mp3_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            convert_wav_to_mp3(wav_file, mp3_output_path)
            
            # Save JSON data
            json_output_path = output_path / relative_path.with_suffix('.json')
            with open(json_output_path, 'w') as f:
                json.dump(text_data, f, indent=2)
                
            print(f"Processed {wav_file}")
            
        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process audio files and text files')
    parser.add_argument('input_path', help='Path to input directory containing .wav and .txt files')
    parser.add_argument('output_path', help='Path to output directory')
    parser.add_argument('speech_condition', choices=['typically developing', 'speech disorder'],
                       help='Speech condition of the samples')
    
    args = parser.parse_args()
    
    process_directory(args.input_path, args.output_path, args.speech_condition) 