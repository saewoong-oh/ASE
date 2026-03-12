import argparse
import subprocess
import os
from pydub import AudioSegment

def process_sample(args):
    print(f"1. Slicing '{args.input}' from {args.start}s to {args.end}s...")
    song = AudioSegment.from_file(args.input)
    
    # pydub works in milliseconds
    snippet = song[args.start * 1000 : args.end * 1000] 
    sliced_file = "temp_slice.wav"
    snippet.export(sliced_file, format="wav")

    print("2. Running 4-stem separation (htdemucs)...")
    subprocess.run([
        "demucs", 
        "-n", "htdemucs", 
        sliced_file
    ], check=True)

    print("3. Loading stems and adjusting volume...")
    # Updated directory path for the 4-stem model
    stem_dir = "separated/htdemucs/temp_slice"
    stems = ["vocals", "drums", "bass", "other"]
    
    mixed = None

    for stem_name in stems:
        stem_path = f"{stem_dir}/{stem_name}.wav"
        stem_audio = AudioSegment.from_file(stem_path)
        
        # Apply the gain reduction to the targeted stem
        if stem_name == args.stem:
            print(f"   -> Reducing {stem_name} by {args.reduce} dB")
            stem_audio = stem_audio + args.reduce
            
        # Mix the tracks
        if mixed is None:
            mixed = stem_audio
        else:
            mixed = mixed.overlay(stem_audio)

    print("4. Exporting final files...")
    base_name = os.path.splitext(args.input)[0]
    
    mixed.export(f"{base_name}_sample.wav", format="wav")
    mixed.export(f"{base_name}_sample.mp3", format="mp3", bitrate="320k")

    print(f"Done! Check your folder for {base_name}_sample.wav and .mp3")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolate a sample and attenuate a specific stem using the 4-stem model.")
    
    parser.add_argument("input", help="The input audio file (e.g., song.mp3)")
    parser.add_argument("--start", "-s", type=float, required=True, help="Start time in seconds")
    parser.add_argument("--end", "-e", type=float, required=True, help="End time in seconds")
    
    # Updated choices and default
    parser.add_argument("--stem", "-t", type=str, default="other", 
                        choices=["vocals", "drums", "bass", "other"], 
                        help="Which stem to lower (default: other)")
    parser.add_argument("--reduce", "-r", type=float, default=-15.0, 
                        help="How much to reduce the volume in dB (default: -15.0)")
    
    args = parser.parse_args()
    process_sample(args)