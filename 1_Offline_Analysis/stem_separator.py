"""
Stem separation using Demucs + audio loading utilities.
"""

import os
import subprocess
import numpy as np
import soundfile as sf

STEM_NAMES = ["vocals", "drums", "bass", "other"]


def separate(audio_path: str, output_dir: str,
             model: str = "htdemucs") -> dict[str, str]:
    """Run Demucs. Returns {stem_name: wav_path}."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "python", "-m", "demucs",
        "-n", model,
        "-o", output_dir,
        audio_path,
    ]
    subprocess.check_call(cmd)

    track_name = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(output_dir, model, track_name)

    stems = {}
    for name in STEM_NAMES:
        p = os.path.join(stem_dir, f"{name}.wav")
        if os.path.isfile(p):
            stems[name] = p
    return stems


def load_stem(path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """Load any audio file as mono float64, resampled to target_sr."""
    data, sr = sf.read(path, dtype="float64", always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        mono = resample_poly(mono, target_sr // g, sr // g)
        sr = target_sr
    return mono, sr


def load_mix(path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """Load the full mix as mono float64."""
    return load_stem(path, target_sr=target_sr)

# import os
# import subprocess
# import numpy as np
# import librosa

# STEM_NAMES = ["vocals", "drums", "bass", "other"]

# def separate(input_path: str, output_dir: str, stem_name: str | None = None) -> dict[str, str]:
#     """
#     Run Demucs to separate stems.
#     Returns a dictionary mapping stem names to their extracted .wav file paths.
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Base command
#     cmd = ['python', '-m', 'demucs', '-n', 'htdemucs', '-o', output_dir]
    
#     # ONLY add the two-stems flag if a valid stem is explicitly provided
#     if stem_name and str(stem_name).lower() != "none":
#         cmd.extend([f'--two-stems={stem_name}'])
    
#     cmd.append(input_path)
    
#     # Run the separation
#     subprocess.check_call(cmd)
    
#     # Demucs creates a subfolder based on the model name and the input file name
#     base_name = os.path.splitext(os.path.basename(input_path))[0]
#     track_dir = os.path.join(output_dir, 'htdemucs', base_name)
    
#     # Build the output paths
#     stem_paths = {}
#     if stem_name and str(stem_name).lower() != "none":
#         stem_paths[stem_name] = os.path.join(track_dir, f"{stem_name}.wav")
#         stem_paths["no_" + stem_name] = os.path.join(track_dir, f"no_{stem_name}.wav")
#     else:
#         for name in STEM_NAMES:
#             stem_paths[name] = os.path.join(track_dir, f"{name}.wav")
            
#     return stem_paths

# def load_stem(path: str, sr: int = 44100) -> tuple[np.ndarray, int]:
#     """
#     Load a stem as a mono numpy array.
#     """
#     audio, loaded_sr = librosa.load(path, sr=sr, mono=True)
#     return audio, loaded_sr

# def load_mix(path: str, sr: int = 44100) -> tuple[np.ndarray, int]:
#     """
#     Load the original mix as a mono numpy array.
#     """
#     audio, loaded_sr = librosa.load(path, sr=sr, mono=True)
#     return audio, loaded_sr


# import os
# import subprocess
# import numpy as np
# import soundfile as sf


# STEM_NAMES = ["vocals", "drums", "bass", "other"]


# def separate(audio_path: str, output_dir: str,
#              model: str = "htdemucs") -> dict[str, str]:
#     """
#     Run Demucs on *audio_path*, write stems to *output_dir*.
#     Returns {stem_name: wav_path}.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     cmd = [
#         "python", "-m", "demucs",
#         "--two-stems=None",
#         "-n", model,
#         "-o", output_dir,
#         audio_path,
#     ]
#     subprocess.check_call(cmd)

#     # Demucs writes  <output_dir>/<model>/<track_name>/<stem>.wav
#     track_name = os.path.splitext(os.path.basename(audio_path))[0]
#     stem_dir = os.path.join(output_dir, model, track_name)

#     stems = {}
#     for name in STEM_NAMES:
#         p = os.path.join(stem_dir, f"{name}.wav")
#         if os.path.isfile(p):
#             stems[name] = p
#     return stems


# def load_stem(path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
#     """Load a stem wav as mono float64, resampled to target_sr."""
#     data, sr = sf.read(path, dtype="float64", always_2d=True)
#     # mix to mono
#     mono = data.mean(axis=1)
#     if sr != target_sr:
#         from scipy.signal import resample_poly
#         from math import gcd
#         g = gcd(sr, target_sr)
#         mono = resample_poly(mono, target_sr // g, sr // g)
#         sr = target_sr
#     return mono, sr


# def load_mix(path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
#     """Load the full mix as mono float64."""
#     return load_stem(path, target_sr)
