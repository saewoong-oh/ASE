"""
Stem separation using Demucs and audio loading utilities.

Provides three functions:
  - separate(): run Demucs CLI to split a song into stems (vocals, drums, bass, other)
  - load_stem(): load a single stem WAV as mono float64, resampled to a target rate
  - load_mix():  load a full mix WAV as mono float64 (convenience wrapper)
"""

import os
import subprocess
import numpy as np
import soundfile as sf

# The four stem names produced by the htdemucs model
STEM_NAMES = ["vocals", "drums", "bass", "other"]


def separate(audio_path: str, output_dir: str,
             model: str = "htdemucs") -> dict[str, str]:
    """
    Run Demucs source separation on an audio file.
    
    Invokes the Demucs CLI as a subprocess with the specified model.
    Returns a dict mapping each stem name to the path of its output WAV file.
    Only stems that actually exist on disk after separation are included.
    
    Args:
        audio_path: Path to the input audio file
        output_dir: Directory for Demucs output (stems saved under output_dir/model/trackname/)
        model:      Demucs model name (default: htdemucs, the hybrid transformer model)
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "python", "-m", "demucs",
        "-n", model,
        "-o", output_dir,
        audio_path,
    ]
    subprocess.check_call(cmd)

    # Demucs saves stems under: output_dir/model_name/track_name/stem.wav
    track_name = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(output_dir, model, track_name)

    stems = {}
    for name in STEM_NAMES:
        p = os.path.join(stem_dir, f"{name}.wav")
        if os.path.isfile(p):
            stems[name] = p
    return stems


def load_stem(path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """
    Load any audio file as mono float64, resampled to target_sr.
    
    Reads the file via soundfile, downmixes to mono by averaging channels,
    and resamples using polyphase filtering if the source rate differs.
    
    Returns:
        (mono_audio, sample_rate) tuple
    """
    data, sr = sf.read(path, dtype="float64", always_2d=True)
    # Downmix to mono by averaging all channels
    mono = data.mean(axis=1)
    if sr != target_sr:
        # Polyphase resampling: exact rational rate conversion
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        mono = resample_poly(mono, target_sr // g, sr // g)
        sr = target_sr
    return mono, sr


def load_mix(path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """Load the full mix as mono float64, resampled to target_sr."""
    return load_stem(path, target_sr=target_sr)