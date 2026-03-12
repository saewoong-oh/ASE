# !TEX root = python
#!/usr/bin/env python3
"""
analyze_and_export.py
─────────────────────
Separate stems → C++ DFT analysis → export iOS-ready JSON and diagnostic audio.
"""

import argparse, json, os, re, sys
import numpy as np

STEM_BANDS = {
    "drums":  [(60.0,   250.0,  1.0), (5000.0, 16000.0, 0.5)],
    "bass":   [(40.0,   300.0,  1.0)],
    "vocals": [(250.0,  3500.0, 1.0)],
    "other":  [(500.0,  8000.0, 1.0)],
    "guitar": [(150.0,  5000.0, 1.0)],
    "piano":  [(120.0,  4000.0, 1.0)],
    "keys":   [(120.0,  4000.0, 1.0)],
}
DEFAULT_BAND = [(100.0, 8000.0, 1.0)]

def sanitize_filename(name: str) -> str:
    safe = re.sub(r'[^\w\s-]', '', name.lower())
    safe = re.sub(r'[\s-]+', '_', safe).strip('_')
    return safe or "map"

def load_audio_mono(path: str, target_sr: int = 44100) -> np.ndarray:
    try:
        import soundfile as sf
        audio, sr = sf.read(path, always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            try:
                import resampy
                audio = resampy.resample(audio, sr, target_sr)
                print(f"      Resampled {sr} → {target_sr} Hz")
            except ImportError:
                print(f"      WARNING: resampy not installed; audio stays at {sr} Hz")
        return np.ascontiguousarray(audio, dtype=np.float64)
    except Exception:
        import librosa
        audio, _ = librosa.load(path, sr=target_sr, mono=True)
        return np.ascontiguousarray(audio, dtype=np.float64)

def compute_raw_chroma(mix_audio: np.ndarray, sr: int, fft_size: int = 4096, hop_size: int = 1024) -> list:
    import ase_core
    stft       = ase_core.STFT(fft_size, hop_size, sr, ase_core.Window.HANN)
    chroma_ext = ase_core.ChromaExtractor(fft_size, hop_size, sr)
    
    n_frames = max(0, (len(mix_audio) - fft_size) // hop_size + 1)
    chroma_seq = []

    for idx in range(n_frames):
        start = idx * hop_size
        chunk = np.ascontiguousarray(mix_audio[start:start + fft_size], dtype=np.float64)
        frame = stft.analyze_frame(chunk, start / sr)
        mag   = np.array(frame.magnitude)
        chroma_seq.append(chroma_ext.analyze_frame(mag))

        if n_frames >= 1000 and idx % 1000 == 0:
            print(f"      raw-chroma frame {idx}/{n_frames} …", end="\r", flush=True)

    if n_frames >= 1000:
        print(f"      raw-chroma {n_frames} frames done.          ")

    return chroma_seq

def compute_band_rms_from_mix(mix_audio: np.ndarray, sr: int, stem_names: list, fft_size: int = 4096, hop_size: int = 1024) -> dict:
    import ase_core
    stft     = ase_core.STFT(fft_size, hop_size, sr, ase_core.Window.HANN)
    n_frames = max(0, (len(mix_audio) - fft_size) // hop_size + 1)
    result = {name: np.zeros(n_frames, dtype=np.float64) for name in stem_names}

    for idx in range(n_frames):
        start = idx * hop_size
        chunk = np.ascontiguousarray(mix_audio[start:start + fft_size], dtype=np.float64)
        frame = stft.analyze_frame(chunk, start / sr)
        mag   = np.array(frame.magnitude)

        for name in stem_names:
            bands       = STEM_BANDS.get(name, DEFAULT_BAND)
            band_energy = 0.0
            for (low_hz, high_hz, weight) in bands:
                lo = max(1, int(np.ceil (low_hz  * fft_size / sr)))
                hi = min(len(mag) - 1, int(np.floor(high_hz * fft_size / sr)))
                if lo <= hi:
                    band_energy += float(np.sqrt(np.mean(mag[lo:hi + 1] ** 2))) * weight
            result[name][idx] = band_energy

    out = {}
    for name in stem_names:
        out[name] = result[name].tolist()
    return out

def synthesize_chroma(chroma_seq, sr, hop_size, out_path):
    print(f"\n[5/5] Synthesizing chroma debug audio...")
    
    # Standard frequencies for C4 to B4
    freqs = [261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88]
    
    audio = np.zeros(len(chroma_seq) * hop_size, dtype=np.float32)
    t_frame = np.arange(hop_size) / sr
    phases = np.zeros(12)
    
    for i, frame in enumerate(chroma_seq):
        start = i * hop_size
        end = start + hop_size
        frame_audio = np.zeros(hop_size, dtype=np.float32)
        
        for c in range(12):
            mag = frame[c]
            if mag > 0.05: # Slight noise gate to keep it clean
                f = freqs[c]
                omega = 2 * np.pi * f
                wave = mag * np.sin(omega * t_frame + phases[c])
                frame_audio += wave
                phases[c] = (phases[c] + omega * (hop_size / sr)) % (2 * np.pi)
                
        audio[start:end] = frame_audio
        
    # Normalize volume to 80% to prevent speaker blowout
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = (audio / max_amp) * 0.8
        
    import soundfile as sf
    sf.write(out_path, audio, sr)
    print(f"      ✓ Saved audio to: {out_path}")

def run(ref_path: str, display_name: str, out_dir: str, sr: int = 44100, fft_size: int = 4096, hop_size: int = 1024, peak_db: float = -60.0, keep_gz: bool = False):
    import ase_core
    from stem_separator     import separate, load_stem
    from reference_analyzer import analyze_reference

    os.makedirs(out_dir, exist_ok=True)
    file_stem = sanitize_filename(display_name)
    map_stem  = f"map_{file_stem}"
    json_path = os.path.join(out_dir, f"{map_stem}.json")
    synth_path = os.path.join(out_dir, f"listen_{map_stem}.wav")

    print(f"\n[1/5] Loading full mix...")
    mix_audio = load_audio_mono(ref_path, target_sr=sr)

    print(f"\n[2/5] Separating stems with Demucs...")
    stem_paths = separate(ref_path, out_dir)

    stems = {}
    for name, path in stem_paths.items():
        audio, stem_sr = load_stem(path, target_sr=sr)
        stems[name] = (audio, stem_sr)

    print(f"\n[3/5] Running reference analysis...")
    fmap = analyze_reference(stems, full_mix=mix_audio, full_mix_sr=sr, fft_size=fft_size, hop_size=hop_size, peak_threshold_db=peak_db)

    print(f"\n[4/5] Extracting tracking map from HARMONIC STEMS (No Drums)...")
    
    # 🌟 THE FIX: Combine everything EXCEPT drums to create a pure tonal map
    harmonic_mix = np.zeros_like(mix_audio)
    for name, (audio, _) in stems.items():
        if name.lower() not in ["drums", "percussion"]:
            length = min(len(harmonic_mix), len(audio))
            harmonic_mix[:length] += audio[:length]

    # Extract chroma from the clean, drumless audio
    raw_chroma = compute_raw_chroma(harmonic_mix, sr, fft_size=fft_size, hop_size=hop_size)
    
    stem_names     = fmap.stem_names()
    # But keep the full mix for the fader automation levels
    stems_band_rms = compute_band_rms_from_mix(mix_audio, sr, stem_names, fft_size=fft_size, hop_size=hop_size)

    chroma_frames = len(raw_chroma)
    for name in stem_names:
        seq = stems_band_rms[name]
        if len(seq) < chroma_frames:
            stems_band_rms[name] = seq + [0.0] * (chroma_frames - len(seq))
        elif len(seq) > chroma_frames:
            stems_band_rms[name] = seq[:chroma_frames]

    data = {
        "display_name":    display_name,
        "stem_names":      stem_names,
        "combined_chroma": [list(c) for c in raw_chroma], 
        "stems_rms":       stems_band_rms,
    }

    # 🌟 SYNTHESIZE AUDIO
    synthesize_chroma(raw_chroma, sr, hop_size, synth_path)

    with open(json_path, 'w') as f:
        json.dump(data, f)

    print(f"\n      ✓ JSON ready: {json_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out",  default="ref_data")
    ap.add_argument("--sr",      type=int,   default=44100)
    ap.add_argument("--fft",     type=int,   default=4096)
    ap.add_argument("--hop",     type=int,   default=1024)
    ap.add_argument("--peak-db", type=float, default=-60.0)
    ap.add_argument("--keep-gz", action="store_true")
    args = ap.parse_args()

    display_name = args.name or os.path.splitext(os.path.basename(args.ref))[0]
    run(args.ref, display_name, args.out, args.sr, args.fft, args.hop, args.peak_db, args.keep_gz)

if __name__ == "__main__":
    main()