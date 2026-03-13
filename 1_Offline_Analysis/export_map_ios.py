# Deprecated method







#!/usr/bin/env python3
"""
Export a .gz frequency map to a .json file ready for Xcode.

Usage:
    python export_ios_map.py --input ref_data/map_jazz_trio.gz --ref song.wav --name "Jazz Trio"
    python export_ios_map.py --input ref_data/map_jazz_trio.gz --ref song.wav

The output JSON file will be written next to the .gz file with the same stem:
    ref_data/map_jazz_trio.json

Drag the resulting .json into your Xcode project, then add an entry to
ReferenceLibrary.swift:
    ReferenceEntry(displayName: "Jazz Trio", mapFileName: "map_jazz_trio")
"""

import argparse, json, os, re, sys
import numpy as np

# ── Must match STEM_BANDS in ASEWrapper.mm exactly ──────────────────────────
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


def load_audio(path: str, target_sr: int = 44100) -> np.ndarray:
    """Load any audio file to mono float64 at target_sr."""
    try:
        import soundfile as sf
        audio, sr = sf.read(path, always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            try:
                import resampy
                audio = resampy.resample(audio, sr, target_sr)
            except ImportError:
                print(f"  WARNING: resampy not installed, cannot resample from {sr} to {target_sr}")
    except Exception:
        try:
            import librosa
            audio, sr = librosa.load(path, sr=target_sr, mono=True)
        except Exception as e:
            print(f"ERROR loading {path}: {e}")
            sys.exit(1)
    return np.ascontiguousarray(audio, dtype=np.float64)


def compute_band_rms_from_mix(mix_audio: np.ndarray,
                               sr: int,
                               stem_names: list,
                               fft_size: int = 4096,
                               hop_size: int = 1024) -> dict:
    """
    Compute per-stem band-limited RMS sequences FROM THE FULL MIX.

    This is the key insight: since the live estimator sees the full mix,
    the reference must also be computed from the full mix using the same
    spectral bands. The bands just define which frequency region is
    'characteristic' of each stem, but we measure from the same mixed signal.

    This ensures reference and live values are on the same scale.
    """
    import ase_core
    stft = ase_core.STFT(fft_size, hop_size, sr, ase_core.Window.HANN)
    n_frames = max(0, (len(mix_audio) - fft_size) // hop_size + 1)

    # Initialize per-stem sequences
    stem_rms = {name: [] for name in stem_names}

    print(f"  Computing band RMS from mix: {n_frames} frames …")

    for idx in range(n_frames):
        start = idx * hop_size
        chunk = np.ascontiguousarray(mix_audio[start:start + fft_size], dtype=np.float64)
        frame = stft.analyze_frame(chunk, start / sr)
        mag   = np.array(frame.magnitude)

        for name in stem_names:
            bands = STEM_BANDS.get(name, DEFAULT_BAND)
            band_energy = 0.0

            for (low_hz, high_hz, weight) in bands:
                low_bin  = max(1, int(np.ceil(low_hz  * fft_size / sr)))
                high_bin = min(len(mag) - 1, int(np.floor(high_hz * fft_size / sr)))
                if low_bin > high_bin:
                    continue
                energy = np.sqrt(np.mean(mag[low_bin:high_bin + 1] ** 2))
                band_energy += energy * weight

            stem_rms[name].append(float(band_energy))

    return stem_rms


def compute_band_rms_from_stems(stem_audios: dict,
                                 sr: int,
                                 fft_size: int = 4096,
                                 hop_size: int = 1024) -> dict:
    """
    Compute per-stem band-limited RMS sequences from isolated stem audio.
    Only used when isolated stem audio is available.
    Each stem is measured in its own characteristic band FROM ITS OWN AUDIO.
    """
    import ase_core
    stft = ase_core.STFT(fft_size, hop_size, sr, ase_core.Window.HANN)
    stem_rms = {}

    for name, audio in stem_audios.items():
        bands  = STEM_BANDS.get(name, DEFAULT_BAND)
        audio  = np.ascontiguousarray(audio, dtype=np.float64)
        n_frames = max(0, (len(audio) - fft_size) // hop_size + 1)
        rms_seq  = []

        for idx in range(n_frames):
            start = idx * hop_size
            chunk = np.ascontiguousarray(audio[start:start + fft_size], dtype=np.float64)
            frame = stft.analyze_frame(chunk, start / sr)
            mag   = np.array(frame.magnitude)
            band_energy = 0.0

            for (low_hz, high_hz, weight) in bands:
                low_bin  = max(1, int(np.ceil(low_hz  * fft_size / sr)))
                high_bin = min(len(mag) - 1, int(np.floor(high_hz * fft_size / sr)))
                if low_bin > high_bin:
                    continue
                energy = np.sqrt(np.mean(mag[low_bin:high_bin + 1] ** 2))
                band_energy += energy * weight

            rms_seq.append(float(band_energy))

        stem_rms[name] = rms_seq
        peak = max(rms_seq) if rms_seq else 0.0
        print(f"  {name:8s}: {len(rms_seq)} frames, peak band RMS = {peak:.5f}")

    return stem_rms


def export_for_ios(gz_path: str, display_name: str, ref_audio_path: str | None = None):
    base_dir = os.path.dirname(gz_path)
    stem     = os.path.splitext(os.path.basename(gz_path))[0]
    out_path = os.path.join(base_dir, f"{stem}.json")

    sys.path.insert(0, os.path.dirname(__file__))
    from frequency_map import FrequencyMap

    print(f"Loading {gz_path} …")
    fmap = FrequencyMap.load(gz_path)

    stem_names = fmap.stem_names()
    print(f"Stems        : {stem_names}")
    print(f"Display name : {display_name}")
    print(f"Output       : {out_path}")

    # ── Determine how to compute reference RMS ────────────────────────────
    #
    # Priority:
    #   1. Full mix audio provided via --ref  → compute from mix (BEST, matches live)
    #   2. Isolated stem audio available      → compute from stems
    #   3. Fallback: broadband RMS × scale    (least accurate)
    #
    # Option 1 is strongly preferred because it matches what the live
    # estimator will see: a mixed signal measured in each stem's band.

    stems_band_rms = {}

    if ref_audio_path is not None:
        # ── Option 1: compute from full mix (RECOMMENDED) ─────────────────
        print(f"\nComputing reference RMS from full mix: {ref_audio_path}")
        mix_audio = load_audio(ref_audio_path, target_sr=fmap.sr)
        print(f"  Mix duration: {len(mix_audio)/fmap.sr:.1f}s at {fmap.sr} Hz")

        stems_band_rms = compute_band_rms_from_mix(
            mix_audio, fmap.sr, stem_names,
            fft_size=fmap.fft_size, hop_size=fmap.hop_size)

        for name in stem_names:
            seq  = stems_band_rms[name]
            peak = max(seq) if seq else 0.0
            print(f"  {name:8s}: {len(seq)} frames, peak band RMS = {peak:.5f}")

    else:
        # ── Option 2 / 3: try stored stem audio, else fallback ────────────
        print("\nNo --ref mix provided. Trying stored stem audio …")
        BAND_FRACTION = {
            "drums": 0.35, "bass": 0.40, "vocals": 0.45,
            "other": 0.30, "guitar": 0.30, "piano": 0.30, "keys": 0.30,
        }

        for name, profile in fmap.stems.items():
            bands = STEM_BANDS.get(name, DEFAULT_BAND)

            if hasattr(profile, '_audio') and profile._audio is not None:
                # Option 2
                audio = np.ascontiguousarray(profile._audio, dtype=np.float64)
                rms_seq = compute_band_rms_from_stems(
                    {name: audio}, fmap.sr,
                    fft_size=fmap.fft_size, hop_size=fmap.hop_size)[name]
                print(f"  {name}: computed from stored stem audio")
            else:
                # Option 3 — least accurate
                scale   = BAND_FRACTION.get(name, 0.35)
                rms_seq = [r * scale for r in profile.rms]
                print(f"  WARNING: {name} using broadband RMS × {scale:.2f} "
                      f"(no raw audio — pass --ref for best results)")

            stems_band_rms[name] = rms_seq
            peak = max(rms_seq) if rms_seq else 0.0
            print(f"  {name:8s}: {len(rms_seq)} frames, peak band RMS = {peak:.5f}")

    # ── Align frame counts ────────────────────────────────────────────────
    # The chroma and RMS sequences must have the same number of frames.
    n_chroma = len(fmap.combined_chroma)
    for name in stem_names:
        seq = stems_band_rms[name]
        if len(seq) < n_chroma:
            # Pad with zeros
            stems_band_rms[name] = seq + [0.0] * (n_chroma - len(seq))
        elif len(seq) > n_chroma:
            # Trim
            stems_band_rms[name] = seq[:n_chroma]

    print(f"\nAligned to {n_chroma} frames ({n_chroma * fmap.hop_size / fmap.sr:.1f}s)")

    data = {
        "display_name":    display_name,
        "stem_names":      stem_names,
        "combined_chroma": fmap.combined_chroma.tolist(),
        "stems_rms":       stems_band_rms,
    }

    with open(out_path, 'w') as f:
        json.dump(data, f)

    print(f"\n✓  Saved: {out_path}")
    print(f"\nAdd to ReferenceLibrary.swift:")
    print(f'    ReferenceEntry(displayName: "{display_name}", mapFileName: "{stem}")')
    print(f"\nDrag  {out_path}  into your Xcode project (Copy items if needed).")


def main():
    ap = argparse.ArgumentParser(
        description="Export a .gz frequency map to iOS-ready JSON")
    ap.add_argument("--input",  required=True,
                    help="Path to the .gz map (e.g. ref_data/map_jazz_trio.gz)")
    ap.add_argument("--ref",    default=None,
                    help="Path to the original full-mix audio file (WAV/MP3/FLAC). "
                         "STRONGLY RECOMMENDED: ensures reference RMS matches "
                         "what the live estimator will see. If omitted, falls back "
                         "to stored broadband RMS values (less accurate).")
    ap.add_argument("--name",   default=None,
                    help="Human-readable display name (e.g. 'Jazz Trio'). "
                         "Defaults to the file stem.")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: file not found: {args.input}"); sys.exit(1)

    if args.ref and not os.path.isfile(args.ref):
        print(f"ERROR: ref audio not found: {args.ref}"); sys.exit(1)

    file_stem    = os.path.splitext(os.path.basename(args.input))[0]
    display_name = args.name if args.name else file_stem

    export_for_ios(args.input, display_name, ref_audio_path=args.ref)


if __name__ == "__main__":
    main()