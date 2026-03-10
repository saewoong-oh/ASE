#!/usr/bin/env python3
"""
Export a .gz frequency map to a .json file ready for Xcode.

Usage:
    python export_ios_map.py --input ref_data/map_jazz_trio.gz --name "Jazz Trio"
    python export_ios_map.py --input ref_data/map_jazz_trio.gz
        (name defaults to the file stem, e.g. "map_jazz_trio")

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


def compute_band_rms_sequence(audio, sr, fft_size=4096, hop_size=1024, bands=None):
    import ase_core
    stft = ase_core.STFT(fft_size, hop_size, sr, ase_core.Window.HANN)
    n_frames = max(0, (len(audio) - fft_size) // hop_size + 1)
    rms_sequence = []

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

        rms_sequence.append(float(band_energy))

    return rms_sequence


def export_for_ios(gz_path: str, display_name: str):
    # Derive output path: same directory, same stem, .json extension
    base_dir  = os.path.dirname(gz_path)
    stem      = os.path.splitext(os.path.basename(gz_path))[0]   # e.g. map_jazz_trio
    out_path  = os.path.join(base_dir, f"{stem}.json")

    # Import here so the script can print usage without requiring ase_core
    sys.path.insert(0, os.path.dirname(__file__))
    from frequency_map import FrequencyMap

    print(f"Loading {gz_path} …")
    fmap = FrequencyMap.load(gz_path)

    stem_names = fmap.stem_names()
    print(f"Stems        : {stem_names}")
    print(f"Display name : {display_name}")
    print(f"Output       : {out_path}")
    print("Computing band-limited RMS sequences …")

    stems_band_rms = {}
    BAND_FRACTION = {
        "drums": 0.35, "bass": 0.40, "vocals": 0.45,
        "other": 0.30, "guitar": 0.30, "piano": 0.30, "keys": 0.30,
    }

    for name, profile in fmap.stems.items():
        bands = STEM_BANDS.get(name, DEFAULT_BAND)

        if hasattr(profile, '_audio') and profile._audio is not None:
            audio, sr = profile._audio, fmap.sr
            band_rms = compute_band_rms_sequence(audio, sr,
                                                  fft_size=fmap.fft_size,
                                                  hop_size=fmap.hop_size,
                                                  bands=bands)
        else:
            scale    = BAND_FRACTION.get(name, 0.35)
            band_rms = [r * scale for r in profile.rms]
            print(f"  WARNING: {name} using broadband RMS × {scale:.2f} (no raw audio)")

        stems_band_rms[name] = band_rms
        peak = max(band_rms) if band_rms else 0.0
        print(f"  {name:8s}: {len(band_rms)} frames, peak band RMS = {peak:.5f}")

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
    ap.add_argument("--name",   default=None,
                    help="Human-readable display name (e.g. 'Jazz Trio'). "
                         "Defaults to the file stem.")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: file not found: {args.input}"); sys.exit(1)

    stem = os.path.splitext(os.path.basename(args.input))[0]
    display_name = args.name if args.name else stem

    export_for_ios(args.input, display_name)


if __name__ == "__main__":
    main()