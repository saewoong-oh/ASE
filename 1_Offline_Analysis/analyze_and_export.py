#!/usr/bin/env python3
"""
analyze_and_export.py
─────────────────────
Separate stems → C++ DFT analysis → export iOS-ready JSON in one step.

Usage:
    python analyze_and_export.py song.wav
    python analyze_and_export.py song.wav --name "Jazz Trio"
    python analyze_and_export.py song.wav --name "Jazz Trio" --out ref_data/
    python analyze_and_export.py song.wav --keep-gz        # also save .gz

Output:
    ref_data/map_jazz_trio.json   ← drag this into Xcode
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def sanitize_filename(name: str) -> str:
    safe = re.sub(r'[^\w\s-]', '', name.lower())
    safe = re.sub(r'[\s-]+', '_', safe).strip('_')
    return safe or "map"


def compute_band_rms_sequence(audio, sr, fft_size=4096, hop_size=1024, bands=None):
    """
    Compute per-hop band-limited RMS — mirrors ASEWrapper.mm estimateStemRMS exactly.
    """
    import ase_core
    stft     = ase_core.STFT(fft_size, hop_size, sr, ase_core.Window.HANN)
    n_frames = max(0, (len(audio) - fft_size) // hop_size + 1)
    seq      = []

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

        seq.append(float(band_energy))

    return seq


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(ref_path: str,
        display_name: str,
        out_dir: str,
        sr: int      = 44100,
        fft_size: int = 4096,
        hop_size: int = 1024,
        peak_db: float = -60.0,
        keep_gz: bool  = False):

    try:
        import ase_core
    except ImportError:
        print("ERROR: C++ extension not built.  Run:  pip install .")
        sys.exit(1)

    from stem_separator    import separate, load_stem
    from reference_analyzer import analyze_reference
    from frequency_map      import FrequencyMap

    os.makedirs(out_dir, exist_ok=True)

    file_stem    = sanitize_filename(display_name)   # e.g. "jazz_trio"
    map_stem     = f"map_{file_stem}"                # e.g. "map_jazz_trio"
    gz_path      = os.path.join(out_dir, f"{map_stem}.gz")
    json_path    = os.path.join(out_dir, f"{map_stem}.json")

    print(f"\n  Display name : {display_name}")
    print(f"  File stem    : {map_stem}")
    print(f"  Output dir   : {out_dir}")

    # ── 1. Stem separation ───────────────────────────────────────────────────
    print(f"\n[1/3] Separating stems with Demucs …")
    print(f"      source : {ref_path}")
    stem_paths = separate(ref_path, out_dir)
    print(f"      stems  : {list(stem_paths.keys())}")

    # ── 2. Load stems ────────────────────────────────────────────────────────
    print(f"\n[2/3] Loading stems …")
    stems      = {}
    stem_audio = {}   # keep raw audio for band-RMS computation later
    for name, path in stem_paths.items():
        audio, stem_sr = load_stem(path, target_sr=sr)
        stems[name]      = (audio, stem_sr)
        stem_audio[name] = (audio, stem_sr)
        dur = len(audio) / stem_sr
        print(f"      {name:8s}  {dur:.1f}s  sr={stem_sr}")

    # ── 3. C++ DFT analysis ──────────────────────────────────────────────────
    print(f"\n[3/3] Running C++ DFT analysis …")
    fmap = analyze_reference(stems,
                             fft_size=fft_size,
                             hop_size=hop_size,
                             peak_threshold_db=peak_db)

    total_notes = sum(len(s.notes) for s in fmap.stems.values())
    dur_s       = fmap.n_frames * fmap.frame_duration
    print(f"\n  Frames : {fmap.n_frames}   Duration : {dur_s:.1f}s")
    print(f"  Notes  : {total_notes}")
    for name, s in fmap.stems.items():
        peak = max(s.rms) if s.rms else 0.0
        print(f"  [{name:8s}]  notes={len(s.notes):4d}  peak RMS={peak:.4f}")

    # Optionally persist the .gz (useful for re-exporting without re-analyzing)
    if keep_gz:
        fmap.serialize(gz_path)
        print(f"\n  .gz saved : {gz_path}")

    # ── 4. Export iOS JSON ───────────────────────────────────────────────────
    print(f"\nExporting iOS JSON …")

    BAND_FRACTION = {
        "drums": 0.35, "bass": 0.40, "vocals": 0.45,
        "other": 0.30, "guitar": 0.30, "piano": 0.30, "keys": 0.30,
    }

    stems_band_rms = {}
    for name, profile in fmap.stems.items():
        bands = STEM_BANDS.get(name, DEFAULT_BAND)

        if name in stem_audio:
            audio, stem_sr = stem_audio[name]
            band_rms = compute_band_rms_sequence(
                audio, stem_sr,
                fft_size=fft_size,
                hop_size=hop_size,
                bands=bands)
        else:
            # Fallback: scale broadband RMS (should never happen in normal use)
            scale    = BAND_FRACTION.get(name, 0.35)
            band_rms = [r * scale for r in profile.rms]
            print(f"  WARNING: {name} — using broadband RMS × {scale:.2f} (no raw audio)")

        stems_band_rms[name] = band_rms
        peak = max(band_rms) if band_rms else 0.0
        print(f"  {name:8s}: {len(band_rms)} frames, peak band RMS = {peak:.5f}")

    data = {
        "display_name":    display_name,
        "stem_names":      fmap.stem_names(),
        "combined_chroma": fmap.combined_chroma.tolist(),
        "stems_rms":       stems_band_rms,
    }

    with open(json_path, 'w') as f:
        json.dump(data, f)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  ✓  JSON ready : {json_path}")
    print(f"{'─'*60}")
    print(f"\n  1. Drag  {os.path.basename(json_path)}  into your Xcode project")
    print(f"     (tick 'Copy items if needed', add to your app target)")
    print(f"\n  2. Add one line to ReferenceLibrary.swift:")
    print(f'     ReferenceEntry(displayName: "{display_name}",')
    print(f'                    mapFileName: "{map_stem}")')
    print(f"\n  3. Build & run — the new reference will appear in the picker.")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Analyze a reference track and export an iOS-ready JSON map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    ap.add_argument("ref",
                    help="Path to reference audio file (wav / mp3 / flac …)")
    ap.add_argument("--name", default=None,
                    help="Human-readable display name shown in the app "
                         "(e.g. 'Jazz Trio').  Defaults to the filename stem.")
    ap.add_argument("--out",  default="ref_data",
                    help="Output directory  (default: ref_data/)")
    ap.add_argument("--sr",      type=int,   default=44100,
                    help="Sample rate  (default: 44100)")
    ap.add_argument("--fft",     type=int,   default=4096,
                    help="FFT size  (default: 4096)")
    ap.add_argument("--hop",     type=int,   default=1024,
                    help="Hop size  (default: 1024)")
    ap.add_argument("--peak-db", type=float, default=-60.0,
                    help="Peak detection threshold in dB  (default: -60)")
    ap.add_argument("--keep-gz", action="store_true",
                    help="Also save the intermediate .gz file "
                         "(lets you re-export JSON without re-running Demucs)")

    args = ap.parse_args()

    if not os.path.isfile(args.ref):
        print(f"ERROR: file not found: {args.ref}")
        sys.exit(1)

    display_name = args.name or os.path.splitext(os.path.basename(args.ref))[0]

    run(ref_path     = args.ref,
        display_name = display_name,
        out_dir      = args.out,
        sr           = args.sr,
        fft_size     = args.fft,
        hop_size     = args.hop,
        peak_db      = args.peak_db,
        keep_gz      = args.keep_gz)


if __name__ == "__main__":
    main()