#!/usr/bin/env python3
"""
Analyze a reference song: separate stems → C++ DFT analysis → frequency map.

Usage:
    python run_analyze.py song.wav
    python run_analyze.py song.wav --name "My Song Name"
    python run_analyze.py song.wav --out my_ref_data/ --name "Jazz Trio"
"""

import argparse, os, sys

def sanitize_filename(name: str) -> str:
    """Convert a display name to a safe filename."""
    import re
    # lowercase, replace spaces/special chars with underscores
    safe = re.sub(r'[^\w\s-]', '', name.lower())
    safe = re.sub(r'[\s-]+', '_', safe).strip('_')
    return safe or "map"

def main():
    ap = argparse.ArgumentParser(description="Analyze a reference track")
    ap.add_argument("ref", help="Path to reference audio file")
    ap.add_argument("--name", default=None,
                    help="Display name for this reference (e.g. 'Jazz Trio'). "
                         "Used to name the output files. Defaults to the input filename stem.")
    ap.add_argument("--out", default="ref_data",
                    help="Output directory (default: ref_data/)")
    ap.add_argument("--sr",      type=int,   default=44100)
    ap.add_argument("--fft",     type=int,   default=4096)
    ap.add_argument("--hop",     type=int,   default=1024)
    ap.add_argument("--peak-db", type=float, default=-60.0)
    args = ap.parse_args()

    if not os.path.isfile(args.ref):
        print(f"ERROR: file not found: {args.ref}"); sys.exit(1)

    # ── derive display name and safe file stem ──
    if args.name:
        display_name = args.name
    else:
        display_name = os.path.splitext(os.path.basename(args.ref))[0]

    file_stem = sanitize_filename(display_name)   # e.g. "jazz_trio"
    map_filename = f"map_{file_stem}"             # e.g. "map_jazz_trio"

    print(f"\n  Display name : {display_name}")
    print(f"  File stem    : {map_filename}")

    # ── dependency checks ──
    try:
        import ase_core
    except ImportError:
        print("ERROR: C++ extension not built.  Run:  pip install .")
        sys.exit(1)

    from python.stem_separator import separate, load_stem
    from python.reference_analyzer import analyze_reference

    os.makedirs(args.out, exist_ok=True)

    # 1. Stem separation
    print(f"\n[1/3] Separating stems with Demucs …")
    print(f"      source: {args.ref}")
    stem_paths = separate(args.ref, args.out)
    print(f"      stems:  {list(stem_paths.keys())}\n")

    # 2. Load
    print(f"[2/3] Loading stems …")
    stems = {}
    for name, path in stem_paths.items():
        audio, sr = load_stem(path, target_sr=args.sr)
        stems[name] = (audio, sr)
        dur = len(audio) / sr
        print(f"      {name:8s}  {dur:.1f}s  sr={sr}")

    # 3. Analyze (C++ DFT engine)
    print(f"\n[3/3] Running C++ DFT analysis …")
    fmap = analyze_reference(stems,
                             fft_size=args.fft,
                             hop_size=args.hop,
                             peak_threshold_db=args.peak_db)

    map_path = os.path.join(args.out, f"{map_filename}.gz")
    fmap.serialize(map_path)
    total_notes = sum(len(s.notes) for s in fmap.stems.values())
    dur = fmap.n_frames * fmap.frame_duration

    print(f"\n  ✓  Frequency map saved to: {map_path}")
    print(f"     Display name: {display_name}")
    print(f"     Frames: {fmap.n_frames}   Duration: {dur:.1f}s")
    print(f"     Total notes detected: {total_notes}")
    for name, s in fmap.stems.items():
        peak = max(s.rms) if s.rms else 0
        print(f"     [{name:8s}]  notes={len(s.notes):4d}  "
              f"peak RMS={peak:.4f}")
    print()
    print(f"  Next step — export for iOS:")
    print(f"    python export_ios_map.py "
          f"--input {map_path} "
          f"--name \"{display_name}\"")
    print()


if __name__ == "__main__":
    main()