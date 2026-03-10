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

IMPORTANT: The reference RMS values are computed from the FULL MIX, not
from isolated stems. This matches what the iOS live estimator will see
(a mixed signal measured in each stem's characteristic frequency band).
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


def load_audio_mono(path: str, target_sr: int = 44100) -> np.ndarray:
    """Load any audio file as mono float64 at target_sr."""
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
        try:
            import librosa
            audio, _ = librosa.load(path, sr=target_sr, mono=True)
            return np.ascontiguousarray(audio, dtype=np.float64)
        except Exception as e:
            print(f"ERROR loading {path}: {e}")
            sys.exit(1)


def compute_band_rms_from_mix(mix_audio: np.ndarray,
                               sr: int,
                               stem_names: list,
                               fft_size: int = 4096,
                               hop_size: int = 1024) -> dict:
    """
    Compute per-stem band-limited RMS sequences from the FULL MIX audio.

    This is the correct approach because:
      - The iOS tap sees the full mix (playback or mic).
      - Each stem's band is measured from that same mixed signal.
      - So the reference must also be measured from the mix in those bands.

    The band definitions tell us WHICH frequency region is characteristic
    of each stem, but we always measure from the same mixed signal — both
    here (offline) and on iOS (live).

    Formula: sqrt(mean(magnitude[low_bin:high_bin]^2))
    This matches ASEWrapper.mm computeBandEnergy exactly.
    """
    import ase_core

    stft     = ase_core.STFT(fft_size, hop_size, sr, ase_core.Window.HANN)
    n_frames = max(0, (len(mix_audio) - fft_size) // hop_size + 1)

    # Pre-allocate output arrays
    result = {name: np.zeros(n_frames, dtype=np.float64) for name in stem_names}

    for idx in range(n_frames):
        start = idx * hop_size
        chunk = np.ascontiguousarray(mix_audio[start:start + fft_size],
                                     dtype=np.float64)
        frame = stft.analyze_frame(chunk, start / sr)
        mag   = np.array(frame.magnitude)   # shape: (fft_size//2 + 1,)

        for name in stem_names:
            bands       = STEM_BANDS.get(name, DEFAULT_BAND)
            band_energy = 0.0
            for (low_hz, high_hz, weight) in bands:
                lo = max(1, int(np.ceil (low_hz  * fft_size / sr)))
                hi = min(len(mag) - 1,
                         int(np.floor(high_hz * fft_size / sr)))
                if lo > hi:
                    continue
                # sqrt(mean(mag^2)) — identical to iOS computeBandEnergy
                e = float(np.sqrt(np.mean(mag[lo:hi + 1] ** 2)))
                band_energy += e * weight
            result[name][idx] = band_energy

        if n_frames >= 1000 and idx % 1000 == 0:
            print(f"      band-RMS frame {idx}/{n_frames} …", end="\r", flush=True)

    if n_frames >= 1000:
        print(f"      band-RMS {n_frames} frames done.          ")

    # Convert to lists and print stats
    out = {}
    for name in stem_names:
        seq  = result[name].tolist()
        peak = float(result[name].max()) if n_frames > 0 else 0.0
        mean = float(result[name].mean()) if n_frames > 0 else 0.0
        print(f"      {name:8s}: {n_frames} frames  "
              f"mean={mean:.5f}  peak={peak:.5f}")
        out[name] = seq
    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(ref_path: str,
        display_name: str,
        out_dir: str,
        sr: int       = 44100,
        fft_size: int = 4096,
        hop_size: int = 1024,
        peak_db: float = -60.0,
        keep_gz: bool  = False):

    try:
        import ase_core
    except ImportError:
        print("ERROR: C++ extension not built.  Run:  pip install .")
        sys.exit(1)

    from stem_separator     import separate, load_stem
    from reference_analyzer import analyze_reference

    os.makedirs(out_dir, exist_ok=True)

    file_stem = sanitize_filename(display_name)
    map_stem  = f"map_{file_stem}"
    gz_path   = os.path.join(out_dir, f"{map_stem}.gz")
    json_path = os.path.join(out_dir, f"{map_stem}.json")

    print(f"\n  Display name : {display_name}")
    print(f"  File stem    : {map_stem}")
    print(f"  Output dir   : {out_dir}")

    # ── 1. Load full mix (mono) for band-RMS reference ───────────────────────
    # We load this NOW before Demucs runs, from the original source file.
    # This guarantees we have the exact same signal the iOS player will play.
    print(f"\n[0/3] Loading full mix for reference RMS …")
    print(f"      source : {ref_path}")
    mix_audio = load_audio_mono(ref_path, target_sr=sr)
    mix_dur   = len(mix_audio) / sr
    mix_frames = max(0, (len(mix_audio) - fft_size) // hop_size + 1)
    print(f"      {len(mix_audio)} samples @ {sr} Hz = {mix_dur:.1f}s  "
          f"→ {mix_frames} STFT frames")

    # ── 2. Stem separation ───────────────────────────────────────────────────
    print(f"\n[1/3] Separating stems with Demucs …")
    stem_paths = separate(ref_path, out_dir)
    print(f"      stems  : {list(stem_paths.keys())}")

    # ── 3. Load stems ────────────────────────────────────────────────────────
    print(f"\n[2/3] Loading stems …")
    stems = {}
    for name, path in stem_paths.items():
        audio, stem_sr = load_stem(path, target_sr=sr)
        stems[name] = (audio, stem_sr)
        print(f"      {name:8s}  {len(audio)/stem_sr:.1f}s  sr={stem_sr}")

    # ── 4. C++ DFT analysis (chroma + notes, uses isolated stems) ───────────
    print(f"\n[3/3] Running C++ DFT analysis …")
    fmap = analyze_reference(stems,
                             full_mix=mix_audio,
                             full_mix_sr=sr,
                             fft_size=fft_size,
                             hop_size=hop_size,
                             peak_threshold_db=peak_db)

    total_notes = sum(len(s.notes) for s in fmap.stems.values())
    dur_s       = fmap.n_frames * fmap.frame_duration
    print(f"\n  Frames : {fmap.n_frames}   Duration : {dur_s:.1f}s")
    print(f"  Notes  : {total_notes}")
    for name, s in fmap.stems.items():
        peak = max(s.rms) if s.rms else 0.0
        print(f"  [{name:8s}]  notes={len(s.notes):4d}  peak broadband RMS={peak:.4f}")

    if keep_gz:
        fmap.serialize(gz_path)
        print(f"\n  .gz saved : {gz_path}")

    # ── 5. Compute band-RMS from FULL MIX ────────────────────────────────────
    # This is the critical step: measure each stem's characteristic band
    # from the mixed signal, NOT from isolated stems.
    print(f"\nComputing band-RMS from full mix …")
    stem_names     = fmap.stem_names()
    stems_band_rms = compute_band_rms_from_mix(
        mix_audio, sr, stem_names,
        fft_size=fft_size,
        hop_size=hop_size)

    # Sanity check: warn if frame counts don't match chroma
    chroma_frames = len(fmap.combined_chroma)
    if mix_frames != chroma_frames:
        print(f"\n  NOTE: mix frames ({mix_frames}) != chroma frames ({chroma_frames}). "
              f"Trimming/padding to match chroma.")

    # Align all RMS sequences to chroma frame count
    for name in stem_names:
        seq = stems_band_rms[name]
        if len(seq) < chroma_frames:
            stems_band_rms[name] = seq + [0.0] * (chroma_frames - len(seq))
        elif len(seq) > chroma_frames:
            stems_band_rms[name] = seq[:chroma_frames]

    # ── 6. Write JSON ─────────────────────────────────────────────────────────
    data = {
        "display_name":    display_name,
        "stem_names":      stem_names,
        "combined_chroma": fmap.combined_chroma.tolist(),
        "stems_rms":       stems_band_rms,
    }

    with open(json_path, 'w') as f:
        json.dump(data, f)

    size_mb = os.path.getsize(json_path) / 1e6

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  ✓  JSON ready : {json_path}  ({size_mb:.1f} MB)")
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
                    help="Also save the intermediate .gz file")

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