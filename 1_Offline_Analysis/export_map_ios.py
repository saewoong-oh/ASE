import json
import numpy as np
from frequency_map import FrequencyMap

# Must match STEM_BANDS in ASEWrapper.mm exactly
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
    """
    Compute per-hop band-limited RMS sequence for audio signal.
    Matches the spectral band energy computation in ASEWrapper.mm.
    """
    import ase_core

    stft = ase_core.STFT(fft_size, hop_size, sr, ase_core.Window.HANN)
    n_frames = max(0, (len(audio) - fft_size) // hop_size + 1)
    rms_sequence = []

    for idx in range(n_frames):
        start = idx * hop_size
        end = start + fft_size
        chunk = np.ascontiguousarray(audio[start:end], dtype=np.float64)
        frame = stft.analyze_frame(chunk, start / sr)

        mag = np.array(frame.magnitude)
        band_energy = 0.0

        for (low_hz, high_hz, weight) in bands:
            low_bin  = max(1, int(np.ceil(low_hz  * fft_size / sr)))
            high_bin = min(len(mag) - 1, int(np.floor(high_hz * fft_size / sr)))

            if low_bin > high_bin:
                continue

            band_mag = mag[low_bin:high_bin+1]
            energy   = np.sqrt(np.mean(band_mag ** 2))
            band_energy += energy * weight

        rms_sequence.append(float(band_energy))

    return rms_sequence


def export_for_ios(map_gz_path, out_json_path):
    print(f"Loading {map_gz_path}...")
    fmap = FrequencyMap.load(map_gz_path)

    stem_names = fmap.stem_names()
    print(f"Stems: {stem_names}")
    print("Computing band-limited RMS sequences to match iOS spectral estimator...")

    stems_band_rms = {}
    for name, profile in fmap.stems.items():
        # We need the raw audio to recompute — use the stored per-frame RMS
        # as a fallback if audio isn't available, but ideally recompute from audio.
        # Since FrequencyMap stores profile.rms (broadband), we recompute
        # using the stem audio if available, otherwise fall back.
        bands = STEM_BANDS.get(name, DEFAULT_BAND)

        if hasattr(profile, '_audio') and profile._audio is not None:
            # Ideal path: recompute from raw audio
            audio, sr = profile._audio, fmap.sr
            band_rms = compute_band_rms_sequence(audio, sr,
                                                  fft_size=fmap.fft_size,
                                                  hop_size=fmap.hop_size,
                                                  bands=bands)
        else:
            # Fallback: use stored broadband RMS
            # Apply a rough scaling factor per stem based on typical
            # band energy fraction of broadband energy.
            # These factors were empirically derived and can be tuned.
            BAND_FRACTION = {
                "drums":  0.35,
                "bass":   0.40,
                "vocals": 0.45,
                "other":  0.30,
                "guitar": 0.30,
                "piano":  0.30,
                "keys":   0.30,
            }
            scale = BAND_FRACTION.get(name, 0.35)
            band_rms = [r * scale for r in profile.rms]
            print(f"  WARNING: {name} using broadband RMS * {scale:.2f} (no raw audio)")

        stems_band_rms[name] = band_rms
        n = len(band_rms)
        peak = max(band_rms) if band_rms else 0.0
        print(f"  {name:8s}: {n} frames, peak band RMS = {peak:.5f}")

    data = {
        "stem_names":     stem_names,
        "combined_chroma": fmap.combined_chroma.tolist(),
        "stems_rms":      stems_band_rms,
    }

    print(f"\nExporting to {out_json_path}...")
    with open(out_json_path, 'w') as f:
        json.dump(data, f)
    print("Done! Drag map.json into your Xcode project.")
    print("\nNOTE: For best results, re-run stem separation and pass raw stem")
    print("audio to compute_band_rms_sequence() instead of using the fallback.")


if __name__ == "__main__":
    export_for_ios("ref_data/map.gz", "map.json")