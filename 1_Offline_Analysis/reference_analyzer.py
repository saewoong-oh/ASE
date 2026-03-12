"""
Reference analyzer — AGGRESSIVE filtering for realistic note counts.
"""

import numpy as np
import ase_core
from frequency_map import FrequencyMap, StemProfile


def _vec(x):
    return np.ascontiguousarray(x, dtype=np.float64)


# ══════════════════════════════════════════════════════════
#  TUNED PARAMETERS — these are the ones that matter
# ══════════════════════════════════════════════════════════
DEFAULT_PEAK_THRESHOLD_DB = -20.0   # ignore everything below -35 dB
MAX_PEAKS_PER_FRAME       = 4       # only top 4 — fundamental + harmonics
MIN_PARTIAL_DURATION      = 0.08    # partials shorter than 80ms = noise
PARTIAL_TOLERANCE_CENTS   = 50.0    # frequency matching window
MAX_PARTIAL_GAP_FRAMES    = 2       # max 2 missed frames before death
MIN_NOTE_DURATION         = 0.20    # notes shorter than 200ms = not real
NOTE_PITCH_GATE_CENTS     = 100.0   # pitch must be stable within 1 semitone
MIN_NOTE_AMPLITUDE_DB     = -30.0   # kill anything quieter than -30 dB


def analyze_reference(stems: dict[str, tuple[np.ndarray, int]],
                      full_mix: np.ndarray | None = None,
                      full_mix_sr: int | None = None,
                      fft_size: int = 4096,
                      hop_size: int = 1024,
                      peak_threshold_db: float = DEFAULT_PEAK_THRESHOLD_DB,
                      max_peaks: int = MAX_PEAKS_PER_FRAME,
                      partial_tol_cents: float = PARTIAL_TOLERANCE_CENTS,
                      min_note_dur: float = MIN_NOTE_DURATION,
                      min_note_amp_db: float = MIN_NOTE_AMPLITUDE_DB
                      ) -> FrequencyMap:

    fmap = FrequencyMap()
    fmap.fft_size = fft_size
    fmap.hop_size = hop_size

    min_note_amp_linear = 10 ** (min_note_amp_db / 20.0)

    for name, (audio, sr) in stems.items():
        audio = _vec(audio)
        fmap.sr = sr

        stft = ase_core.STFT(fft_size, hop_size, sr, ase_core.Window.HANN)

        # Tighter partial tracker: shorter gap tolerance
        tracker = ase_core.PartialTracker(
            partial_tol_cents,
            MIN_PARTIAL_DURATION,
            MAX_PARTIAL_GAP_FRAMES)

        profile = StemProfile(name)
        n_frames = max(0, (len(audio) - fft_size) // hop_size + 1)

        # Compute the stem's peak RMS for relative amplitude gating
        stem_rms = float(ase_core.compute_rms(_vec(audio)))
        stem_peak_db = 20 * np.log10(stem_rms + 1e-12)

        total_peaks = 0

        for idx in range(n_frames):
            start = idx * hop_size
            end = start + fft_size
            chunk = _vec(audio[start:end])

            frame = stft.analyze_frame(chunk, start / sr)

            # Detect peaks above threshold
            peaks = stft.detect_peaks(frame, peak_threshold_db)

            # LIMIT to top N by amplitude (already sorted descending)
            peaks = peaks[:max_peaks]
            total_peaks += len(peaks)

            tracker.feed(peaks, frame.time)

            profile.times.append(frame.time)
            profile.peak_freqs.append([p.freq for p in peaks])
            profile.peak_amps.append([p.amp for p in peaks])
            profile.rms.append(float(ase_core.compute_rms(chunk)))

        # ── chroma (uses ALL spectral energy, not just peaks) ──
        chroma_ext = ase_core.ChromaExtractor(fft_size, hop_size, sr)
        stem_chroma = np.array(chroma_ext.analyze(_vec(audio)))
        for i in range(n_frames):
            if i < len(stem_chroma):
                profile.chroma.append(stem_chroma[i])
            else:
                profile.chroma.append(np.zeros(12))

        # ── partials → notes with STRICT filtering ──
        tracker.finish()
        partials = tracker.completed()

        raw_notes = ase_core.PartialTracker.extract_notes(
            partials, min_note_dur, NOTE_PITCH_GATE_CENTS)

        # Amplitude gate: remove quiet ghost notes
        notes = [n for n in raw_notes if n.amp >= min_note_amp_linear]

        # Frequency gate: remove sub-bass rumble and ultrasonic noise
        notes = [n for n in notes if 50.0 <= n.freq <= 8000.0]

        profile.partials = partials
        profile.notes = notes
        fmap.stems[name] = profile

        avg_peaks = total_peaks / max(n_frames, 1)
        print(f"    {name:8s}  {n_frames} frames, "
              f"avg {avg_peaks:.1f} peaks/frame, "
              f"{len(partials):5d} partials → "
              f"{len(raw_notes):5d} raw → "
              f"{len(notes):4d} notes")

    # ── combined chroma from original mix ──
    if full_mix is not None and full_mix_sr is not None:
        print("    computing combined chroma from original mix …", end=" ")
        ce = ase_core.ChromaExtractor(fft_size, hop_size, full_mix_sr)
        combined = np.array(ce.analyze(_vec(full_mix)))
        fmap.combined_chroma = combined
        print(f"({combined.shape[0]} frames)")
    else:
        print("    WARNING: no full_mix provided")
        _build_combined_chroma_fallback(fmap)

    return fmap


def _build_combined_chroma_fallback(fmap: FrequencyMap):
    max_len = max(len(s.chroma) for s in fmap.stems.values())
    combined = np.zeros((max_len, 12), dtype=np.float64)
    for s in fmap.stems.values():
        for i, c in enumerate(s.chroma):
            combined[i] += np.asarray(c)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    combined /= norms
    fmap.combined_chroma = combined
