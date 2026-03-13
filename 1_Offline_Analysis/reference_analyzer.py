"""
Reference analyzer — performs per-stem spectral analysis with aggressive
filtering to produce realistic note counts.

This module is the bridge between Demucs-separated stems and the FrequencyMap
data structure. For each stem it runs STFT peak detection, partial tracking,
note extraction, chroma analysis, and RMS computation using the C++ ase_core
engine. Strict amplitude and duration gates suppress noise artifacts to ensure
only musically meaningful events survive.

Note: This module is considered deprecated in favor of the band-filtered
approach used in analyze_and_export.py for the iOS JSON export. It remains
useful for detailed offline analysis and diagnostic purposes.
"""

import numpy as np
import ase_core
from frequency_map import FrequencyMap, StemProfile


def _vec(x):
    """Ensure array is contiguous float64 for safe handoff to C++ code."""
    return np.ascontiguousarray(x, dtype=np.float64)


# ══════════════════════════════════════════════════════════
#  Analysis parameters — these thresholds control the tradeoff
#  between detecting subtle musical events and rejecting noise.
# ══════════════════════════════════════════════════════════
DEFAULT_PEAK_THRESHOLD_DB = -20.0   # Ignore spectral peaks below this dB level
MAX_PEAKS_PER_FRAME       = 4       # Keep only the top N peaks (fundamental + harmonics)
MIN_PARTIAL_DURATION      = 0.08    # Partials shorter than 80ms are treated as noise
PARTIAL_TOLERANCE_CENTS   = 50.0    # Frequency matching window for partial continuation
MAX_PARTIAL_GAP_FRAMES    = 2       # Maximum consecutive missed frames before a partial dies
MIN_NOTE_DURATION         = 0.20    # Notes shorter than 200ms are discarded
NOTE_PITCH_GATE_CENTS     = 100.0   # Pitch must stay within 1 semitone to remain one note
MIN_NOTE_AMPLITUDE_DB     = -30.0   # Notes quieter than -30 dB are considered ghost notes


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
    """
    Run full spectral analysis on each separated stem and build a FrequencyMap.
    
    For each stem:
      1. STFT analysis with peak detection (top N peaks per frame)
      2. Partial tracking across frames (frequency continuity)
      3. Note extraction from stable partials (pitch + duration gating)
      4. Per-frame chroma extraction (full spectral energy, not just peaks)
      5. Per-frame RMS energy computation
    
    Finally, computes a combined chromagram from the original full mix
    (preferred) or falls back to summing per-stem chromas.
    
    Args:
        stems:              Dict mapping stem name → (audio_array, sample_rate)
        full_mix:           Optional full mix audio for combined chroma extraction
        full_mix_sr:        Sample rate of the full mix
        fft_size:           FFT window size in samples
        hop_size:           Hop size between frames in samples
        peak_threshold_db:  Minimum dB level for peak detection
        max_peaks:          Maximum peaks to retain per frame
        partial_tol_cents:  Frequency tolerance for partial tracking (cents)
        min_note_dur:       Minimum note duration to keep (seconds)
        min_note_amp_db:    Minimum note amplitude to keep (dB)
    
    Returns:
        A fully populated FrequencyMap ready for serialization or iOS export.
    """

    fmap = FrequencyMap()
    fmap.fft_size = fft_size
    fmap.hop_size = hop_size

    # Convert dB amplitude gate to linear for note filtering
    min_note_amp_linear = 10 ** (min_note_amp_db / 20.0)

    for name, (audio, sr) in stems.items():
        audio = _vec(audio)
        fmap.sr = sr

        # Initialize C++ STFT engine with a Hann window
        stft = ase_core.STFT(fft_size, hop_size, sr, ase_core.Window.HANN)

        # Initialize partial tracker with tight gap tolerance for clean results
        tracker = ase_core.PartialTracker(
            partial_tol_cents,
            MIN_PARTIAL_DURATION,
            MAX_PARTIAL_GAP_FRAMES)

        profile = StemProfile(name)
        n_frames = max(0, (len(audio) - fft_size) // hop_size + 1)

        # Compute overall stem RMS for context (used in diagnostic logging)
        stem_rms = float(ase_core.compute_rms(_vec(audio)))
        stem_peak_db = 20 * np.log10(stem_rms + 1e-12)

        total_peaks = 0

        for idx in range(n_frames):
            start = idx * hop_size
            end = start + fft_size
            chunk = _vec(audio[start:end])

            # Run STFT on this frame
            frame = stft.analyze_frame(chunk, start / sr)

            # Detect spectral peaks above threshold
            peaks = stft.detect_peaks(frame, peak_threshold_db)

            # Limit to top N peaks by amplitude (detect_peaks returns sorted descending)
            peaks = peaks[:max_peaks]
            total_peaks += len(peaks)

            # Feed peaks into the partial tracker for cross-frame continuity
            tracker.feed(peaks, frame.time)

            # Store per-frame analysis results
            profile.times.append(frame.time)
            profile.peak_freqs.append([p.freq for p in peaks])
            profile.peak_amps.append([p.amp for p in peaks])
            profile.rms.append(float(ase_core.compute_rms(chunk)))

        # Chroma extraction: uses the full spectral energy (all bins, not just peaks)
        # for a more complete representation of harmonic content
        chroma_ext = ase_core.ChromaExtractor(fft_size, hop_size, sr)
        stem_chroma = np.array(chroma_ext.analyze(_vec(audio)))
        for i in range(n_frames):
            if i < len(stem_chroma):
                profile.chroma.append(stem_chroma[i])
            else:
                profile.chroma.append(np.zeros(12))

        # Finalize partial tracking and extract notes with strict filtering
        tracker.finish()
        partials = tracker.completed()

        # Extract notes from partials: groups of stable-pitch partial segments
        raw_notes = ase_core.PartialTracker.extract_notes(
            partials, min_note_dur, NOTE_PITCH_GATE_CENTS)

        # Amplitude gate: remove quiet ghost notes that are likely crosstalk or noise
        notes = [n for n in raw_notes if n.amp >= min_note_amp_linear]

        # Frequency gate: remove sub-bass rumble (<50 Hz) and ultrasonic artifacts (>8 kHz)
        notes = [n for n in notes if 50.0 <= n.freq <= 8000.0]

        profile.partials = partials
        profile.notes = notes
        fmap.stems[name] = profile

        # Summary statistics for this stem
        avg_peaks = total_peaks / max(n_frames, 1)
        print(f"    {name:8s}  {n_frames} frames, "
              f"avg {avg_peaks:.1f} peaks/frame, "
              f"{len(partials):5d} partials → "
              f"{len(raw_notes):5d} raw → "
              f"{len(notes):4d} notes")

    # Build the combined chromagram from the original full mix for best quality.
    # The full mix chroma captures inter-stem harmonic interactions that are
    # lost when analyzing stems individually.
    if full_mix is not None and full_mix_sr is not None:
        print("    computing combined chroma from original mix …", end=" ")
        ce = ase_core.ChromaExtractor(fft_size, hop_size, full_mix_sr)
        combined = np.array(ce.analyze(_vec(full_mix)))
        fmap.combined_chroma = combined
        print(f"({combined.shape[0]} frames)")
    else:
        # Fallback: sum per-stem chromas and normalize
        print("    WARNING: no full_mix provided")
        _build_combined_chroma_fallback(fmap)

    return fmap


def _build_combined_chroma_fallback(fmap: FrequencyMap):
    """
    Build a combined chromagram by summing all per-stem chromas and L2-normalizing.
    
    This is a fallback when the original full mix is not available. The result
    is less accurate than extracting chroma directly from the mix because
    stem separation introduces artifacts.
    """
    max_len = max(len(s.chroma) for s in fmap.stems.values())
    combined = np.zeros((max_len, 12), dtype=np.float64)
    for s in fmap.stems.values():
        for i, c in enumerate(s.chroma):
            combined[i] += np.asarray(c)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    combined /= norms
    fmap.combined_chroma = combined