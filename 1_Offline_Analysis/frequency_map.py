"""
FrequencyMap : the central data-store that maps
  (stem, time-index) → {frequencies, amplitudes, RMS, chroma, notes}

Used both for song-position identification (via chroma matching) and
for RMS-based mix advisory (target gain calculation per stem).

This module defines two core data classes:
  - StemProfile: per-frame analysis data for a single instrument stem
  - FrequencyMap: top-level container holding all stems plus combined chroma
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class StemProfile:
    """
    All per-frame analysis data for one separated stem (e.g., vocals, drums).
    
    Each list is indexed by frame number. The frame rate is determined by
    hop_size / sample_rate (typically ~23.2ms per frame at 44100 Hz / 1024 hop).
    """
    name:       str
    times:      list[float]                = field(default_factory=list)   # Timestamp (seconds) of each frame center
    peak_freqs: list[list[float]]          = field(default_factory=list)   # Detected spectral peak frequencies per frame
    peak_amps:  list[list[float]]          = field(default_factory=list)   # Amplitudes of detected peaks per frame
    rms:        list[float]                = field(default_factory=list)   # Root-mean-square energy per frame
    chroma:     list[Any]                  = field(default_factory=list)   # 12-bin chroma vector per frame
    partials:   list                       = field(default_factory=list)   # Tracked sinusoidal partials (from PartialTracker)
    notes:      list                       = field(default_factory=list)   # Extracted musical notes (from partial grouping)

    # ---- queries ------------------------------------------------

    def rms_at(self, frame_idx: int) -> float:
        """Return RMS energy at a specific frame index, or 0.0 if out of bounds."""
        if 0 <= frame_idx < len(self.rms):
            return self.rms[frame_idx]
        return 0.0

    def rms_at_time(self, t: float) -> float:
        """Return RMS energy at the frame nearest to time t (seconds)."""
        idx = self._time_to_idx(t)
        return self.rms_at(idx)

    def chroma_at(self, frame_idx: int) -> np.ndarray:
        """Return the 12-bin chroma vector at a frame index, or zeros if out of bounds."""
        if 0 <= frame_idx < len(self.chroma):
            return np.asarray(self.chroma[frame_idx])
        return np.zeros(12)

    def freqs_at(self, frame_idx: int) -> list[float]:
        """Return the list of detected peak frequencies at a frame index."""
        if 0 <= frame_idx < len(self.peak_freqs):
            return self.peak_freqs[frame_idx]
        return []

    def _time_to_idx(self, t: float) -> int:
        """
        Binary search to find the frame index closest to time t.
        Returns the index of the first frame with time >= t.
        """
        if not self.times:
            return 0
        lo, hi = 0, len(self.times) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.times[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        return lo


@dataclass
class FrequencyMap:
    """
    Top-level container for the complete reference analysis of a song.
    
    Holds per-stem StemProfile objects and a combined chromagram derived
    from either the original mix or a fallback sum of stem chromas.
    This is the primary data structure serialized for the iOS app.
    """
    stems:             dict[str, StemProfile] = field(default_factory=dict)
    combined_chroma:   np.ndarray = field(default_factory=lambda: np.empty(0))  # Shape [n_frames, 12]
    sr:                int = 44100     # Sample rate used during analysis
    fft_size:          int = 4096     # FFT window size
    hop_size:          int = 1024     # Hop size between consecutive frames

    @property
    def n_frames(self) -> int:
        """Total number of analysis frames in the combined chromagram."""
        return len(self.combined_chroma)

    @property
    def frame_duration(self) -> float:
        """Duration of one analysis frame in seconds."""
        return self.hop_size / self.sr

    def time_of_frame(self, idx: int) -> float:
        """Convert a frame index to its corresponding time in seconds."""
        return idx * self.frame_duration

    def stem_names(self) -> list[str]:
        """Return the list of stem names present in this map."""
        return list(self.stems.keys())

    # ---- lookup at a reference frame index ----------------------

    def rms_snapshot(self, frame_idx: int) -> dict[str, float]:
        """Return a dict of {stem_name: rms} for all stems at a given reference frame."""
        return {name: s.rms_at(frame_idx)
                for name, s in self.stems.items()}

    def chroma_at(self, frame_idx: int) -> np.ndarray:
        """Return the combined 12-bin chroma vector at a frame index."""
        if 0 <= frame_idx < len(self.combined_chroma):
            return self.combined_chroma[frame_idx]
        return np.zeros(12)

    def serialize(self, path: str):
        """
        Persist the FrequencyMap to disk as a gzipped pickle file.
        
        C++ objects (ase_core.Partial, ase_core.Note) are converted to plain
        Python dicts so that the file can be loaded without the C++ extension.
        """
        import pickle, gzip
        
        serializable_stems = {}
        for name, profile in self.stems.items():
            # Convert C++ Partial objects to serializable dicts
            clean_partials = []
            for p in profile.partials:
                clean_partials.append({
                    'id': p.id,
                    'times': list(p.times),
                    'freqs': list(p.freqs),
                    'amps': list(p.amps),
                    'phases': list(p.phases)
                })
            
            # Convert C++ Note objects to serializable dicts
            clean_notes = []
            for n in profile.notes:
                clean_notes.append({
                    'start': n.start,
                    'end': n.end,
                    'freq': n.freq,
                    'amp': n.amp,
                    'midi': n.midi
                })

            serializable_stems[name] = {
                'times': profile.times,
                'peak_freqs': profile.peak_freqs,
                'peak_amps': profile.peak_amps,
                'rms': profile.rms,
                'chroma': profile.chroma,
                'partials': clean_partials,
                'notes': clean_notes
            }

        data = {
            'stems': serializable_stems,
            'combined_chroma': self.combined_chroma,
            'sr': self.sr,
            'fft_size': self.fft_size,
            'hop_size': self.hop_size
        }

        with gzip.open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "FrequencyMap":
        """
        Load a FrequencyMap from a gzipped pickle file and reconstruct
        StemProfile objects.
        
        Note: partials and notes are loaded as plain dicts, not C++ objects.
        This is sufficient for Python-side usage; reconstructing C++ types
        would require an additional conversion step.
        """
        import pickle, gzip
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        
        fmap = FrequencyMap(
            combined_chroma=data['combined_chroma'],
            sr=data['sr'],
            fft_size=data['fft_size'],
            hop_size=data['hop_size']
        )

        for name, s_data in data['stems'].items():
            profile = StemProfile(
                name=name,
                times=s_data['times'],
                peak_freqs=s_data['peak_freqs'],
                peak_amps=s_data['peak_amps'],
                rms=s_data['rms'],
                chroma=s_data['chroma'],
                partials=s_data['partials'],
                notes=s_data['notes']
            )
            fmap.stems[name] = profile
            
        return fmap