"""
FrequencyMap : the central data-store that maps
  (stem, time-index) → {frequencies, amplitudes, RMS, chroma, notes}

Used both for song-position identification and for RMS matching.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class StemProfile:
    """All per-frame analysis data for one stem."""
    name:       str
    times:      list[float]                = field(default_factory=list)
    peak_freqs: list[list[float]]          = field(default_factory=list)
    peak_amps:  list[list[float]]          = field(default_factory=list)
    rms:        list[float]                = field(default_factory=list)
    chroma:     list[Any]                  = field(default_factory=list)
    partials:   list                       = field(default_factory=list)
    notes:      list                       = field(default_factory=list)

    # ---- queries ------------------------------------------------
    def rms_at(self, frame_idx: int) -> float:
        if 0 <= frame_idx < len(self.rms):
            return self.rms[frame_idx]
        return 0.0

    def rms_at_time(self, t: float) -> float:
        idx = self._time_to_idx(t)
        return self.rms_at(idx)

    def chroma_at(self, frame_idx: int) -> np.ndarray:
        if 0 <= frame_idx < len(self.chroma):
            return np.asarray(self.chroma[frame_idx])
        return np.zeros(12)

    def freqs_at(self, frame_idx: int) -> list[float]:
        if 0 <= frame_idx < len(self.peak_freqs):
            return self.peak_freqs[frame_idx]
        return []

    def _time_to_idx(self, t: float) -> int:
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
    """Top-level container for full reference analysis."""
    stems:             dict[str, StemProfile] = field(default_factory=dict)
    combined_chroma:   np.ndarray = field(default_factory=lambda: np.empty(0))
    sr:                int = 44100
    fft_size:          int = 4096
    hop_size:          int = 1024

    @property
    def n_frames(self) -> int:
        return len(self.combined_chroma)

    @property
    def frame_duration(self) -> float:
        return self.hop_size / self.sr

    def time_of_frame(self, idx: int) -> float:
        return idx * self.frame_duration

    def stem_names(self) -> list[str]:
        return list(self.stems.keys())

    # ---- lookup at a reference frame index ----------------------
    def rms_snapshot(self, frame_idx: int) -> dict[str, float]:
        """Return {stem_name: rms} at a given reference frame."""
        return {name: s.rms_at(frame_idx)
                for name, s in self.stems.items()}

    def chroma_at(self, frame_idx: int) -> np.ndarray:
        if 0 <= frame_idx < len(self.combined_chroma):
            return self.combined_chroma[frame_idx]
        return np.zeros(12)

    def serialize(self, path: str):
        """Persist to disk by flattening C++ objects into Python dicts."""
        import pickle, gzip
        
        # We need to transform the data structure to remove C++ objects
        serializable_stems = {}
        for name, profile in self.stems.items():
            # Convert partials (ase_core.Partial) to dicts
            clean_partials = []
            for p in profile.partials:
                clean_partials.append({
                    'id': p.id,
                    'times': list(p.times),
                    'freqs': list(p.freqs),
                    'amps': list(p.amps),
                    'phases': list(p.phases)
                })
            
            # Convert notes (ase_core.Note) to dicts
            clean_notes = []
            for n in profile.notes:
                clean_notes.append({
                    'start': n.start,
                    'end': n.end,
                    'freq': n.freq,
                    'amp': n.amp,
                    'midi': n.midi
                })

            # Create a serializable version of the StemProfile
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
        """Load from disk and reconstruct the FrequencyMap / StemProfile objects."""
        import pickle, gzip
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        
        fmap = FrequencyMap(
            combined_chroma=data['combined_chroma'],
            sr=data['sr'],
            fft_size=data['fft_size'],
            hop_size=data['hop_size']
        )

        # Reconstruct StemProfile objects
        # Note: partials/notes remain as dicts here, which is fine for Python use.
        # If your C++ engine needs them back as C++ types, you'd need a reconstruction loop.
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

# """
# FrequencyMap : the central data-store that maps
#   (stem, time-index) → {frequencies, amplitudes, RMS, chroma, notes}

# Used both for song-position identification and for RMS matching.
# """

# from dataclasses import dataclass, field
# from typing import Any
# import numpy as np


# @dataclass
# class StemProfile:
#     """All per-frame analysis data for one stem."""
#     name:       str
#     times:      list[float]              = field(default_factory=list)
#     peak_freqs: list[list[float]]        = field(default_factory=list)
#     peak_amps:  list[list[float]]        = field(default_factory=list)
#     rms:        list[float]              = field(default_factory=list)
#     chroma:     list[Any]                = field(default_factory=list)
#     partials:   list                     = field(default_factory=list)
#     notes:      list                     = field(default_factory=list)

#     # ---- queries ------------------------------------------------
#     def rms_at(self, frame_idx: int) -> float:
#         if 0 <= frame_idx < len(self.rms):
#             return self.rms[frame_idx]
#         return 0.0

#     def rms_at_time(self, t: float) -> float:
#         idx = self._time_to_idx(t)
#         return self.rms_at(idx)

#     def chroma_at(self, frame_idx: int) -> np.ndarray:
#         if 0 <= frame_idx < len(self.chroma):
#             return np.asarray(self.chroma[frame_idx])
#         return np.zeros(12)

#     def freqs_at(self, frame_idx: int) -> list[float]:
#         if 0 <= frame_idx < len(self.peak_freqs):
#             return self.peak_freqs[frame_idx]
#         return []

#     def _time_to_idx(self, t: float) -> int:
#         if not self.times:
#             return 0
#         # binary search
#         lo, hi = 0, len(self.times) - 1
#         while lo < hi:
#             mid = (lo + hi) // 2
#             if self.times[mid] < t:
#                 lo = mid + 1
#             else:
#                 hi = mid
#         return lo


# @dataclass
# class FrequencyMap:
#     """Top-level container for full reference analysis."""
#     stems:            dict[str, StemProfile] = field(default_factory=dict)
#     combined_chroma:  np.ndarray = field(default_factory=lambda: np.empty(0))
#     sr:               int = 44100
#     fft_size:         int = 4096
#     hop_size:         int = 1024

#     @property
#     def n_frames(self) -> int:
#         return len(self.combined_chroma)

#     @property
#     def frame_duration(self) -> float:
#         return self.hop_size / self.sr

#     def time_of_frame(self, idx: int) -> float:
#         return idx * self.frame_duration

#     def stem_names(self) -> list[str]:
#         return list(self.stems.keys())

#     # ---- lookup at a reference frame index ----------------------
#     def rms_snapshot(self, frame_idx: int) -> dict[str, float]:
#         """Return {stem_name: rms} at a given reference frame."""
#         return {name: s.rms_at(frame_idx)
#                 for name, s in self.stems.items()}

#     def chroma_at(self, frame_idx: int) -> np.ndarray:
#         if 0 <= frame_idx < len(self.combined_chroma):
#             return self.combined_chroma[frame_idx]
#         return np.zeros(12)

#     def serialize(self, path: str):
#         """Persist to disk (numpy npz + pickle for notes)."""
#         import pickle, gzip
#         with gzip.open(path, "wb") as f:
#             pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

#     @staticmethod
#     def load(path: str) -> "FrequencyMap":
#         import pickle, gzip
#         with gzip.open(path, "rb") as f:
#             return pickle.load(f)