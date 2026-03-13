#!/usr/bin/env python3
"""
generate_comparative_results.py
-------------------------------
Runs a fully synthetic simulation of the ASE pipeline to compare the
theoretical Demucs real-time input against the HPSS iOS simulation.

This script generates synthetic reference data (chroma + RMS), simulates
a live performance with tempo drift and noise, runs the Kalman-based
position tracker, and measures both tracking accuracy and per-stem
RMS advisory error.

The comparison quantifies how much accuracy is lost when using HPSS
(available in real-time on iOS) versus Demucs (offline only) for
stem-level metering.

Outputs:
  results_comparative_metrics.json   – tracking and advisory accuracy numbers
  fig_tracking_confidence.png        – confidence, error, and position over time
  fig_confidence_histogram.png       – distribution of per-frame confidence scores
  fig_rms_error_comparison.png       – per-stem dB error: Demucs vs HPSS
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# -- Reproducibility & Simulation Parameters -----------------------------------

RNG = np.random.default_rng(42)  # Fixed seed for reproducible results

SR = 44100                          # Sample rate (Hz)
HOP = 1024                         # Hop size in samples
FFT_SIZE = 4096                     # FFT window size
N_REF = 4096                        # Number of reference frames (~95.1 seconds of audio)
N_LIVE = 3800                       # Number of simulated live frames
TEMPO_DRIFT = 1.02                  # Simulated tempo: live runs 2% faster than reference
NOISE_STD = 0.08                    # Standard deviation of additive Gaussian noise on chroma
LOCK_DELAY = int(SR / HOP) * 2     # Initial frames before tracker receives meaningful input

STEM_NAMES = ["drums", "bass", "vocals", "other"]

# Kalman filter / position tracker hyper-parameters
CONF_THRESH = 0.15                  # Minimum confidence to accept a match
MAX_STEP = 3.0                      # Maximum forward position jump per frame
MIN_STEP = -2.0                     # Maximum backward position correction per frame
PLOT_DPI = 180                      # Resolution for saved figure images


# -- 1. Synthetic reference chroma --------------------------------------------

def make_reference_chroma(n: int) -> np.ndarray:
    """
    Generate a synthetic reference chromagram with smoothly varying pitch content.
    
    Each of the 12 chroma bins follows a unique sinusoidal envelope, producing
    a deterministic but musically plausible pattern. The result is L2-normalized
    per frame to match real chroma behavior.
    """
    t = np.linspace(0, 1, n)
    chroma = np.zeros((n, 12))
    for c in range(12):
        freq = 0.3 + c * 0.07        # Each chroma bin oscillates at a different rate
        phase = RNG.uniform(0, 2 * np.pi)
        chroma[:, c] = np.clip(0.5 + 0.5 * np.sin(2 * np.pi * freq * t + phase), 0, 1)
    # L2 normalize each frame (same normalization used in the real pipeline)
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return chroma / norms


# -- 2. Synthetic live chroma -------------------------------------------------

def make_live_chroma(ref: np.ndarray, n_live: int, tempo_drift: float, 
                     noise_std: float, lock_delay: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a live chroma stream from a reference, with tempo drift and noise.
    
    The first `lock_delay` frames are pure random (simulating the pre-lock period
    before the tracker acquires the song position). After that, frames are
    noisy copies of the reference at the tempo-drifted position.
    
    Returns:
        live:       (n_live, 12) array of noisy live chroma frames
        gt_pos:     (n_live,) array of ground-truth reference frame indices
    """
    live = np.zeros((n_live, 12))
    gt_pos = np.zeros(n_live, dtype=int)
    for i in range(n_live):
        # Ground truth position accounts for tempo drift
        true_pos = int(np.clip(i * tempo_drift, 0, len(ref) - 1))
        gt_pos[i] = true_pos
        if i < lock_delay:
            # Pre-lock: random garbage to test tracker's acquisition behavior
            frame = RNG.uniform(0, 1, 12)
        else:
            # Post-lock: reference + Gaussian noise simulating real-world conditions
            frame = ref[true_pos] + RNG.normal(0, noise_std, 12)
            frame = np.clip(frame, 0, None)
        norm = np.linalg.norm(frame)
        live[i] = frame / norm if norm > 1e-12 else frame
    return live, gt_pos


# -- 3. Kalman-based position tracker -----------------------------------------

@dataclass
class PositionTracker:
    """
    Python reimplementation of the C++ PositionTracker for simulation purposes.
    
    Uses sliding-window chroma cross-correlation with a Gaussian continuity
    prior to estimate the current song position. An EMA-smoothed confidence
    score gates the position update to prevent jumps during low-quality matches.
    
    This mirrors the logic in position_tracker.cpp but runs in pure Python
    for offline benchmarking.
    """
    ref_norm: np.ndarray          # Pre-normalized reference chroma, shape [N, 12]
    history_len: int = 300        # Maximum number of live frames to keep in the sliding template
    radius: int = 100             # Search radius (frames) around the expected position when locked
    sigma: float = 10.0           # Gaussian continuity prior width (frames)
    conf_thresh: float = 0.15     # Minimum similarity to accept a position update
    max_tempo: float = 1.10       # Upper bound for tempo ratio estimation
    min_tempo: float = 0.90       # Lower bound for tempo ratio estimation

    # Internal state
    pos: float = 0.0              # Current estimated position (fractional frame index)
    tempo: float = 1.0            # Estimated tempo ratio (live/reference speed)
    conf_ema: float = 0.0         # Exponentially smoothed confidence
    confidence: float = 0.0       # Published confidence value [0, 1]
    locked: bool = False          # Whether the tracker has acquired a stable lock
    lock_count: int = 0           # Consecutive frames with good confidence
    history: list = field(default_factory=list)  # Sliding window of recent live chroma frames
    _j: int = 0                   # Frame counter

    def process(self, live_frame: np.ndarray) -> Tuple[int, float]:
        """
        Process one live chroma frame and return (estimated_position, confidence).
        
        The algorithm:
          1. Normalize the incoming frame and append to history
          2. Determine search window size based on lock state
          3. Slide the history template over the reference within the window
          4. Score each candidate by similarity × Gaussian continuity prior
          5. Update position using the best match, gated by confidence
        """
        ln = np.linalg.norm(live_frame)
        if ln < 1e-3:
            # Silence: coast forward at current tempo, decay confidence
            self.pos = np.clip(self.pos + self.tempo, 0, len(self.ref_norm) - 1)
            self.confidence *= 0.95
            self._j += 1
            return int(round(self.pos)), self.confidence

        # L2 normalize the live frame
        lc = live_frame / ln
        self.history.append(lc)
        if len(self.history) > self.history_len:
            self.history.pop(0)

        T = len(self.history)    # Current template length
        N = len(self.ref_norm)   # Total reference length
        exp = np.clip(self.pos + self.tempo, 0, N - 1)  # Expected next position

        # Adaptive search window: wide when unlocked, narrow when locked
        if T < 15:
            r, sig = 2, self.sigma * 0.5        # Very early: minimal search
        elif not self.locked:
            r, sig = min(1500, N // 2), self.sigma * 20.0  # Unlocked: wide search
        else:
            r, sig = self.radius, self.sigma     # Locked: tight search

        # Define the search region in reference frame indices
        lo = max(0, int(exp) - T + 1 - r)
        hi = min(N, int(exp) + r + 1)
        S = hi - lo  # Source window length

        if S < T:
            # Search window too small for the template
            self.pos = exp
            self._j += 1
            return int(round(self.pos)), self.confidence

        # Build template and source matrices for cross-correlation
        tpl = np.array(self.history)
        src = self.ref_norm[lo:hi]

        # Brute-force sliding dot product with Gaussian continuity weighting
        best_score = -1e30
        best_sim = 0.0
        best_pos = exp
        result_len = S - T + 1

        for d in range(result_len):
            # Average per-frame cosine similarity over the template window
            sim = float(np.sum(tpl * src[d:d + T]) / T)
            ref_p = float(lo + d + T - 1)       # Reference position this match corresponds to
            offset = ref_p - exp                 # Deviation from expected position
            cont = np.exp(-0.5 * (offset / sig) ** 2)  # Gaussian continuity prior
            score = sim * cont
            if score > best_score:
                best_score = score
                best_sim = sim
                best_pos = ref_p

        # Update confidence with exponential moving average
        self.conf_ema = 0.75 * self.conf_ema + 0.25 * best_sim
        self.confidence = np.clip(self.conf_ema, 0.0, 1.0)

        # Position update, gated by confidence threshold
        if best_sim >= self.conf_thresh:
            innovation = best_pos - exp
            if not self.locked and T >= 15:
                cl = innovation                  # Unlocked: allow large jumps for initial acquisition
            else:
                cl = np.clip(innovation, MIN_STEP, MAX_STEP)  # Locked: clamp step size

            # Adaptive gain: higher similarity → stronger correction
            gain = min(0.85, best_sim)
            if best_sim < 0.6:
                gain *= best_sim / 0.6           # Further attenuate when match quality is marginal

            self.pos = np.clip(exp + gain * cl, 0, N - 1)
            self.lock_count = min(self.lock_count + 1, 60)
        else:
            # Poor match: coast at expected position, decay lock count
            self.pos = exp
            self.lock_count = max(0, self.lock_count - 2)

        self.locked = self.lock_count >= 5
        self._j += 1
        return int(round(self.pos)), self.confidence


# -- 4. Synthetic stem RMS profiles -------------------------------------------

def make_stem_rms(n: int) -> Dict[str, np.ndarray]:
    """
    Generate synthetic per-stem RMS envelopes that mimic typical instrument dynamics.
    
    Each stem gets a sinusoidal envelope with instrument-appropriate amplitude
    range and modulation rate, simulating how drums are louder and punchier
    while vocals have a slower, more sustained envelope.
    """
    t = np.linspace(0, 1, n)
    profiles = {}
    # (min_rms, max_rms, modulation_frequency) for each stem
    params = {
        "drums":  (0.25, 0.80, 1.5),
        "bass":   (0.10, 0.40, 0.7),
        "vocals": (0.05, 0.35, 1.1),
        "other":  (0.08, 0.30, 0.9),
    }
    for name, (lo, hi, freq) in params.items():
        env = lo + (hi - lo) * np.clip(
            0.5 + 0.5 * np.sin(2 * np.pi * freq * t + RNG.uniform(0, np.pi)), 0, 1)
        profiles[name] = env
    return profiles


# -- 5. Plotting Functions ----------------------------------------------------

def plot_confidence_and_error(confidences, pos_errors, gt_positions, positions):
    """
    Generate a 3-panel time-series plot showing:
      - Top: tracking confidence over time with threshold lines
      - Middle: absolute position error in seconds vs ground truth
      - Bottom: estimated vs ground-truth song position
    """
    t = np.arange(len(confidences)) * HOP / SR
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    # Panel 1: Confidence curve
    axes[0].plot(t, confidences, color="#3498DB", lw=0.9, label="Confidence")
    axes[0].axhline(0.85, color="grey", ls="--", lw=0.8, label="85 % threshold")
    axes[0].axhline(0.50, color="grey", ls=":", lw=0.8, label="50 % threshold")
    axes[0].set_ylabel("Confidence")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Tracking Confidence over Time")

    # Panel 2: Position error in seconds
    err_s = pos_errors * HOP / SR
    axes[1].plot(t, err_s, color="#E74C3C", lw=0.9)
    axes[1].set_ylabel("Position Error (s)")
    axes[1].set_title("Absolute Position Error vs. Ground Truth")

    # Panel 3: Estimated position vs ground truth
    axes[2].plot(t, gt_positions * HOP / SR, color="black", lw=0.8, ls="--", label="Ground Truth")
    axes[2].plot(t, positions * HOP / SR, color="#27AE60", lw=0.9, alpha=0.85, label="Estimated")
    axes[2].set_ylabel("Song Position (s)")
    axes[2].set_xlabel("Wall-Clock Time (s)")
    axes[2].set_title("Estimated Position vs. Ground Truth")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("fig_tracking_confidence.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: fig_tracking_confidence.png")


def plot_confidence_histogram(confidences):
    """Plot histogram of per-frame confidence scores to visualize lock quality distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(confidences, bins=50, color="#3498DB", edgecolor="white", alpha=0.85)
    ax.axvline(0.85, color="red", ls="--", lw=1.2, label="85 % threshold")
    ax.axvline(0.50, color="grey", ls=":", lw=1.2, label="50 % threshold")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frame Count")
    ax.set_title("Distribution of Per-Frame Tracking Confidence")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("fig_confidence_histogram.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: fig_confidence_histogram.png")


def plot_rms_error_comparison(demucs_errors, hpss_errors):
    """
    Plot per-stem dB error over time for both Demucs (ideal) and HPSS (iOS) inputs.
    
    This visualizes the accuracy penalty of using HPSS harmonic/percussive
    separation instead of true stem isolation for mix advisory metering.
    """
    t_axis = np.arange(N_LIVE) * HOP / SR
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for idx, n in enumerate(STEM_NAMES):
        axes[idx].plot(t_axis, demucs_errors[n], label="Demucs Error (Ideal)", color="#2ECC71", alpha=0.8, lw=1)
        axes[idx].plot(t_axis, hpss_errors[n], label="HPSS Error (iOS)", color="#E74C3C", alpha=0.8, lw=1)
        axes[idx].set_ylabel("Error (dB)")
        axes[idx].set_title(n.capitalize(), fontsize=10)
        axes[idx].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Wall-Clock Time (s)")
    plt.tight_layout()
    plt.savefig("fig_rms_error_comparison.png", dpi=PLOT_DPI)
    plt.close()
    print("Saved: fig_rms_error_comparison.png")


# -- 6. Run simulation & Compare ----------------------------------------------

def run_simulation():
    """
    Main simulation entry point.
    
    Generates synthetic data, runs the position tracker, computes Demucs vs HPSS
    advisory errors, prints summary metrics, and saves plots + JSON results.
    """
    print("Building reference chroma …")
    ref_chroma = make_reference_chroma(N_REF)

    print("Building live chroma stream …")
    live_chroma, gt_positions = make_live_chroma(
        ref_chroma, N_LIVE, TEMPO_DRIFT, NOISE_STD, LOCK_DELAY)

    print("Building reference RMS profiles …")
    ref_rms = make_stem_rms(N_REF)

    # Simulate what Demucs and HPSS would produce as real-time stem-level meters.
    # Demucs gives per-stem RMS directly; HPSS lumps all non-drum stems together.
    print("Simulating Demucs & HPSS Real-Time Streams …")
    live_demucs = {}
    for name, env in ref_rms.items():
        # Demucs output: reference RMS at the tempo-drifted position + ±15% gain variation
        live_demucs[name] = np.array([
            env[int(np.clip(i * TEMPO_DRIFT, 0, N_REF - 1))] * RNG.uniform(0.85, 1.15)
            for i in range(N_LIVE)
        ])

    # HPSS only separates into two streams: percussive (≈drums) and harmonic (≈everything else)
    live_hpss = {
        "percussive": live_demucs["drums"],
        "harmonic": live_demucs["bass"] + live_demucs["vocals"] + live_demucs["other"]
    }

    # Run the position tracker on the simulated live chroma
    tracker = PositionTracker(ref_norm=ref_chroma)

    positions = np.zeros(N_LIVE, dtype=int)
    confidences = np.zeros(N_LIVE)
    pos_errors = np.zeros(N_LIVE)

    print("Running frame-by-frame tracking …")
    for i in range(N_LIVE):
        pos, conf = tracker.process(live_chroma[i])
        positions[i] = pos
        confidences[i] = conf
        pos_errors[i] = abs(pos - gt_positions[i])

    # Calculate per-stem dB errors for both Demucs and HPSS scenarios
    print("Calculating dB Errors …")
    demucs_errors = {n: [] for n in STEM_NAMES}
    hpss_errors = {n: [] for n in STEM_NAMES}

    for i in range(N_LIVE):
        est_pos = positions[i]
        for n in STEM_NAMES:
            # Reference RMS at the estimated position (floor at -120 dB)
            r_val = max(ref_rms[n][est_pos], 1e-6)
            # Demucs: per-stem RMS directly available
            ld_val = max(live_demucs[n][i], 1e-6)
            # HPSS: drums map to percussive, everything else maps to harmonic
            lh_val = max(live_hpss["percussive"][i] if n == "drums" else live_hpss["harmonic"][i], 1e-6)

            # Convert to dB and compute absolute error
            r_db = 20 * np.log10(r_val)
            ld_db = 20 * np.log10(ld_val)
            lh_db = 20 * np.log10(lh_val)

            demucs_errors[n].append(abs(r_db - ld_db))
            hpss_errors[n].append(abs(r_db - lh_db))

    # --- Compute Extended Tracking Metrics ---
    frame_dur = HOP / SR
    mean_conf = float(np.mean(confidences))
    above_85 = float(np.mean(confidences >= 0.85) * 100)
    above_50 = float(np.mean(confidences >= 0.50) * 100)
    locked_frac = float(np.mean(confidences >= CONF_THRESH) * 100)

    mean_err_s = float(np.mean(pos_errors) * frame_dur)
    median_err_s = float(np.median(pos_errors) * frame_dur)
    p90_err_s = float(np.percentile(pos_errors, 90) * frame_dur)

    # Time-to-lock: first frame where 20 consecutive frames have confidence >= 50%
    lock_frame = None
    for i in range(len(confidences) - 20):
        if np.all(confidences[i:i + 20] >= 0.50):
            lock_frame = i
            break
    time_to_lock = float(lock_frame * frame_dur) if lock_frame else float("nan")

    # Assemble results dictionary
    metrics = {
        "tracking": {
            "mean_confidence": mean_conf,
            "frames_above_85pct_conf": above_85,
            "frames_above_50pct_conf": above_50,
            "frames_above_thresh": locked_frac,
            "mean_position_error_s": mean_err_s,
            "median_position_error_s": median_err_s,
            "p90_position_error_s": p90_err_s,
            "time_to_lock_s": time_to_lock,
        },
        "advisory_accuracy_demucs": {
            n: {"MAE_dB": float(np.mean(demucs_errors[n]))} for n in STEM_NAMES
        },
        "advisory_accuracy_hpss": {
            n: {"MAE_dB": float(np.mean(hpss_errors[n]))} for n in STEM_NAMES
        }
    }

    with open("results_comparative_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary to console
    print("\n── Key Tracking Metrics ─────────────────────────────────────────")
    t = metrics["tracking"]
    print(f"  Mean confidence:           {t['mean_confidence']:.3f}")
    print(f"  Frames >= 85 % confidence: {t['frames_above_85pct_conf']:.1f} %")
    print(f"  Frames >= 50 % confidence: {t['frames_above_50pct_conf']:.1f} %")
    print(f"  Mean position error:       {t['mean_position_error_s']:.3f} s")
    print(f"  Median position error:     {t['median_position_error_s']:.3f} s")
    print(f"  90th-pct position error:   {t['p90_position_error_s']:.3f} s")
    print(f"  Time to lock:              {t['time_to_lock_s']:.2f} s")

    print("\n── HPSS Real-Time Input (iOS Simulation) ────────────────────────")
    for n in STEM_NAMES:
        print(f"  {n:8s} Meter Avg Error: {metrics['advisory_accuracy_hpss'][n]['MAE_dB']:.2f} dB")

    print("\n── Demucs Real-Time Input (Theoretical Ideal) ───────────────────")
    for n in STEM_NAMES:
        print(f"  {n:8s} Meter Avg Error: {metrics['advisory_accuracy_demucs'][n]['MAE_dB']:.2f} dB")

    # Generate and save all diagnostic plots
    print("\nGenerating plots …")
    plot_confidence_and_error(confidences, pos_errors, gt_positions, positions)
    plot_confidence_histogram(confidences)
    plot_rms_error_comparison(demucs_errors, hpss_errors)
    print("\nSaved: results_comparative_metrics.json")
    print("All outputs written.")


if __name__ == "__main__":
    run_simulation()