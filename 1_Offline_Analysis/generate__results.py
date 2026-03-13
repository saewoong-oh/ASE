#!/usr/bin/env python3
"""
generate_results.py
-------------------
Runs a fully synthetic simulation of the ASE pipeline and produces
quantitative metrics suitable for the Results section of the report.

No real audio file is required. The script simulates:
  1. A synthetic reference chroma sequence (12-dim, 4096 frames ≈ 95 s at
     hop=1024, sr=44100).
  2. A synthetic "live" chroma sequence with additive Gaussian noise,
     a 2-second startup delay before lock, and a mild tempo drift (+2 %).
  3. The Kalman-based position filter as implemented in position_tracker.cpp.
  4. The RMS gain computation as implemented in rms_matcher.cpp.

Outputs
-------
  results_metrics.json         – all numeric results (paste into LaTeX)
  fig_tracking_confidence.png  – confidence and position error vs. time
  fig_gain_adjustment.png      – per-stem gain over time
  fig_rms_comparison.png       – reference vs live RMS per stem
  fig_confidence_histogram.png – distribution of per-frame confidence

Run: python generate_results.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# -- Reproducibility ----------------------------------------------------------
RNG = np.random.default_rng(42)

# -- Simulation parameters ----------------------------------------------------
SR = 44100
HOP = 1024
FFT_SIZE = 4096
N_REF = 4096            # reference frames ≈ 95.1 s
N_LIVE = 3800           # live frames to simulate
TEMPO_DRIFT = 1.02      # live runs 2 % faster than reference
NOISE_STD = 0.08        # additive Gaussian noise on chroma
LOCK_DELAY = int(SR / HOP) * 2   # 2-second delay before stable lock

STEM_NAMES = ["drums", "bass", "vocals", "other"]
ANCHOR = "drums"

# Kalman filter hyper-parameters (matching position_tracker.cpp)
CONF_THRESH = 0.15
EMA_ALPHA = 0.75        # confidence EMA
GAIN_ALPHA = 0.65       # Kalman gain blend weight
MAX_STEP = 3.0
MIN_STEP = -2.0

# RMS matcher (matching rms_matcher.cpp)
GAIN_EMA_ALPHA = 0.65
MAX_GAIN_DB = 12.0
MIN_GAIN_DB = -12.0
CALIB_FRAMES = 48


# -- 1. Synthetic reference chroma --------------------------------------------
def make_reference_chroma(n: int) -> np.ndarray:
    """
    Build a smooth, musically plausible reference chroma by superimposing
    several slowly varying sinusoids across the 12 chroma bins.
    """
    t = np.linspace(0, 1, n)
    chroma = np.zeros((n, 12))
    # Assign different slow oscillation rates to different pitch classes
    for c in range(12):
        freq = 0.3 + c * 0.07          # cycle frequency in 1/N units
        phase = RNG.uniform(0, 2 * np.pi)
        chroma[:, c] = np.clip(0.5 + 0.5 * np.sin(2 * np.pi * freq * t + phase), 0, 1)
    # L2 normalise each frame
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return chroma / norms


# -- 2. Synthetic live chroma -------------------------------------------------
def make_live_chroma(ref: np.ndarray,
                     n_live: int,
                     tempo_drift: float,
                     noise_std: float,
                     lock_delay: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a live chroma stream by sampling the reference with tempo drift
    and adding noise. Returns (live_chroma, true_ref_positions).
    """
    live = np.zeros((n_live, 12))
    gt_pos = np.zeros(n_live, dtype=int)

    for i in range(n_live):
        true_pos = int(np.clip(i * tempo_drift, 0, len(ref) - 1))
        gt_pos[i] = true_pos
        if i < lock_delay:
            # Before lock: random noise only
            frame = RNG.uniform(0, 1, 12)
        else:
            frame = ref[true_pos] + RNG.normal(0, noise_std, 12)
            frame = np.clip(frame, 0, None)
        norm = np.linalg.norm(frame)
        live[i] = frame / norm if norm > 1e-12 else frame

    return live, gt_pos


# -- 3. Kalman-based position tracker (Python mirror of position_tracker.cpp) -
@dataclass
class PositionTracker:
    ref_norm: np.ndarray
    history_len: int = 300
    radius: int = 100
    sigma: float = 10.0
    conf_thresh: float = 0.15
    max_tempo: float = 1.10
    min_tempo: float = 0.90

    pos: float = 0.0
    tempo: float = 1.0
    conf_ema: float = 0.0
    confidence: float = 0.0
    locked: bool = False
    lock_count: int = 0
    history: list = field(default_factory=list)
    _j: int = 0

    def process(self, live_frame: np.ndarray) -> Tuple[int, float]:
        ln = np.linalg.norm(live_frame)
        if ln < 1e-3:
            self.pos = np.clip(self.pos + self.tempo, 0, len(self.ref_norm) - 1)
            self.confidence *= 0.95
            self._j += 1
            return int(round(self.pos)), self.confidence

        lc = live_frame / ln
        self.history.append(lc)
        if len(self.history) > self.history_len:
            self.history.pop(0)

        T = len(self.history)
        N = len(self.ref_norm)
        exp = np.clip(self.pos + self.tempo, 0, N - 1)

        # Search window
        if T < 15:
            r, sig = 2, self.sigma * 0.5
        elif not self.locked:
            r, sig = min(1500, N // 2), self.sigma * 20.0
        else:
            r, sig = self.radius, self.sigma

        lo = max(0, int(exp) - T + 1 - r)
        hi = min(N, int(exp) + r + 1)
        S = hi - lo

        if S < T:
            self.pos = exp
            self._j += 1
            return int(round(self.pos)), self.confidence

        # Cosine similarity of history against reference window
        tpl = np.array(self.history)          # (T, 12)
        src = self.ref_norm[lo:hi]            # (S, 12)

        # Sliding-window dot product (simplified cross-correlation)
        best_score = -1e30
        best_sim = 0.0
        best_pos = exp
        result_len = S - T + 1

        for d in range(result_len):
            sim = float(np.sum(tpl * src[d:d + T]) / T)
            ref_p = float(lo + d + T - 1)
            offset = ref_p - exp
            cont = np.exp(-0.5 * (offset / sig) ** 2)
            score = sim * cont
            if score > best_score:
                best_score = score
                best_sim = sim
                best_pos = ref_p

        self.conf_ema = 0.75 * self.conf_ema + 0.25 * best_sim
        self.confidence = np.clip(self.conf_ema, 0.0, 1.0)

        if best_sim >= self.conf_thresh:
            innovation = best_pos - exp
            if not self.locked and T >= 15:
                cl = innovation
            else:
                cl = np.clip(innovation, MIN_STEP, MAX_STEP)

            gain = min(0.85, best_sim)
            if best_sim < 0.6:
                gain *= best_sim / 0.6

            self.pos = np.clip(exp + gain * cl, 0, N - 1)
            self.lock_count = min(self.lock_count + 1, 60)
        else:
            self.pos = exp
            self.lock_count = max(0, self.lock_count - 2)

        self.locked = self.lock_count >= 5
        self._j += 1
        return int(round(self.pos)), self.confidence


# -- 4. Synthetic stem RMS profiles -------------------------------------------
def make_stem_rms(n: int) -> Dict[str, np.ndarray]:
    """Smooth per-stem RMS envelopes for the reference."""
    t = np.linspace(0, 1, n)
    profiles: Dict[str, np.ndarray] = {}
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


# -- 5. RMS matcher (Python mirror of rms_matcher.cpp) ------------------------
class RMSMatcher:
    def __init__(self, names: List[str], anchor: str):
        self.names = names
        self.anchor = anchor
        self.gains = {n: 1.0 for n in names}
        self.max_g = 10 ** (MAX_GAIN_DB / 20)
        self.min_g = 10 ** (MIN_GAIN_DB / 20)

    def compute(self,
                ref_rms: Dict[str, float],
                live_rms: Dict[str, float],
                confidence: float) -> Dict[str, float]:
        ref_a = max(ref_rms.get(self.anchor, 1e-10), 1e-10)
        live_a = max(live_rms.get(self.anchor, 1e-10), 1e-10)

        new_gains: Dict[str, float] = {}
        for name in self.names:
            r = ref_rms.get(name, 0.0)
            l = live_rms.get(name, 0.0)

            if name == self.anchor:
                target = 1.0
            else:
                ref_ratio = r / ref_a
                live_ratio = max(l / live_a, 1e-10)
                target = ref_ratio / live_ratio

            target = np.clip(target, self.min_g, self.max_g)
            target = confidence * target + (1 - confidence) * 1.0
            self.gains[name] = (GAIN_EMA_ALPHA * self.gains[name]
                                + (1 - GAIN_EMA_ALPHA) * target)
            new_gains[name] = self.gains[name]

        self.gains[self.anchor] = 1.0
        new_gains[self.anchor] = 1.0
        return new_gains


# -- 6. Run simulation --------------------------------------------------------
def run_simulation():
    print("Building reference chroma …")
    ref_chroma = make_reference_chroma(N_REF)

    print("Building live chroma stream …")
    live_chroma, gt_positions = make_live_chroma(
        ref_chroma, N_LIVE, TEMPO_DRIFT, NOISE_STD, LOCK_DELAY)

    print("Building reference RMS profiles …")
    ref_rms_profiles = make_stem_rms(N_REF)

    # Live RMS: same profile shifted by tempo drift with noise
    live_rms_profiles: Dict[str, np.ndarray] = {}
    for name, env in ref_rms_profiles.items():
        live_env = np.array([
            env[int(np.clip(i * TEMPO_DRIFT, 0, N_REF - 1))]
            * RNG.uniform(0.6, 1.4)
            for i in range(N_LIVE)
        ])
        live_rms_profiles[name] = live_env

    # Trackers
    tracker = PositionTracker(ref_norm=ref_chroma)
    matcher = RMSMatcher(STEM_NAMES, ANCHOR)

    # Storage
    positions = np.zeros(N_LIVE, dtype=int)
    confidences = np.zeros(N_LIVE)
    pos_errors = np.zeros(N_LIVE)
    all_gains: Dict[str, List[float]] = {n: [] for n in STEM_NAMES}
    live_rms_log: Dict[str, List[float]] = {n: [] for n in STEM_NAMES}
    ref_rms_log: Dict[str, List[float]] = {n: [] for n in STEM_NAMES}

    # Level calibration
    level_scalar = 1.0
    level_calibrated = False
    calib_live_sum = 0.0
    calib_ref_sum = 0.0
    calib_count = 0

    print("Running frame-by-frame simulation …")
    for i in range(N_LIVE):
        pos, conf = tracker.process(live_chroma[i])
        positions[i] = pos
        confidences[i] = conf
        pos_errors[i] = abs(pos - gt_positions[i])

        # Live RMS at frame i
        live_rms = {n: live_rms_profiles[n][i] for n in STEM_NAMES}
        ref_rms = {n: ref_rms_profiles[n][pos] for n in STEM_NAMES}

        # Level calibration
        if not level_calibrated and conf > 0.35:
            la = live_rms.get(ANCHOR, 0.0)
            ra = ref_rms.get(ANCHOR, 0.0)
            if la > 1e-6 and ra > 1e-6:
                calib_live_sum += la
                calib_ref_sum += ra
                calib_count += 1
                if calib_count >= CALIB_FRAMES:
                    level_scalar = calib_ref_sum / calib_live_sum
                    level_calibrated = True

        live_rms_scaled = {n: live_rms[n] * level_scalar for n in STEM_NAMES}

        gains = matcher.compute(ref_rms, live_rms_scaled, conf)

        for n in STEM_NAMES:
            all_gains[n].append(gains[n])
            live_rms_log[n].append(live_rms_scaled[n])
            ref_rms_log[n].append(ref_rms[n])

        if i % 500 == 0:
            print(f"  frame {i:4d}/{N_LIVE}  pos={pos:4d}  conf={conf:.3f}  err={pos_errors[i]:.1f}")

    return (positions, confidences, pos_errors,
            all_gains, live_rms_log, ref_rms_log, gt_positions)


# -- 7. Compute metrics -------------------------------------------------------
def compute_metrics(positions, confidences, pos_errors, all_gains,
                    live_rms_log, ref_rms_log):
    frame_dur = HOP / SR

    # Confidence
    mean_conf = float(np.mean(confidences))
    above_85 = float(np.mean(confidences >= 0.85) * 100)
    above_50 = float(np.mean(confidences >= 0.50) * 100)
    locked_frac = float(np.mean(confidences >= CONF_THRESH) * 100)

    # Position error (frames → seconds)
    mean_err_s = float(np.mean(pos_errors) * frame_dur)
    median_err_s = float(np.median(pos_errors) * frame_dur)
    p90_err_s = float(np.percentile(pos_errors, 90) * frame_dur)

    # Time to lock: first frame where conf >= 0.50 stays above for 20 frames
    lock_frame = None
    for i in range(len(confidences) - 20):
        if np.all(confidences[i:i + 20] >= 0.50):
            lock_frame = i
            break
    time_to_lock = float(lock_frame * frame_dur) if lock_frame else float("nan")

    # Gain statistics per stem
    gain_stats: Dict[str, Dict] = {}
    for name in STEM_NAMES:
        g = np.array(all_gains[name])
        g_db = 20 * np.log10(np.clip(g, 1e-6, None))
        gain_stats[name] = {
            "mean_gain_db": float(np.mean(g_db)),
            "std_gain_db": float(np.std(g_db)),
            "max_gain_db": float(np.max(g_db)),
            "min_gain_db": float(np.min(g_db)),
        }

    # RMS residual per stem
    rms_residual: Dict[str, float] = {}
    for name in STEM_NAMES:
        l = np.array(live_rms_log[name])
        r = np.array(ref_rms_log[name])
        g = np.array(all_gains[name])
        corrected = l * g
        residual = np.sqrt(np.mean((corrected - r) ** 2))
        rms_residual[name] = float(residual)

    metrics = {
        "simulation_parameters": {
            "n_reference_frames": N_REF,
            "n_live_frames": N_LIVE,
            "reference_duration_s": float(N_REF * HOP / SR),
            "live_duration_s": float(N_LIVE * HOP / SR),
            "tempo_drift_percent": (TEMPO_DRIFT - 1.0) * 100,
            "noise_std": NOISE_STD,
            "lock_delay_frames": LOCK_DELAY,
            "fft_size": FFT_SIZE,
            "hop_size": HOP,
            "sample_rate": SR,
        },
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
        "gain_adjustment": gain_stats,
        "rms_residual_after_correction": rms_residual,
    }
    return metrics


# -- 8. Plots -----------------------------------------------------------------
PLOT_DPI = 180
COLOURS = {"drums": "#E74C3C", "bass": "#3498DB",
           "vocals": "#2ECC71", "other": "#F39C12"}


def plot_confidence_and_error(confidences, pos_errors, gt_positions, positions):
    t = np.arange(len(confidences)) * HOP / SR
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    # Confidence
    axes[0].plot(t, confidences, color="#3498DB", lw=0.9, label="Confidence")
    axes[0].axhline(0.85, color="grey", ls="--", lw=0.8, label="85 % threshold")
    axes[0].axhline(0.50, color="grey", ls=":", lw=0.8, label="50 % threshold")
    axes[0].set_ylabel("Confidence")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Tracking Confidence over Time")

    # Position error
    err_s = pos_errors * HOP / SR
    axes[1].plot(t, err_s, color="#E74C3C", lw=0.9)
    axes[1].set_ylabel("Position Error (s)")
    axes[1].set_title("Absolute Position Error vs. Ground Truth")

    # Estimated vs ground truth position
    axes[2].plot(t, gt_positions * HOP / SR, color="black",
                 lw=0.8, ls="--", label="Ground Truth")
    axes[2].plot(t, positions * HOP / SR, color="#27AE60",
                 lw=0.9, alpha=0.85, label="Estimated")
    axes[2].set_ylabel("Song Position (s)")
    axes[2].set_xlabel("Wall-Clock Time (s)")
    axes[2].set_title("Estimated Position vs. Ground Truth")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("fig_tracking_confidence.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: fig_tracking_confidence.png")


def plot_gains(all_gains):
    t = np.arange(N_LIVE) * HOP / SR
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in STEM_NAMES:
        g = np.array(all_gains[name])
        g_db = 20 * np.log10(np.clip(g, 1e-6, None))
        ax.plot(t, g_db, lw=0.9, label=name.capitalize(), color=COLOURS[name])
    ax.axhline(0, color="black", ls="--", lw=0.7)
    ax.set_xlabel("Wall-Clock Time (s)")
    ax.set_ylabel("Stem Gain (dB)")
    ax.set_title("Per-Stem Gain Adjustment over Time")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("fig_gain_adjustment.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: fig_gain_adjustment.png")


def plot_rms_comparison(live_rms_log, ref_rms_log, all_gains):
    t = np.arange(N_LIVE) * HOP / SR
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes = axes.flatten()

    for idx, name in enumerate(STEM_NAMES):
        l = np.array(live_rms_log[name])
        r = np.array(ref_rms_log[name])
        g = np.array(all_gains[name])
        corrected = l * g

        axes[idx].plot(t, r, lw=0.9, ls="--", color="black", label="Reference RMS")
        axes[idx].plot(t, l, lw=0.8, alpha=0.6,
                       color=COLOURS[name], label="Live RMS (pre-gain)")
        axes[idx].plot(t, corrected, lw=0.9,
                       color=COLOURS[name], label="Live RMS (post-gain)")
        axes[idx].set_title(name.capitalize(), fontsize=9)
        axes[idx].set_ylabel("RMS Amplitude")
        if idx >= 2:
            axes[idx].set_xlabel("Wall-Clock Time (s)")
        axes[idx].legend(fontsize=7)

    plt.suptitle("Reference vs. Live RMS Before and After Gain Correction", fontsize=10)
    plt.tight_layout()
    plt.savefig("fig_rms_comparison.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: fig_rms_comparison.png")


def plot_confidence_histogram(confidences):
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


# -- 9. Main ------------------------------------------------------------------
def main():
    (positions, confidences, pos_errors,
     all_gains, live_rms_log, ref_rms_log, gt_positions) = run_simulation()

    metrics = compute_metrics(positions, confidences, pos_errors,
                              all_gains, live_rms_log, ref_rms_log)

    with open("results_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved: results_metrics.json")
    print("\n── Key Metrics ──────────────────────────────────────────────────")
    t = metrics["tracking"]
    print(f"  Mean confidence:           {t['mean_confidence']:.3f}")
    print(f"  Frames ≥ 85 % confidence:  {t['frames_above_85pct_conf']:.1f} %")
    print(f"  Frames ≥ 50 % confidence:  {t['frames_above_50pct_conf']:.1f} %")
    print(f"  Mean position error:       {t['mean_position_error_s']:.3f} s")
    print(f"  Median position error:     {t['median_position_error_s']:.3f} s")
    print(f"  90th-pct position error:   {t['p90_position_error_s']:.3f} s")
    print(f"  Time to lock:              {t['time_to_lock_s']:.2f} s")
    print("\n── Gain Statistics ──────────────────────────────────────────────")
    for name in STEM_NAMES:
        gs = metrics["gain_adjustment"][name]
        print(f"  {name:8s}  mean={gs['mean_gain_db']:+.2f} dB  "
              f"std={gs['std_gain_db']:.2f} dB  "
              f"[{gs['min_gain_db']:+.2f}, {gs['max_gain_db']:+.2f}] dB")
    print("\n── RMS Residual After Correction ────────────────────────────────")
    for name in STEM_NAMES:
        print(f"  {name:8s}  {metrics['rms_residual_after_correction'][name]:.4f}")

    print("\nGenerating plots …")
    plot_confidence_and_error(confidences, pos_errors, gt_positions, positions)
    plot_gains(all_gains)
    plot_rms_comparison(live_rms_log, ref_rms_log, all_gains)
    plot_confidence_histogram(confidences)
    print("\nAll outputs written. Run LaTeX after copying PNGs to your source directory.")


if __name__ == "__main__":
    main()