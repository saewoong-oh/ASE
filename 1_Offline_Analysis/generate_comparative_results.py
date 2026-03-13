#!/usr/bin/env python3
"""
generate_comparative_results.py
-------------------------------
Runs a fully synthetic simulation of the ASE pipeline to compare the 
theoretical Demucs real-time input against the HPSS iOS simulation.

This script includes the full suite of tracking confidence metrics and plots,
alongside the raw Mean Absolute Error (MAE) comparison in decibels 
caused by frequency masking in the HPSS harmonic stream.

Outputs:
  results_comparative_metrics.json
  fig_tracking_confidence.png
  fig_confidence_histogram.png
  fig_rms_error_comparison.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# -- Reproducibility & Params -------------------------------------------------
RNG = np.random.default_rng(42)

SR = 44100
HOP = 1024
FFT_SIZE = 4096
N_REF = 4096            # reference frames ≈ 95.1 s
N_LIVE = 3800           # live frames to simulate
TEMPO_DRIFT = 1.02      # live runs 2 % faster than reference
NOISE_STD = 0.08        # additive Gaussian noise on chroma
LOCK_DELAY = int(SR / HOP) * 2

STEM_NAMES = ["drums", "bass", "vocals", "other"]

# Kalman filter hyper-parameters
CONF_THRESH = 0.15
MAX_STEP = 3.0
MIN_STEP = -2.0
PLOT_DPI = 180

# -- 1. Synthetic reference chroma --------------------------------------------
def make_reference_chroma(n: int) -> np.ndarray:
    t = np.linspace(0, 1, n)
    chroma = np.zeros((n, 12))
    for c in range(12):
        freq = 0.3 + c * 0.07
        phase = RNG.uniform(0, 2 * np.pi)
        chroma[:, c] = np.clip(0.5 + 0.5 * np.sin(2 * np.pi * freq * t + phase), 0, 1)
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return chroma / norms

# -- 2. Synthetic live chroma -------------------------------------------------
def make_live_chroma(ref: np.ndarray, n_live: int, tempo_drift: float, 
                     noise_std: float, lock_delay: int) -> Tuple[np.ndarray, np.ndarray]:
    live = np.zeros((n_live, 12))
    gt_pos = np.zeros(n_live, dtype=int)
    for i in range(n_live):
        true_pos = int(np.clip(i * tempo_drift, 0, len(ref) - 1))
        gt_pos[i] = true_pos
        if i < lock_delay:
            frame = RNG.uniform(0, 1, 12)
        else:
            frame = ref[true_pos] + RNG.normal(0, noise_std, 12)
            frame = np.clip(frame, 0, None)
        norm = np.linalg.norm(frame)
        live[i] = frame / norm if norm > 1e-12 else frame
    return live, gt_pos

# -- 3. Kalman-based position tracker -----------------------------------------
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

        tpl = np.array(self.history)
        src = self.ref_norm[lo:hi]

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
    t = np.linspace(0, 1, n)
    profiles = {}
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
    t = np.arange(len(confidences)) * HOP / SR
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(t, confidences, color="#3498DB", lw=0.9, label="Confidence")
    axes[0].axhline(0.85, color="grey", ls="--", lw=0.8, label="85 % threshold")
    axes[0].axhline(0.50, color="grey", ls=":", lw=0.8, label="50 % threshold")
    axes[0].set_ylabel("Confidence")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Tracking Confidence over Time")

    err_s = pos_errors * HOP / SR
    axes[1].plot(t, err_s, color="#E74C3C", lw=0.9)
    axes[1].set_ylabel("Position Error (s)")
    axes[1].set_title("Absolute Position Error vs. Ground Truth")

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
    print("Building reference chroma …")
    ref_chroma = make_reference_chroma(N_REF)

    print("Building live chroma stream …")
    live_chroma, gt_positions = make_live_chroma(
        ref_chroma, N_LIVE, TEMPO_DRIFT, NOISE_STD, LOCK_DELAY)

    print("Building reference RMS profiles …")
    ref_rms = make_stem_rms(N_REF)

    print("Simulating Demucs & HPSS Real-Time Streams …")
    live_demucs = {}
    for name, env in ref_rms.items():
        live_demucs[name] = np.array([
            env[int(np.clip(i * TEMPO_DRIFT, 0, N_REF - 1))] * RNG.uniform(0.85, 1.15)
            for i in range(N_LIVE)
        ])

    live_hpss = {
        "percussive": live_demucs["drums"],
        "harmonic": live_demucs["bass"] + live_demucs["vocals"] + live_demucs["other"]
    }

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

    print("Calculating dB Errors …")
    demucs_errors = {n: [] for n in STEM_NAMES}
    hpss_errors = {n: [] for n in STEM_NAMES}

    for i in range(N_LIVE):
        est_pos = positions[i]
        for n in STEM_NAMES:
            r_val = max(ref_rms[n][est_pos], 1e-6)
            ld_val = max(live_demucs[n][i], 1e-6)
            # The fixed line:
            lh_val = max(live_hpss["percussive"][i] if n == "drums" else live_hpss["harmonic"][i], 1e-6)

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

    lock_frame = None
    for i in range(len(confidences) - 20):
        if np.all(confidences[i:i + 20] >= 0.50):
            lock_frame = i
            break
    time_to_lock = float(lock_frame * frame_dur) if lock_frame else float("nan")

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

    print("\nGenerating plots …")
    plot_confidence_and_error(confidences, pos_errors, gt_positions, positions)
    plot_confidence_histogram(confidences)
    plot_rms_error_comparison(demucs_errors, hpss_errors)
    print("\nSaved: results_comparative_metrics.json")
    print("All outputs written.")

if __name__ == "__main__":
    run_simulation()