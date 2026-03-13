# ASE — Auto Sound Engineer

**Real-time mix advisory system that listens to a live performance and tells you how to adjust each stem's level to match a reference recording.**

The system works in two phases:

1. **Offline Analysis** (Python + C++): A reference song is separated into stems via [Demucs](https://github.com/facebookresearch/demucs), analyzed spectrally using a custom C++ DSP engine, and exported as a lightweight JSON tracking map.

2. **Real-Time Tracking** (iOS / Swift + C++): The iOS app listens through the microphone, identifies the current position in the song using chroma-based cross-correlation, and computes per-stem gain advisories by comparing the live frequency-band energy against the reference.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
  - [1. Python Environment](#1-python-environment)
  - [2. Build the C++ Extension (ase\_core)](#2-build-the-c-extension-ase_core)
  - [3. Verify the Installation](#3-verify-the-installation)
- [Usage](#usage)
  - [Analyze a Reference Song](#analyze-a-reference-song)
  - [Process a Quick Sample](#process-a-quick-sample)
  - [Run the Synthetic Simulation](#run-the-synthetic-simulation)
- [iOS App (ASE\_iOS)](#ios-app-ase_ios)
  - [Building in Xcode](#building-in-xcode)
  - [Adding a New Reference Song](#adding-a-new-reference-song)
  - [Running on Device](#running-on-device)
- [Standalone C++ Build (Optional)](#standalone-c-build-optional)
- [Architecture Overview](#architecture-overview)
- [File Reference](#file-reference)
  - [Offline Analysis (Python)](#offline-analysis-python)
  - [C++ DSP Engine](#c-dsp-engine)
  - [iOS App (Swift + Obj-C++ Bridge)](#ios-app-swift--obj-c-bridge)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Project Structure
ASE/
├── 1_Offline_Analysis/              # Python offline pipeline + C++ extension
│   ├── cpp/                         # C++ source for the Python extension module
│   │   ├── bindings.cpp             # pybind11 bindings exposing C++ to Python
│   │   ├── dft_engine.cpp           # Core DSP: FFT, STFT, chroma, HPSS, partial tracking
│   │   └── dft_engine.h             # Header for the DSP engine
│   ├── ref_data/                    # Output directory for generated JSON maps
│   ├── ref_songs/                   # Place your reference audio files here
│   ├── separated/                   # Demucs stem separation output
│   ├── analyze_and_export.py        # Main pipeline: stems → analysis → JSON export
│   ├── frequency_map.py             # FrequencyMap / StemProfile data structures
│   ├── reference_analyzer.py        # Per-stem spectral analysis (peaks, partials, notes)
│   ├── stem_separator.py            # Demucs wrapper + audio loading utilities
│   ├── sample_processor.py          # Quick stem-isolation utility for A/B demos
│   ├── generate_comparative_results.py  # Synthetic Demucs-vs-HPSS simulation
│   ├── export_map_ios.py            # (Legacy) alternate JSON exporter
│   ├── run_analyze.py               # (Legacy) alternate analysis entry point
│   ├── setup.py                     # Python extension build configuration
│   ├── pyproject.toml               # PEP 517 build metadata
│   └── requirements.txt             # Python dependencies
│
├── ASE_iOS/                         # Xcode project for the iOS app
│   ├── ASE_iOS.xcodeproj/
│   └── ASE_iOS/
│       ├── ASE_iOSApp.swift         # App entry point
│       ├── ContentView.swift        # Main SwiftUI view
│       ├── Audio_Layer/
│       │   ├── AudioController.swift      # Basic mic capture controller
│       │   └── AudioFileTester.swift      # Full tracking controller (file + mic)
│       ├── Cpp_Bridge/
│       │   ├── ASEWrapper.h               # Obj-C++ header for Swift interop
│       │   └── ASEWrapper.mm              # Obj-C++ implementation bridging Swift ↔ C++
│       ├── DSP_Engine/
│       │   ├── ASE_iOS-Bridging-Header.h  # Xcode bridging header
│       │   ├── dft_engine.h               # C++ DSP engine (same source as offline)
│       │   ├── dft_engine.cpp
│       │   ├── position_tracker.h         # Chroma-based song position tracker
│       │   ├── position_tracker.cpp
│       │   ├── rms_matcher.h              # Per-stem gain advisory calculator
│       │   └── rms_matcher.cpp
│       ├── Resources/                     # Reference JSON maps + test audio files
│       │   ├── Yellow.json
│       │   ├── Smooth_Operator.json
│       │   ├── map_creep_normal.json
│       │   └── *.mp3                      # Test audio files
│       └── UI/
│           └── ReferenceLibrary.swift     # Song selection UI
│
└── CMakeLists.txt                   # Standalone C++ build (independent of Xcode/Python)
Copy
---

## Requirements

| Component | Version / Notes |
|---|---|
| **Python** | 3.12.3 (tested; 3.10+ should work) |
| **C++ Compiler** | C++17 or later (GCC, Clang, or MSVC) |
| **CMake** | 3.16+ (for standalone C++ build only) |
| **Xcode** | 15.0+ (for the iOS app) |
| **iOS Deployment Target** | 16.0+ |
| **FFmpeg** | Required by Demucs for MP3/FLAC input |
| **pybind11** | Installed automatically via `pip` |

### Python Dependencies

All Python dependencies are listed in `requirements.txt`. Key packages:

- **demucs** — Neural source separation (stem splitting)
- **numpy** — Numerical arrays
- **soundfile** — Audio I/O (WAV, FLAC)
- **pydub** — Audio manipulation (used by `sample_processor.py`)
- **resampy** — High-quality resampling (optional but recommended)
- **matplotlib** — Plotting (used by the simulation script)
- **pybind11** — C++/Python binding (build dependency)
- **scipy** — Signal processing (polyphase resampling in `stem_separator.py`)

---

## Setup

### 1. Python Environment

```bash
cd 1_Offline_Analysis

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
2. Build the C++ Extension (ase_core)
The offline Python pipeline uses a C++ extension module called ase_core that provides the STFT, chroma extraction, partial tracking, and HPSS algorithms. It is built from the sources in 1_Offline_Analysis/cpp/ using pybind11.
bashCopy# From inside 1_Offline_Analysis/ with the venv activated:
pip install -e .
This runs setup.py which compiles cpp/bindings.cpp and cpp/dft_engine.cpp into a shared library (ase_core.*.so or ase_core.*.pyd) that Python can import directly.

Note: On Linux you need a C++ compiler installed (sudo apt install build-essential).
On macOS the Xcode Command Line Tools provide clang++ (xcode-select --install).
On Windows you need Visual Studio Build Tools with the "C++ Desktop" workload.

3. Verify the Installation
bashCopypython -c "import ase_core; print('ase_core OK —', dir(ase_core))"
You should see a list of exported names including STFT, ChromaExtractor, PartialTracker, compute_rms, Window, etc.

Usage
Analyze a Reference Song
This is the main workflow. It takes a reference audio file (MP3, WAV, FLAC, etc.), separates it into stems, runs spectral analysis, and exports an iOS-ready JSON tracking map.
bashCopycd 1_Offline_Analysis
source venv/bin/activate

python analyze_and_export.py path/to/song.mp3 \
    --name "Song Title" \
    --out ref_data \
    --sr 44100 \
    --fft 4096 \
    --hop 1024 \
    --peak-db -60.0
Arguments:
FlagDefaultDescriptionref(required)Path to the reference audio file--nameFilename stemDisplay name for the song (used in JSON and filenames)--outref_dataOutput directory--sr44100Target sample rate (Hz)--fft4096FFT window size (samples)--hop1024Hop size between frames (samples)--peak-db-60.0Peak detection threshold (dB)--keep-gzoffKeep intermediate gzipped files
Output:
Copyref_data/
├── map_song_title.json         # iOS tracking map (chroma + per-stem RMS)
└── listen_map_song_title.wav   # Diagnostic audio: chroma rendered as sine tones
The JSON file is what you copy into the iOS app's Resources/ folder.
Pipeline steps (printed during execution):

Load the full mix as mono audio
Separate into stems via Demucs (vocals, drums, bass, other)
Run per-stem spectral analysis (peaks, partials, notes, chroma, RMS)
Extract chroma from a drumless harmonic submix (cleaner tonal fingerprint)
Synthesize diagnostic chroma audio for verification


Tip: The first run downloads the Demucs model weights (~1.5 GB). Subsequent runs reuse the cached model.

Process a Quick Sample
Extract a time slice from a song, separate stems, reduce one stem's volume, and remix. Useful for creating A/B comparison demos.
bashCopypython sample_processor.py song.mp3 \
    --start 30 \
    --end 60 \
    --stem other \
    --reduce -15.0
This produces song_sample.wav and song_sample.mp3 with the "other" stem reduced by 15 dB.
Run the Synthetic Simulation
Run a fully synthetic benchmark comparing Demucs (theoretical ideal) vs HPSS (iOS real-time) for position tracking accuracy and stem-level metering error.
bashCopypython generate_comparative_results.py
Output:
Copyresults_comparative_metrics.json    # Tracking + advisory accuracy numbers
fig_tracking_confidence.png         # Confidence, error, and position over time
fig_confidence_histogram.png        # Distribution of per-frame confidence
fig_rms_error_comparison.png        # Per-stem dB error: Demucs vs HPSS
This script requires no audio files — it generates synthetic chroma and RMS data internally.

iOS App (ASE_iOS)
Building in Xcode

Open ASE_iOS/ASE_iOS.xcodeproj in Xcode 15+.
Select your development team under Signing & Capabilities.
The C++ DSP engine files (dft_engine.cpp, position_tracker.cpp, rms_matcher.cpp) are compiled directly by Xcode as part of the app target — no separate build step needed.
The Objective-C++ bridge (ASEWrapper.mm) is exposed to Swift via the bridging header at DSP_Engine/ASE_iOS-Bridging-Header.h.
Build and run on a physical iOS device (the microphone is not available in the Simulator).

Adding a New Reference Song

Run the offline analysis pipeline to generate a JSON map:
bashCopypython analyze_and_export.py path/to/new_song.mp3 --name "New Song"

Copy the generated ref_data/map_new_song.json into ASE_iOS/ASE_iOS/Resources/.
Add the file to the Xcode project (drag into the Resources group, check "Copy items if needed").
Register the song in UI/ReferenceLibrary.swift so it appears in the song selection list.
(Optional) Add a test audio file (MP3) to Resources for file-playback testing mode.

Running on Device
The app supports two modes:

File Playback Mode: Plays a bundled audio file through the speaker and tracks against it. Useful for testing without a live performance.
Microphone Mode: Captures live audio from the built-in microphone and tracks in real-time. This is the production use case.

Both modes display:

Current song position and progress bar
Tracking confidence meter
Per-stem gain advisories (how much to boost/cut each stem)
Per-stem live vs reference dB levels


Important: The app requests microphone permission on first launch. On iOS 17+, make sure to accept the permission prompt.


Standalone C++ Build (Optional)
The CMakeLists.txt at the repository root builds the C++ DSP engine as a standalone library, independent of both Xcode and Python. This is useful for testing, benchmarking, or integrating the engine into other C++ projects.
bashCopy# From the repository root:
mkdir build && cd build
cmake ..
make -j$(nproc)
This produces a static or shared library containing the ase namespace classes (STFT, ChromaExtractor, PositionTracker, RMSMatcher, HPSS, PartialTracker).

Note: This build does not include the pybind11 bindings or the iOS app — it's just the core DSP library.


Architecture Overview
Copy┌─────────────────────────────────────────────────────────────────┐
│                     OFFLINE (Python + C++)                       │
│                                                                 │
│  Reference Audio ──► Demucs ──► Stem Separation                 │
│       │                              │                          │
│       │                    ┌─────────┴──────────┐               │
│       │                    │  Per-Stem Analysis  │               │
│       │                    │  (STFT, Peaks,      │               │
│       │                    │   Partials, Notes)  │               │
│       │                    └─────────┬──────────┘               │
│       │                              │                          │
│       ▼                              ▼                          │
│  Harmonic Submix ──► Chroma    Per-Stem Band RMS                │
│  (no drums)          Extraction     │                           │
│       │                   │         │                           │
│       └───────────────────┴─────────┘                           │
│                    │                                            │
│                    ▼                                            │
│            JSON Tracking Map                                    │
│         (chroma + stems_rms)                                    │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     │  Bundled into iOS app
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   REAL-TIME (iOS / Swift + C++)                  │
│                                                                 │
│  Microphone ──► AVAudioEngine ──► Ring Buffer                   │
│                                       │                         │
│                                  ┌────┴────┐                    │
│                                  │  STFT   │                    │
│                                  └────┬────┘                    │
│                            ┌──────────┼──────────┐              │
│                            ▼          ▼          ▼              │
│                        Chroma    Band-Filtered  Level           │
│                       Extraction  Stem RMS     Calibration      │
│                            │          │          │              │
│                            ▼          ▼          ▼              │
│                     Position      RMS Matcher                   │
│                     Tracker       (Gain Advisory)               │
│                            │          │                         │
│                            └────┬─────┘                         │
│                                 ▼                               │
│                          SwiftUI Dashboard                      │
│                    (confidence, position, gains)                 │
└─────────────────────────────────────────────────────────────────┘

File Reference
Offline Analysis (Python)
FilePurposeanalyze_and_export.pyMain pipeline script: stems → analysis → JSON + diagnostic audiofrequency_map.pyFrequencyMap and StemProfile data classesreference_analyzer.pyPer-stem spectral analysis using the C++ enginestem_separator.pyDemucs wrapper and audio loading utilitiessample_processor.pyQuick stem-isolation utility for A/B comparison demosgenerate_comparative_results.pySynthetic simulation comparing Demucs vs HPSS accuracysetup.pyBuild configuration for the ase_core C++ extension modulecpp/bindings.cpppybind11 bindings exposing C++ classes to Pythoncpp/dft_engine.cppCore DSP implementation (shared with iOS)cpp/dft_engine.hCore DSP header (shared with iOS)
C++ DSP Engine
ClassFile(s)Purposease::STFTdft_engine.h/cppWindowed Short-Time Fourier Transform with peak detectionase::ChromaExtractordft_engine.h/cppSpectrum → 12-bin chromagram mappingase::PartialTrackerdft_engine.h/cppCross-frame sinusoidal partial tracking + note extractionase::HPSSdft_engine.h/cppHarmonic-percussive separation via median filteringase::PositionTrackerposition_tracker.h/cppChroma-based real-time song position estimationase::RMSMatcherrms_matcher.h/cppPer-stem gain advisory from reference vs live RMS
iOS App (Swift + Obj-C++ Bridge)
FilePurposeASEWrapper.h/mmObjective-C++ bridge: exposes C++ engine to SwiftAudioFileTester.swiftMain tracking controller: loads JSON, manages AVAudioEngine, feeds C++AudioController.swiftBasic microphone capture controllerContentView.swiftMain SwiftUI dashboardReferenceLibrary.swiftSong selection UIASE_iOSApp.swiftSwiftUI app entry pointASE_iOS-Bridging-Header.hXcode bridging header (imports ASEWrapper.h)

Troubleshooting
ModuleNotFoundError: No module named 'ase_core'
The C++ extension hasn't been built. Run:
bashCopycd 1_Offline_Analysis
pip install -e .
Make sure you have a C++ compiler installed and that you're in the activated virtual environment.
Demucs fails with FileNotFoundError or codec errors
Demucs requires FFmpeg for non-WAV input formats. Install it:
bashCopy# Ubuntu/Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
resampy not installed warning
The pipeline falls back to the source sample rate if resampy is missing. Install it:
bashCopypip install resampy
Xcode build fails with C++ errors

Make sure all .cpp files in DSP_Engine/ are added to the app target's Compile Sources build phase.
Verify the bridging header path is correct in Build Settings → Swift Compiler → Objective-C Bridging Header: ASE_iOS/Cpp_Bridge/ASE_iOS-Bridging-Header.h (or the path relative to your project root).
The C++ Language Dialect should be set to C++17 or later in Build Settings.

iOS app shows no tracking / confidence stays at 0

Verify the JSON file is included in the app bundle (check Build Phases → Copy Bundle Resources).
Check the Xcode console for [ASEWrapper] and [HOP] log messages — these confirm the C++ engine is receiving audio.
If using microphone mode, ensure microphone permission is granted in Settings → Privacy → Microphone.

Simulation script (generate_comparative_results.py) produces NaN for time-to-lock
This means the tracker never achieved 20 consecutive frames above 50% confidence. This can happen if the simulation parameters are too aggressive. The default parameters should work out of the box.

License
This project is for educational and research purposes.