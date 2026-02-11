# FCPXML AutoCut

Automatic scene detection tool for **Final Cut Pro**. Analyzes camera footage using computer vision (optical flow), detects shaky/chaotic segments, and generates a clean FCPXML timeline with only the good scenes.

All processing happens **locally** — no video is uploaded anywhere.

## Features

- **Motion analysis** — detects camera shake, fast panning, chaotic movement, and transitions between shots using dense optical flow (Farneback method)
- **FCPXML 1.9 output** — generates timeline files fully compatible with Final Cut Pro 10.4.1+
- **Batch processing** — automatically processes all video files in the working directory
- **Relative paths** — FCPXML references videos with relative paths, so you can move the folder freely
- **Performance optimized** — skips frames and downscales for analysis (configurable)
- **Beautiful console UI** — progress bars, color output, summary statistics

## Supported Formats

`.mov` `.mp4` `.mkv` `.avi` `.m4v` `.mxf` `.mts` `.m2ts`

## Installation

### Option 1: Download the executable

Go to [Releases](https://github.com/vladslugin987/fcpxml-autocut/releases) and download the latest `fcpxml-autocut.exe` (Windows) or `fcpxml-autocut` (macOS).

### Option 2: Run from source

```bash
pip install -r requirements.txt
python autocut.py
```

## Usage

Place the executable (or run the script) in the same directory as your video files, then run:

```bash
# Process all videos in current directory with default settings
fcpxml-autocut

# More aggressive cutting (higher sensitivity)
fcpxml-autocut -s 0.7

# Keep more footage (lower sensitivity)
fcpxml-autocut -s 0.3

# Set minimum scene duration to 3 seconds
fcpxml-autocut --min-scene 3.0

# More accurate analysis (every 2nd frame instead of every 4th)
fcpxml-autocut --frame-skip 2

# Half resolution for analysis (faster on very large files)
fcpxml-autocut --scale 0.5

# Process a specific directory
fcpxml-autocut -d /path/to/videos

# Save FCPXML files to a different directory
fcpxml-autocut -o /path/to/output
```

### Output

For each video file (e.g. `DJI_0001.MOV`), a corresponding `DJI_0001_autocut.fcpxml` file is created in the same directory.

### Importing into Final Cut Pro

1. Run `fcpxml-autocut` in your video folder
2. Copy the entire folder (videos + `.fcpxml` files) to your Mac
3. In Final Cut Pro: **File → Import → XML...**
4. Select the `.fcpxml` file
5. The timeline will open with only the stable scenes

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `-s`, `--sensitivity` | `0.5` | Detection sensitivity (0.0–1.0). Higher = more aggressive cutting |
| `--min-scene` | `2.0` | Minimum scene duration in seconds |
| `--min-cut` | `0.5` | Minimum gap duration to actually cut (seconds) |
| `--frame-skip` | `4` | Analyze every Nth frame (higher = faster) |
| `--scale` | `0.25` | Scale factor for analysis (0.1–1.0, lower = faster) |
| `-d`, `--directory` | `.` | Input directory with video files |
| `-o`, `--output-dir` | same as input | Output directory for FCPXML files |

## How It Works

1. **Frame extraction** — reads every Nth frame from the video, downscales it
2. **Optical flow** — computes dense optical flow (Farneback) between consecutive analyzed frames
3. **Motion scoring** — combines mean flow magnitude (overall movement) with flow standard deviation (chaotic motion)
4. **Smoothing** — applies a moving average to the motion signal to avoid flickering
5. **Adaptive threshold** — computes a percentile-based threshold adjusted by the sensitivity parameter
6. **Hysteresis** — uses dual thresholds to prevent rapid toggling between good/bad states
7. **Segment filtering** — removes too-short scenes and merges segments with tiny gaps
8. **FCPXML generation** — outputs a valid FCPXML 1.9 document with rational time values

## Requirements

- Python 3.9+
- OpenCV (`opencv-python-headless`)
- NumPy
- Rich (optional, for colored console output)

## License

MIT
