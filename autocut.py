#!/usr/bin/env python3
"""
FCPXML AutoCut - Automatic Scene Detection for Final Cut Pro

Analyzes video files using optical flow to detect camera shake, fast movement,
and chaotic motion. Generates FCPXML files with only stable scenes, ready for
import into Final Cut Pro.
"""

import os
import sys
import time
import argparse
import math
import html
import urllib.parse
from pathlib import Path
from typing import List, Tuple, Optional
from fractions import Fraction

import cv2
import numpy as np

# In frozen exe (PyInstaller) skip Rich to avoid missing rich._unicode_data.* at runtime
RICH_AVAILABLE = False
if not getattr(sys, "frozen", False):
    try:
        from rich.console import Console
        from rich.progress import (
            Progress,
            BarColumn,
            TextColumn,
            TimeRemainingColumn,
            TimeElapsedColumn,
            TaskProgressColumn,
            SpinnerColumn,
        )
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich import box
        from rich.markup import escape

        RICH_AVAILABLE = True
    except ImportError:
        pass

# ============================================================================
# Constants
# ============================================================================

VERSION = "1.0.0"
APP_NAME = "FCPXML AutoCut"
VIDEO_EXTENSIONS = {".mov", ".mp4", ".mkv", ".avi", ".m4v", ".mxf", ".mts", ".m2ts"}
FCPXML_VERSION = "1.9"

# Standard frame rates: (approx_fps, numerator, denominator)
# Real fps = numerator / denominator
# Frame duration = denominator / numerator seconds
STANDARD_FRAME_RATES = [
    (23.976, 24000, 1001),
    (24.0, 24, 1),
    (25.0, 25, 1),
    (29.97, 30000, 1001),
    (30.0, 30, 1),
    (47.952, 48000, 1001),
    (48.0, 48, 1),
    (50.0, 50, 1),
    (59.94, 60000, 1001),
    (60.0, 60, 1),
    (119.88, 120000, 1001),
    (120.0, 120, 1),
]

# Defaults
DEFAULT_FRAME_SKIP = 4
DEFAULT_SCALE_FACTOR = 0.25
DEFAULT_SENSITIVITY = 0.5
DEFAULT_MIN_SCENE_DURATION = 2.0
DEFAULT_MIN_CUT_DURATION = 0.5
DEFAULT_TRIM_START = 0.8  # seconds to trim from the start of each scene


# ============================================================================
# Utility Functions
# ============================================================================


def get_standard_fps(fps: float) -> Tuple[int, int]:
    """Match detected fps to nearest standard frame rate.
    Returns (fps_numerator, fps_denominator)."""
    best_match = None
    best_diff = float("inf")
    for approx, num, den in STANDARD_FRAME_RATES:
        diff = abs(fps - approx)
        if diff < best_diff:
            best_diff = diff
            best_match = (num, den)
    if best_diff > 1.0:
        fps_rounded = round(fps)
        return (fps_rounded, 1)
    return best_match


def frames_to_time(num_frames: int, fps_num: int, fps_den: int) -> str:
    """Convert frame count to FCPXML rational time string.
    Time = num_frames * fps_den / fps_num seconds."""
    if num_frames == 0:
        return "0s"
    numerator = num_frames * fps_den
    return f"{numerator}/{fps_num}s"


def get_fps_label(fps: float) -> str:
    """Get FPS label for FCPXML format naming."""
    mapping = [
        (23.976, "2398"),
        (24.0, "24"),
        (25.0, "25"),
        (29.97, "2997"),
        (30.0, "30"),
        (48.0, "48"),
        (50.0, "50"),
        (59.94, "5994"),
        (60.0, "60"),
        (120.0, "120"),
    ]
    for ref_fps, label in mapping:
        if abs(fps - ref_fps) < 0.1:
            return label
    return str(int(round(fps)))


def format_duration(seconds: float) -> str:
    """Format seconds to MM:SS or HH:MM:SS."""
    if seconds < 0:
        return "00:00"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def discover_videos(directory: Path) -> List[Path]:
    """Find all video files in the given directory (non-recursive)."""
    videos = []
    try:
        for f in sorted(directory.iterdir()):
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(f)
    except PermissionError:
        pass
    return videos


# ============================================================================
# Video Info
# ============================================================================


def get_video_info(video_path: Path) -> dict:
    """Quickly extract video metadata without full analysis."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path.name}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or total_frames <= 0:
        raise ValueError(f"Invalid video metadata: {video_path.name}")

    fps_num, fps_den = get_standard_fps(fps)
    duration = total_frames / fps

    return {
        "path": video_path,
        "fps": fps,
        "fps_num": fps_num,
        "fps_den": fps_den,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration,
    }


# ============================================================================
# Video Analyzer
# ============================================================================


class VideoAnalyzer:
    """Analyzes video motion to detect stable and unstable segments."""

    def __init__(
        self,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        sensitivity: float = DEFAULT_SENSITIVITY,
        min_scene_duration: float = DEFAULT_MIN_SCENE_DURATION,
        min_cut_duration: float = DEFAULT_MIN_CUT_DURATION,
        trim_start: float = DEFAULT_TRIM_START,
    ):
        self.frame_skip = max(1, frame_skip)
        self.scale_factor = max(0.1, min(1.0, scale_factor))
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.min_scene_duration = max(0.5, min_scene_duration)
        self.min_cut_duration = max(0.1, min_cut_duration)
        self.trim_start = max(0.0, trim_start)

    def analyze(self, video_path: Path, video_info: dict, progress_callback=None) -> dict:
        """Analyze video and return detected good segments.

        Args:
            video_path: Path to the video file
            video_info: Pre-extracted video metadata
            progress_callback: Callable(current_frame, total_analyzed_frames)

        Returns:
            dict with 'segments' key containing list of (start_frame, end_frame) tuples
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path.name}")

        fps = video_info["fps"]
        total_frames = video_info["total_frames"]
        frames_to_analyze = total_frames // self.frame_skip

        # Collect motion scores
        motion_scores = []
        prev_gray = None
        frame_idx = 0
        analyzed_count = 0

        while True:
            ret = cap.grab()
            if not ret:
                break

            if frame_idx % self.frame_skip != 0:
                frame_idx += 1
                continue

            ret, frame = cap.retrieve()
            if not ret:
                frame_idx += 1
                continue

            # Downscale for speed
            if self.scale_factor < 1.0:
                new_w = max(64, int(frame.shape[1] * self.scale_factor))
                new_h = max(64, int(frame.shape[0] * self.scale_factor))
                small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                small = frame

            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Compute dense optical flow (Farneback method)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                # Mean magnitude: overall camera movement speed
                avg_mag = np.mean(mag)
                # Std of flow vectors: captures chaotic / non-uniform motion
                # Smooth pan → all pixels move the same direction → low std
                # Shake/jerk → pixels move in different directions → high std
                flow_std = np.std(flow[..., 0]) + np.std(flow[..., 1])
                # Peak motion: sudden spikes indicate jerky transitions
                peak_mag = np.percentile(mag, 95)

                # Combined score: heavily penalize chaos and jerkiness,
                # smooth slow pans (high avg_mag but low flow_std) get low score
                score = avg_mag * 0.2 + flow_std * 0.55 + peak_mag * 0.25

                motion_scores.append(score)
            else:
                motion_scores.append(0.0)

            prev_gray = gray
            analyzed_count += 1
            frame_idx += 1

            if progress_callback and frames_to_analyze > 0:
                progress_callback(analyzed_count, frames_to_analyze)

        cap.release()

        if len(motion_scores) < 3:
            return {"segments": [(0, total_frames)]}

        # Classify into good/bad segments
        segments = self._classify_segments(motion_scores, fps, total_frames)

        return {"segments": segments}

    def _classify_segments(
        self, motion_scores: List[float], fps: float, total_frames: int
    ) -> List[Tuple[int, int]]:
        """Classify motion scores into good/bad segments using adaptive thresholding."""
        scores = np.array(motion_scores, dtype=np.float64)

        # Smooth the signal to avoid flickering between good/bad
        window_size = max(3, int((fps / self.frame_skip) * 0.7))
        if window_size % 2 == 0:
            window_size += 1
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(scores, kernel, mode="same")

        # Adaptive threshold based on sensitivity
        # sensitivity 0.0 → percentile ~85 (keep most footage)
        # sensitivity 0.5 → percentile ~55
        # sensitivity 1.0 → percentile ~25 (aggressive cutting)
        percentile_value = 85.0 - self.sensitivity * 60.0
        threshold = np.percentile(smoothed, percentile_value)

        # Hysteresis: two thresholds to avoid rapid toggling
        high_thresh = threshold * 1.15
        low_thresh = threshold * 0.85

        # Classify frames with hysteresis
        is_good = np.ones(len(smoothed), dtype=bool)
        currently_good = True

        for i in range(len(smoothed)):
            if currently_good:
                if smoothed[i] > high_thresh:
                    currently_good = False
                    is_good[i] = False
            else:
                if smoothed[i] < low_thresh:
                    currently_good = True
                    is_good[i] = True
                else:
                    is_good[i] = False

        # Convert sample indices to frame ranges
        segments = []
        in_good = False
        seg_start = 0

        for i in range(len(is_good)):
            if is_good[i] and not in_good:
                seg_start = i * self.frame_skip
                in_good = True
            elif not is_good[i] and in_good:
                seg_end = min(i * self.frame_skip, total_frames)
                segments.append((seg_start, seg_end))
                in_good = False

        if in_good:
            segments.append((seg_start, min(len(is_good) * self.frame_skip, total_frames)))

        # Filter out too-short scenes
        min_scene_frames = int(self.min_scene_duration * fps)
        segments = [(s, e) for s, e in segments if (e - s) >= min_scene_frames]

        # Merge segments with very short gaps between them
        min_cut_frames = int(self.min_cut_duration * fps)
        segments = self._merge_close_segments(segments, min_cut_frames)

        # Trim the start of each scene (camera settling / focus adjustment)
        if self.trim_start > 0:
            trim_frames = int(self.trim_start * fps)
            trimmed = []
            for s, e in segments:
                new_start = s + trim_frames
                if new_start < e and (e - new_start) >= min_scene_frames:
                    trimmed.append((new_start, e))
            segments = trimmed

        # Safety: if everything was cut, keep the most stable portion
        if not segments:
            segments = self._find_best_segment(smoothed, fps, total_frames)

        return segments

    @staticmethod
    def _merge_close_segments(
        segments: List[Tuple[int, int]], min_gap: int
    ) -> List[Tuple[int, int]]:
        """Merge segments that are closer together than min_gap frames."""
        if len(segments) <= 1:
            return segments

        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end < min_gap:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        return merged

    def _find_best_segment(
        self, smoothed: np.ndarray, fps: float, total_frames: int
    ) -> List[Tuple[int, int]]:
        """Find the most stable region when everything else was cut."""
        window = max(1, int(fps / self.frame_skip * 3))  # 3-second window
        if window >= len(smoothed):
            return [(0, total_frames)]

        best_start = 0
        best_score = float("inf")

        for i in range(len(smoothed) - window):
            avg = np.mean(smoothed[i : i + window])
            if avg < best_score:
                best_score = avg
                best_start = i

        seg_start = best_start * self.frame_skip
        seg_end = min((best_start + window) * self.frame_skip, total_frames)
        return [(seg_start, seg_end)]


# ============================================================================
# FCPXML Generator
# ============================================================================


class FCPXMLGenerator:
    """Generates FCPXML 1.9 files fully compatible with Final Cut Pro."""

    @staticmethod
    def generate(video_info: dict, segments: List[Tuple[int, int]]) -> str:
        """Generate a complete FCPXML document string."""
        fps_num = video_info["fps_num"]
        fps_den = video_info["fps_den"]
        width = video_info["width"]
        height = video_info["height"]
        total_frames = video_info["total_frames"]
        fps = video_info["fps"]
        video_path: Path = video_info["path"]

        def ft(frames: int) -> str:
            """Frames → FCPXML rational time."""
            if frames == 0:
                return "0s"
            n = frames * fps_den
            return f"{n}/{fps_num}s"

        # Format element attributes
        fps_label = get_fps_label(fps)
        format_name = f"FFVideoFormat{width}x{height}p{fps_label}"
        frame_duration = f"{fps_den}/{fps_num}s"

        # Asset attributes
        asset_name = html.escape(video_path.stem)
        media_src = "./" + urllib.parse.quote(video_path.name, safe="")
        total_duration = ft(total_frames)

        # Build asset-clip elements
        clips_lines = []
        offset_frames = 0
        for i, (start_frame, end_frame) in enumerate(segments):
            dur_frames = end_frame - start_frame
            clip = (
                f'                        <asset-clip ref="r2" '
                f'offset="{ft(offset_frames)}" '
                f'name="Scene {i + 1}" '
                f'start="{ft(start_frame)}" '
                f'duration="{ft(dur_frames)}" '
                f'format="r1" '
                f'tcFormat="NDF"/>'
            )
            clips_lines.append(clip)
            offset_frames += dur_frames

        clips_xml = "\n".join(clips_lines)
        seq_duration = ft(offset_frames)

        # Assemble the full FCPXML document
        fcpxml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<!DOCTYPE fcpxml>\n"
            "\n"
            f'<fcpxml version="{FCPXML_VERSION}">\n'
            "    <resources>\n"
            f'        <format id="r1" name="{format_name}" '
            f'frameDuration="{frame_duration}" '
            f'width="{width}" height="{height}"/>\n'
            f'        <asset id="r2" name="{asset_name}" '
            f'start="0s" duration="{total_duration}" '
            f'hasVideo="1" format="r1" hasAudio="1" '
            f'audioSources="1" audioChannels="2" audioRate="48000">\n'
            f'            <media-rep kind="original-media" src="{media_src}"/>\n'
            "        </asset>\n"
            "    </resources>\n"
            "    <library>\n"
            '        <event name="AutoCut">\n'
            f'            <project name="{asset_name} - AutoCut">\n'
            f'                <sequence format="r1" duration="{seq_duration}" '
            f'tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">\n'
            "                    <spine>\n"
            f"{clips_xml}\n"
            "                    </spine>\n"
            "                </sequence>\n"
            "            </project>\n"
            "        </event>\n"
            "    </library>\n"
            "</fcpxml>\n"
        )

        return fcpxml

    @staticmethod
    def save(fcpxml_content: str, output_path: Path) -> None:
        """Write FCPXML content to file."""
        output_path.write_text(fcpxml_content, encoding="utf-8")


# ============================================================================
# Console UI
# ============================================================================


class ConsoleUI:
    """Rich console interface with progress bars and formatted output."""

    HEADER_WIDTH = 64

    def __init__(self):
        if RICH_AVAILABLE:
            self.console = Console()
        self.use_rich = RICH_AVAILABLE

    # ---- Banner ----

    def print_banner(self):
        if self.use_rich:
            banner = Text(justify="center")
            banner.append(f"{APP_NAME}", style="bold cyan")
            banner.append(f"  v{VERSION}\n", style="dim cyan")
            banner.append(
                "Automatic Scene Detection for Final Cut Pro", style="dim white"
            )
            panel = Panel(
                banner, box=box.DOUBLE, border_style="cyan", padding=(1, 4)
            )
            self.console.print(panel)
            self.console.print()
        else:
            w = self.HEADER_WIDTH
            print("\n" + "=" * w)
            title = f"{APP_NAME} v{VERSION}"
            print(title.center(w))
            print("Automatic Scene Detection for Final Cut Pro".center(w))
            print("=" * w + "\n")

    # ---- Scan info ----

    def print_scan_info(self, directory: Path, count: int):
        if self.use_rich:
            self.console.print(f"  [dim]Directory:[/dim]  {directory}")
            if count > 0:
                self.console.print(
                    f"  [green]Found {count} video file(s)[/green]\n"
                )
            else:
                self.console.print(
                    "  [bold red]No video files found.[/bold red]\n"
                )
        else:
            print(f"  Directory:  {directory}")
            if count > 0:
                print(f"  Found {count} video file(s)\n")
            else:
                print("  No video files found.\n")

    # ---- Per-file output ----

    def print_separator(self):
        if self.use_rich:
            self.console.print(
                "  " + "\u2500" * (self.HEADER_WIDTH - 4), style="dim"
            )
        else:
            print("  " + "-" * (self.HEADER_WIDTH - 4))

    def print_file_header(self, index: int, total: int, filename: str):
        if self.use_rich:
            self.console.print(
                f"\n  [bold white][{index}/{total}][/bold white]  "
                f"[bold cyan]{filename}[/bold cyan]"
            )
        else:
            print(f"\n  [{index}/{total}]  {filename}")

    def print_video_info(self, width, height, fps, duration):
        dur_str = format_duration(duration)
        if self.use_rich:
            self.console.print(
                f"        [dim]Resolution:[/dim] {width}\u00d7{height}  "
                f"[dim]FPS:[/dim] {fps:.2f}  "
                f"[dim]Duration:[/dim] {dur_str}"
            )
        else:
            print(
                f"        Resolution: {width}x{height}  "
                f"FPS: {fps:.2f}  Duration: {dur_str}"
            )

    def create_progress(self):
        """Create a rich Progress bar (or None for fallback)."""
        if self.use_rich:
            return Progress(
                TextColumn("        [dim]Analyzing[/dim]"),
                BarColumn(
                    bar_width=40,
                    complete_style="cyan",
                    finished_style="green",
                ),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console,
                transient=False,
            )
        return None

    def print_simple_progress(self, current, total, bar_width=40):
        """Fallback progress bar for terminals without rich."""
        if total <= 0:
            return
        pct = min(current / total, 1.0)
        filled = int(bar_width * pct)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        sys.stdout.write(f"\r        Analyzing [{bar}] {pct * 100:5.1f}%")
        sys.stdout.flush()
        if pct >= 1.0:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def print_analysis_result(
        self, num_scenes, kept_duration, total_duration, kept_pct
    ):
        kept_str = format_duration(kept_duration)
        total_str = format_duration(total_duration)
        if self.use_rich:
            self.console.print(
                f"        [green]\u2713[/green] Detected "
                f"[bold]{num_scenes}[/bold] scene(s)  |  "
                f"Keeping [bold]{kept_str}[/bold] of {total_str} "
                f"([bold]{kept_pct:.1f}%[/bold])"
            )
        else:
            print(
                f"        [OK] Detected {num_scenes} scene(s)  |  "
                f"Keeping {kept_str} of {total_str} ({kept_pct:.1f}%)"
            )

    def print_saved(self, filename: str):
        if self.use_rich:
            self.console.print(
                f"        [green]\u2713[/green] Saved: [bold]{filename}[/bold]"
            )
        else:
            print(f"        [OK] Saved: {filename}")

    def print_error(self, message: str):
        if self.use_rich:
            self.console.print(
                f"        [bold red]\u2717 Error:[/bold red] {escape(message)}"
            )
        else:
            print(f"        [ERROR] {message}")

    # ---- Summary ----

    def print_summary(
        self, files_processed, files_total, total_dur, kept_dur, files_generated
    ):
        kept_pct = (kept_dur / total_dur * 100) if total_dur > 0 else 0

        if self.use_rich:
            self.console.print()
            table = Table(
                title="Complete",
                box=box.HEAVY,
                title_style="bold green",
                border_style="green",
                show_header=False,
                padding=(0, 2),
            )
            table.add_column("Key", style="dim white", min_width=20)
            table.add_column("Value", style="bold white")
            table.add_row("Files processed", f"{files_processed} / {files_total}")
            table.add_row("Total duration", format_duration(total_dur))
            table.add_row(
                "Kept duration",
                f"{format_duration(kept_dur)} ({kept_pct:.1f}%)",
            )
            table.add_row("FCPXML files generated", str(files_generated))
            self.console.print(table)
            self.console.print()
        else:
            print()
            w = self.HEADER_WIDTH
            print("  " + "=" * (w - 4))
            print("  COMPLETE")
            print("  " + "-" * (w - 4))
            print(f"    Files processed:       {files_processed} / {files_total}")
            print(f"    Total duration:        {format_duration(total_dur)}")
            print(
                f"    Kept duration:         {format_duration(kept_dur)} ({kept_pct:.1f}%)"
            )
            print(f"    FCPXML files generated: {files_generated}")
            print("  " + "=" * (w - 4))
            print()


# ============================================================================
# Argument Parser
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        prog="fcpxml-autocut",
        description=f"{APP_NAME} \u2014 Automatic Scene Detection for Final Cut Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  fcpxml-autocut                          Process all videos in current dir\n"
            "  fcpxml-autocut -s 0.7                   More aggressive cutting\n"
            "  fcpxml-autocut -s 0.3                   Keep more footage\n"
            "  fcpxml-autocut --min-scene 3.0          Minimum 3-second scenes\n"
            "  fcpxml-autocut --frame-skip 2           More accurate (slower)\n"
            "  fcpxml-autocut --scale 0.5              Half-res analysis\n"
        ),
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {VERSION}"
    )

    parser.add_argument(
        "-s",
        "--sensitivity",
        type=float,
        default=DEFAULT_SENSITIVITY,
        metavar="VAL",
        help=(
            f"Detection sensitivity 0.0-1.0 (default: {DEFAULT_SENSITIVITY}). "
            "Higher values cut more aggressively."
        ),
    )

    parser.add_argument(
        "--min-scene",
        type=float,
        default=DEFAULT_MIN_SCENE_DURATION,
        metavar="SEC",
        help=f"Minimum scene duration in seconds (default: {DEFAULT_MIN_SCENE_DURATION})",
    )

    parser.add_argument(
        "--min-cut",
        type=float,
        default=DEFAULT_MIN_CUT_DURATION,
        metavar="SEC",
        help=f"Minimum gap between scenes to actually cut (default: {DEFAULT_MIN_CUT_DURATION})",
    )

    parser.add_argument(
        "--trim-start",
        type=float,
        default=DEFAULT_TRIM_START,
        metavar="SEC",
        help=(
            f"Trim N seconds from the start of each scene (default: {DEFAULT_TRIM_START}). "
            "Removes camera settling, focus adjustment, zoom setup at scene start."
        ),
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=DEFAULT_FRAME_SKIP,
        metavar="N",
        help=f"Analyze every Nth frame (default: {DEFAULT_FRAME_SKIP})",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE_FACTOR,
        metavar="F",
        help=f"Scale factor for analysis frames, 0.1-1.0 (default: {DEFAULT_SCALE_FACTOR})",
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=".",
        help="Directory with video files (default: current directory)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for FCPXML files (default: same as input)",
    )

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================


def wait_before_exit(exit_code: int = 0) -> None:
    """Pause before closing so the user can read the output (especially when running as exe)."""
    if exit_code != 0 or getattr(sys, "frozen", False):
        try:
            input("\nPress Enter to exit...")
        except (EOFError, KeyboardInterrupt):
            pass


def main():
    args = parse_args()

    ui = ConsoleUI()
    ui.print_banner()

    # --- Discover videos ---
    work_dir = Path(args.directory).resolve()
    if not work_dir.exists():
        ui.print_error(f"Directory not found: {work_dir}")
        wait_before_exit(1)
        sys.exit(1)

    videos = discover_videos(work_dir)
    ui.print_scan_info(work_dir, len(videos))

    if not videos:
        wait_before_exit(0)
        sys.exit(0)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else work_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Create analyzer ---
    analyzer = VideoAnalyzer(
        frame_skip=args.frame_skip,
        scale_factor=args.scale,
        sensitivity=args.sensitivity,
        min_scene_duration=args.min_scene,
        min_cut_duration=args.min_cut,
        trim_start=args.trim_start,
    )

    # --- Process each video ---
    total_duration = 0.0
    total_kept = 0.0
    files_generated = 0

    for idx, video_path in enumerate(videos, 1):
        if idx > 1:
            ui.print_separator()

        ui.print_file_header(idx, len(videos), video_path.name)

        try:
            # Step 1: quick metadata read
            info = get_video_info(video_path)
            ui.print_video_info(
                info["width"], info["height"], info["fps"], info["duration"]
            )

            # Step 2: motion analysis with progress
            if ui.use_rich:
                progress_ctx = ui.create_progress()
                with progress_ctx as progress:
                    task = progress.add_task("analyze", total=100)

                    def _rich_cb(cur, tot, _p=progress, _t=task):
                        pct = (cur / tot * 100.0) if tot > 0 else 0
                        _p.update(_t, completed=min(pct, 100.0))

                    result = analyzer.analyze(video_path, info, progress_callback=_rich_cb)
                    progress.update(task, completed=100)
            else:
                def _simple_cb(cur, tot):
                    ui.print_simple_progress(cur, tot)

                result = analyzer.analyze(video_path, info, progress_callback=_simple_cb)
                ui.print_simple_progress(1, 1)

            # Step 3: compute stats
            segments = result["segments"]
            kept_frames = sum(e - s for s, e in segments)
            fps = info["fps"]
            kept_duration = kept_frames / fps if fps > 0 else 0
            total_dur = info["duration"]
            kept_pct = (kept_frames / info["total_frames"] * 100) if info["total_frames"] > 0 else 0

            ui.print_analysis_result(len(segments), kept_duration, total_dur, kept_pct)

            # Step 4: generate and save FCPXML
            fcpxml_content = FCPXMLGenerator.generate(info, segments)
            out_name = f"{video_path.stem}_autocut.fcpxml"
            out_path = output_dir / out_name
            FCPXMLGenerator.save(fcpxml_content, out_path)

            ui.print_saved(out_name)

            total_duration += total_dur
            total_kept += kept_duration
            files_generated += 1

        except Exception as exc:
            ui.print_error(str(exc))

    # --- Summary ---
    ui.print_summary(
        files_processed=files_generated,
        files_total=len(videos),
        total_dur=total_duration,
        kept_dur=total_kept,
        files_generated=files_generated,
    )

    wait_before_exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}")
        wait_before_exit(1)
        sys.exit(1)
