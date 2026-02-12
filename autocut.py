#!/usr/bin/env python3
"""
FCPXML AutoCut - Automatic scene detection for Final Cut Pro
Analyzes camera footage using optical flow and generates clean FCPXML timelines
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from fractions import Fraction

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Install 'rich' for better console output: pip install rich")


class VideoAnalyzer:
    """Analyzes video for camera shake and chaotic movement"""
    
    SUPPORTED_FORMATS = {'.mov', '.mp4', '.mkv', '.avi', '.m4v', '.mxf', '.mts', '.m2ts'}
    
    def __init__(self, 
                 sensitivity: float = 0.5,
                 min_scene_duration: float = 2.0,
                 min_cut_duration: float = 0.5,
                 frame_skip: int = 4,
                 scale: float = 0.25):
        
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.min_scene_duration = min_scene_duration
        self.min_cut_duration = min_cut_duration
        self.frame_skip = frame_skip
        self.scale = scale
        
        self.console = Console() if RICH_AVAILABLE else None
    
    def analyze_video(self, video_path: str) -> Tuple[List[Tuple[float, float]], dict]:
        """
        Analyze video and return list of good segments
        
        Returns:
            segments: List of (start_time, end_time) tuples in seconds
            metadata: Video metadata dictionary
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        metadata = {
            'fps': fps,
            'width': width,
            'height': height,
            'duration': duration,
            'total_frames': total_frames
        }
        
        if self.console:
            self.console.print(f"\n[cyan]Analyzing:[/cyan] {Path(video_path).name}")
            self.console.print(f"  Resolution: {width}x{height} | FPS: {fps:.2f} | Duration: {duration:.1f}s")
        
        motion_scores = []
        prev_gray = None
        frame_count = 0
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Processing frames...", total=total_frames // self.frame_skip)
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % self.frame_skip == 0:
                        score = self._analyze_frame(frame, prev_gray)
                        if score is not None:
                            motion_scores.append((frame_count / fps, score))
                        
                        # Update previous frame
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if self.scale != 1.0:
                            new_width = int(width * self.scale)
                            new_height = int(height * self.scale)
                            gray = cv2.resize(gray, (new_width, new_height))
                        prev_gray = gray
                        
                        progress.update(task, advance=1)
                    
                    frame_count += 1
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_skip == 0:
                    score = self._analyze_frame(frame, prev_gray)
                    if score is not None:
                        motion_scores.append((frame_count / fps, score))
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if self.scale != 1.0:
                        new_width = int(width * self.scale)
                        new_height = int(height * self.scale)
                        gray = cv2.resize(gray, (new_width, new_height))
                    prev_gray = gray
                    
                    if frame_count % (self.frame_skip * 100) == 0:
                        print(f"  Processed {frame_count}/{total_frames} frames...")
                
                frame_count += 1
        
        cap.release()
        
        # Process motion scores into segments
        segments = self._extract_segments(motion_scores, duration)
        
        return segments, metadata
    
    def _analyze_frame(self, frame, prev_gray) -> Optional[float]:
        """Analyze single frame for motion"""
        if prev_gray is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.scale != 1.0:
            new_width = int(frame.shape[1] * self.scale)
            new_height = int(frame.shape[0] * self.scale)
            gray = cv2.resize(gray, (new_width, new_height))
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate motion metrics
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mean_motion = np.mean(magnitude)
        std_motion = np.std(magnitude)
        
        # Combine metrics (high mean OR high std indicates shake/chaos)
        motion_score = mean_motion + std_motion * 0.5
        
        return motion_score
    
    def _extract_segments(self, motion_scores: List[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
        """Extract good segments from motion scores"""
        if not motion_scores:
            return [(0, duration)]
        
        times = np.array([t for t, _ in motion_scores])
        scores = np.array([s for _, s in motion_scores])
        
        # Smooth scores
        window = 5
        if len(scores) >= window:
            kernel = np.ones(window) / window
            scores = np.convolve(scores, kernel, mode='same')
        
        # Calculate adaptive threshold
        percentile = 70 - (self.sensitivity * 40)  # 70% for 0.5 sensitivity
        threshold = np.percentile(scores, percentile)
        
        # Hysteresis thresholds
        high_threshold = threshold * 1.2
        low_threshold = threshold * 0.8
        
        # Detect good/bad segments
        is_good = scores < low_threshold
        state = is_good[0]
        segments = []
        segment_start = 0
        
        for i in range(1, len(scores)):
            if state:  # Currently in good segment
                if scores[i] > high_threshold:
                    state = False
                    segments.append((times[segment_start], times[i]))
            else:  # Currently in bad segment
                if scores[i] < low_threshold:
                    state = True
                    segment_start = i
        
        # Close last segment if needed
        if state:
            segments.append((times[segment_start], duration))
        
        # Filter short segments and merge close ones
        segments = self._filter_segments(segments, duration)
        
        return segments
    
    def _filter_segments(self, segments: List[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
        """Filter out short segments and merge close ones"""
        if not segments:
            return []
        
        # Remove segments shorter than minimum
        filtered = [
            (start, end) for start, end in segments
            if (end - start) >= self.min_scene_duration
        ]
        
        if not filtered:
            return []
        
        # Merge segments with small gaps
        merged = [filtered[0]]
        for start, end in filtered[1:]:
            prev_start, prev_end = merged[-1]
            gap = start - prev_end
            
            if gap < self.min_cut_duration:
                # Merge with previous segment
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        
        return merged


class FCPXMLGenerator:
    """Generates FCPXML 1.9 files for Final Cut Pro"""
    
    # Standard FCP format IDs
    FORMATS = {
        23.976: 'FFVideoFormat1080p2398',
        24.0: 'FFVideoFormat1080p24',
        25.0: 'FFVideoFormat1080p25',
        29.97: 'FFVideoFormat1080p2997',
        30.0: 'FFVideoFormat1080p30',
        50.0: 'FFVideoFormat1080p50',
        59.94: 'FFVideoFormat1080p5994',
        60.0: 'FFVideoFormat1080p60',
    }
    
    def __init__(self, video_path: str, segments: List[Tuple[float, float]], metadata: dict):
        self.video_path = Path(video_path)
        self.segments = segments
        self.metadata = metadata
        self.fps = metadata['fps']
        self.width = metadata['width']
        self.height = metadata['height']
        self.duration = metadata['duration']
        
        # Get frame duration as fraction
        self.frame_duration = self._get_frame_duration()
        self.format_id = self._get_format_id()
    
    def _get_frame_duration(self) -> Fraction:
        """Get frame duration as a proper fraction"""
        fps = self.fps
        
        # Common frame rates
        if abs(fps - 23.976) < 0.01:
            return Fraction(1001, 24000)
        elif abs(fps - 29.97) < 0.01:
            return Fraction(1001, 30000)
        elif abs(fps - 59.94) < 0.01:
            return Fraction(1001, 60000)
        elif abs(fps - 119.88) < 0.01:
            return Fraction(1001, 120000)
        else:
            # Round to nearest integer fps
            fps_int = round(fps)
            return Fraction(1, fps_int)
    
    def _get_format_id(self) -> str:
        """Get proper FCP format ID"""
        fps = self.fps
        
        # Find closest standard format
        closest_fps = min(self.FORMATS.keys(), key=lambda x: abs(x - fps))
        
        return self.FORMATS[closest_fps]
    
    def _time_to_rational(self, seconds: float) -> str:
        """Convert seconds to rational time string aligned to frame boundaries"""
        # Round to nearest frame
        frame_num = round(seconds * self.fps)
        
        # Calculate time as frames * frame_duration
        numerator = frame_num * self.frame_duration.numerator
        denominator = self.frame_duration.denominator
        
        return f"{numerator}/{denominator}s"
    
    def generate_fcpxml(self, output_path: str):
        """Generate FCPXML file"""
        
        # Calculate total duration of timeline
        if self.segments:
            timeline_duration = sum(end - start for start, end in self.segments)
        else:
            timeline_duration = self.duration
        
        timeline_duration_rational = self._time_to_rational(timeline_duration)
        source_duration_rational = self._time_to_rational(self.duration)
        
        # Start building XML
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE fcpxml>',
            '<fcpxml version="1.9">',
            '  <resources>',
            f'    <format id="{self.format_id}" name="FFVideoFormat1080p{int(self.fps)}" frameDuration="{self.frame_duration.numerator}/{self.frame_duration.denominator}s" width="{self.width}" height="{self.height}" colorSpace="1-1-1 (Rec. 709)"/>',
            f'    <asset id="r1" name="{self.video_path.name}" uid="{self._generate_uid()}" src="file://{self.video_path.name}" start="0s" duration="{source_duration_rational}" hasVideo="1" format="{self.format_id}" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000"/>',
            '  </resources>',
            '  <library>',
            '    <event name="AutoCut">',
            f'      <project name="{self.video_path.stem}_autocut">',
            f'        <sequence format="{self.format_id}" duration="{timeline_duration_rational}" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48000">',
            '          <spine>',
        ]
        
        # Add clips for each segment
        current_time = Fraction(0)
        
        for i, (start, end) in enumerate(self.segments):
            clip_duration = end - start
            clip_offset = start
            
            # Convert to rational time aligned to frames
            duration_rational = self._time_to_rational(clip_duration)
            offset_rational = self._time_to_rational(clip_offset)
            start_rational = self._time_to_rational(current_time)
            
            xml_lines.append(
                f'            <asset-clip name="{self.video_path.name}" '
                f'offset="{start_rational}" ref="r1" duration="{duration_rational}" '
                f'start="{offset_rational}" tcFormat="NDF"/>'
            )
            
            current_time += Fraction(clip_duration)
        
        # Close XML structure
        xml_lines.extend([
            '          </spine>',
            '        </sequence>',
            '      </project>',
            '    </event>',
            '  </library>',
            '</fcpxml>',
        ])
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(xml_lines))
    
    def _generate_uid(self) -> str:
        """Generate a unique ID for the asset"""
        import hashlib
        hash_input = f"{self.video_path.name}{self.duration}".encode()
        return hashlib.md5(hash_input).hexdigest().upper()[:16]


def process_video(video_path: str, args, console=None) -> dict:
    """Process a single video file"""
    
    # Analyze video
    analyzer = VideoAnalyzer(
        sensitivity=args.sensitivity,
        min_scene_duration=args.min_scene,
        min_cut_duration=args.min_cut,
        frame_skip=args.frame_skip,
        scale=args.scale
    )
    
    try:
        segments, metadata = analyzer.analyze_video(video_path)
    except Exception as e:
        if console:
            console.print(f"[red]✗ Error analyzing video: {e}[/red]")
        else:
            print(f"✗ Error analyzing video: {e}")
        return None
    
    # Calculate statistics
    total_duration = metadata['duration']
    kept_duration = sum(end - start for start, end in segments)
    cut_percentage = ((total_duration - kept_duration) / total_duration * 100) if total_duration > 0 else 0
    
    # Generate FCPXML
    output_dir = Path(args.output_dir) if args.output_dir else Path(video_path).parent
    output_path = output_dir / f"{Path(video_path).stem}_autocut.fcpxml"
    
    generator = FCPXMLGenerator(video_path, segments, metadata)
    generator.generate_fcpxml(str(output_path))
    
    stats = {
        'video': Path(video_path).name,
        'total_duration': total_duration,
        'kept_duration': kept_duration,
        'cut_percentage': cut_percentage,
        'num_scenes': len(segments),
        'output': output_path.name
    }
    
    if console:
        console.print(f"[green]✓ Generated:[/green] {output_path.name}")
        console.print(f"  Scenes: {len(segments)} | Kept: {kept_duration:.1f}s / {total_duration:.1f}s ({100-cut_percentage:.1f}%)")
    else:
        print(f"✓ Generated: {output_path.name}")
        print(f"  Scenes: {len(segments)} | Kept: {kept_duration:.1f}s / {total_duration:.1f}s ({100-cut_percentage:.1f}%)")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='FCPXML AutoCut - Automatic scene detection for Final Cut Pro',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Process all videos in current directory
  %(prog)s -s 0.7                   # More aggressive cutting
  %(prog)s -s 0.3                   # Keep more footage
  %(prog)s --min-scene 3.0          # Minimum 3 second scenes
  %(prog)s -d /path/to/videos       # Process specific directory
  %(prog)s -o /path/to/output       # Save FCPXML to different directory
        """
    )
    
    parser.add_argument('-s', '--sensitivity', type=float, default=0.5,
                        help='Detection sensitivity 0.0-1.0 (default: 0.5)')
    parser.add_argument('--min-scene', type=float, default=2.0,
                        help='Minimum scene duration in seconds (default: 2.0)')
    parser.add_argument('--min-cut', type=float, default=0.5,
                        help='Minimum gap duration to cut (default: 0.5)')
    parser.add_argument('--frame-skip', type=int, default=4,
                        help='Analyze every Nth frame (default: 4)')
    parser.add_argument('--scale', type=float, default=0.25,
                        help='Scale factor for analysis (default: 0.25)')
    parser.add_argument('-d', '--directory', type=str, default='.',
                        help='Input directory with videos (default: current)')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory for FCPXML files (default: same as input)')
    
    args = parser.parse_args()
    
    # Validate arguments
    args.sensitivity = max(0.0, min(1.0, args.sensitivity))
    args.scale = max(0.1, min(1.0, args.scale))
    
    console = Console() if RICH_AVAILABLE else None
    
    # Find video files
    input_dir = Path(args.directory)
    video_files = []
    
    for ext in VideoAnalyzer.SUPPORTED_FORMATS:
        video_files.extend(input_dir.glob(f'*{ext}'))
        video_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    if not video_files:
        if console:
            console.print(f"[red]No video files found in {input_dir}[/red]")
        else:
            print(f"No video files found in {input_dir}")
        return
    
    # Print header
    if console:
        console.print(Panel.fit(
            f"[bold cyan]FCPXML AutoCut[/bold cyan]\n"
            f"Found {len(video_files)} video(s) | Sensitivity: {args.sensitivity}",
            border_style="cyan"
        ))
    else:
        print(f"\n{'='*60}")
        print(f"FCPXML AutoCut")
        print(f"Found {len(video_files)} video(s) | Sensitivity: {args.sensitivity}")
        print(f"{'='*60}\n")
    
    # Process videos
    all_stats = []
    
    for video_file in video_files:
        stats = process_video(str(video_file), args, console)
        if stats:
            all_stats.append(stats)
    
    # Print summary
    if all_stats and console:
        table = Table(title="\n Summary", show_header=True, header_style="bold cyan")
        table.add_column("Video", style="dim")
        table.add_column("Scenes", justify="right")
        table.add_column("Duration", justify="right")
        table.add_column("Kept", justify="right")
        table.add_column("Output", style="green")
        
        for stats in all_stats:
            table.add_row(
                stats['video'],
                str(stats['num_scenes']),
                f"{stats['total_duration']:.1f}s",
                f"{100 - stats['cut_percentage']:.1f}%",
                stats['output']
            )
        
        console.print(table)
    elif all_stats:
        print("\n" + "="*60)
        print("Summary:")
        print("="*60)
        for stats in all_stats:
            print(f"{stats['video']}: {stats['num_scenes']} scenes, {100-stats['cut_percentage']:.1f}% kept → {stats['output']}")
    
    if console:
        console.print("\n[cyan]→ Import the .fcpxml files in Final Cut Pro: File → Import → XML...[/cyan]")
    else:
        print("\n→ Import the .fcpxml files in Final Cut Pro: File → Import → XML...")


if __name__ == '__main__':
    main()