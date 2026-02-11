#!/usr/bin/env python3
"""
AI Rough Cut — один FCPXML из нескольких видео (стабильные сегменты).
Корректный формат для FCP: имя формата p5994, frameDuration, разрешение из видео.
"""

import os
import sys
import urllib.parse

import cv2
import numpy as np

# Импорт корректных меток формата из справки
from fcpxml_format_reference import (
    get_fps_label,
    get_frame_duration,
    format_name_for_fcpxml,
)

# --- Настройки ---
ANALYSIS_STEP = 2
MEAN_SPEED_THRESHOLD = 3.0
JERK_THRESHOLD = 0.9
SMOOTHING_WINDOW = 6
MIN_CLIP_SEC = 1.5
TRIM_IN_SEC = 0.1
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".mkv", ".avi")
DEBUG_MODE = False
FCPXML_VERSION = "1.9"


def analyze_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "", 0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or frame_count <= 0:
        cap.release()
        return None, "", 0, 0, 0

    v_fps = 59.94 if 59.9 < fps < 60.0 else (29.97 if 29.9 < fps < 30.0 else round(fps))
    frame_duration = get_frame_duration(fps)

    proc_w, proc_h = 640, 360
    min_clip_frames = int(MIN_CLIP_SEC * v_fps)
    trim_frames = int(TRIM_IN_SEC * v_fps)

    if DEBUG_MODE:
        os.makedirs("DELETED", exist_ok=True)
        debug_path = os.path.join("DELETED", f"STRICT_{os.path.basename(video_path)}")
        out_debug = cv2.VideoWriter(
            debug_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (proc_w, proc_h)
        )

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None, "", 0, 0, 0

    prev_gray = cv2.cvtColor(cv2.resize(frame, (proc_w, proc_h)), cv2.COLOR_BGR2GRAY)
    raw_speeds = []
    good_segments = []
    start_frame = None
    curr_idx = 0

    while True:
        for _ in range(ANALYSIS_STEP - 1):
            cap.grab()
            curr_idx += 1
        ret, frame = cap.read()
        curr_idx += 1
        if not ret:
            break

        gray = cv2.cvtColor(cv2.resize(frame, (proc_w, proc_h)), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        current_speed = np.mean(mag)
        raw_speeds.append(current_speed)

        if len(raw_speeds) >= SMOOTHING_WINDOW:
            smooth_speed = np.mean(raw_speeds[-SMOOTHING_WINDOW:])
            jerk = abs(
                smooth_speed - np.mean(raw_speeds[-SMOOTHING_WINDOW - 1 : -1])
            )
        else:
            smooth_speed, jerk = current_speed, 0.0

        is_good = (smooth_speed < MEAN_SPEED_THRESHOLD) and (jerk < JERK_THRESHOLD)

        if is_good:
            if start_frame is None:
                start_frame = curr_idx
        else:
            if DEBUG_MODE and "out_debug" in dir():
                out_debug.write(cv2.resize(frame, (proc_w, proc_h)))
            if start_frame is not None:
                duration = curr_idx - start_frame
                if duration > min_clip_frames:
                    seg_dur = duration - 2 * trim_frames
                    if seg_dur > 0:
                        good_segments.append((start_frame + trim_frames, seg_dur))
                start_frame = None
        prev_gray = gray

    if DEBUG_MODE and "out_debug" in dir():
        out_debug.release()

    cap.release()

    if not good_segments:
        return None, frame_duration, v_fps, width, height
    return good_segments, frame_duration, v_fps, width, height


def create_fcpxml(all_results, output_name: str = "AI_Rough_Cut.fcpxml"):
    if not all_results:
        return

    # Первый файл задаёт формат проекта (разрешение и FPS)
    first = all_results[0]
    fd_str, v_fps0, width, height = first[2], first[3], first[4], first[5]
    format_name = format_name_for_fcpxml(width, height, v_fps0)

    def parse_fd(fd_str):
        parts = fd_str.replace("s", "").split("/")
        if len(parts) >= 2:
            return int(parts[0].strip()), int(parts[1].strip())
        return 1, int(parts[0].strip())

    header = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="{FCPXML_VERSION}">
    <resources>
        <format id="r0" name="{format_name}" frameDuration="{fd_str}" width="{width}" height="{height}"/>
'''

    assets = ""
    spine = ""
    current_total_ticks = 0

    for idx, (filename, segments, fd_str, _v_fps, _w, _h) in enumerate(all_results):
        res_id = f"r{idx + 1}"
        tick_num, tick_den = parse_fd(fd_str)
        safe_name = urllib.parse.quote(os.path.basename(filename), safe="")
        assets += f'        <asset id="{res_id}" name="{filename}" start="0s" hasVideo="1" hasAudio="1" format="r0">\n'
        assets += f'            <media-rep kind="original-media" src="./{safe_name}"/>\n'
        assets += f"        </asset>\n"

        for start_f, dur_f in segments:
            start_ticks = start_f * tick_num
            dur_ticks = dur_f * tick_num
            offset_ticks = current_total_ticks
            spine += f'                <asset-clip ref="{res_id}" offset="{offset_ticks}/{tick_den}s" name="{filename}" start="{start_ticks}/{tick_den}s" duration="{dur_ticks}/{tick_den}s" format="r0"/>\n'
            current_total_ticks += dur_ticks

    footer = f'''    </resources>
    <library>
        <event name="AI_Cut">
            <project name="Smart_Rough_Cut">
                <sequence format="r0" tcStart="0s" tcFormat="NDF">
                    <spine>
{spine}                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>'''

    with open(output_name, "w", encoding="utf-8") as f:
        f.write(header + assets + footer)


def main():
    if sys.platform == "win32":
        os.system("chcp 65001 > nul 2>&1")

    video_files = [
        f
        for f in os.listdir(".")
        if f.lower().endswith(VIDEO_EXTENSIONS) and "DELETED" not in f.upper()
    ]
    if not video_files:
        print("Видеофайлы не найдены в текущей папке.")
        return 1

    print("--- AI Rough Cut (один FCPXML, корректный формат) ---")
    processed_data = []
    for i, file in enumerate(video_files, 1):
        try:
            res = analyze_video(file)
            if res[0] is not None:
                processed_data.append((file, *res))
                print(f"  [OK] {i}/{len(video_files)} {file}")
            else:
                print(f"  [--] {i}/{len(video_files)} {file} (нет сегментов)")
        except Exception as e:
            print(f"  [ERROR] {file}: {e}")

    if processed_data:
        create_fcpxml(processed_data)
        print("\n" + "=" * 50)
        print("  AI_Rough_Cut.fcpxml создан. Можно открывать в FCP.")
        print("=" * 50)
    else:
        print("\nНет подходящих сегментов для экспорта.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
