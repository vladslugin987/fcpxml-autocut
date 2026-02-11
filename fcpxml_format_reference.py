"""
Справка: корректная генерация формата для FCPXML (чтобы FCP не выдавал
"Encountered an unexpected value" / "The item is not an edit frame").

Используйте эти значения в своём скрипте (AI_Rough_Cut и т.п.) вместо
жёстко заданных 3840x2160 и p59.94.
"""

# Соответствие FPS → метка в имени формата (без десятичной точки)
FPS_LABEL = {
    23.976: "2398",
    24.0: "24",
    25.0: "25",
    29.97: "2997",
    30.0: "30",
    59.94: "5994",
    60.0: "60",
}

# Соответствие FPS → frameDuration (длительность одного кадра в секундах)
# frameDuration = "знаменатель/числительs" для fps = числитель/знаменатель
FRAME_DURATION = {
    23.976: "1001/24000s",
    24.0: "1/24s",
    25.0: "1/25s",
    29.97: "1001/30000s",
    30.0: "1/30s",
    59.94: "1001/60000s",
    60.0: "1/60s",
}


def get_fps_label(fps: float) -> str:
    for ref, label in FPS_LABEL.items():
        if abs(fps - ref) < 0.1:
            return label
    return str(int(round(fps)))


def get_frame_duration(fps: float) -> str:
    for ref, fd in FRAME_DURATION.items():
        if abs(fps - ref) < 0.1:
            return fd
    return f"1/{int(round(fps))}s"


def format_name_for_fcpxml(width: int, height: int, fps: float) -> str:
    """Имя формата для тега <format name="...">. Без десятичной точки в FPS."""
    label = get_fps_label(fps)
    return f"FFVideoFormat{width}x{height}p{label}"


# Пример корректного <format> и sequence (подставьте свои width, height, fps):
#
#   width, height = 1920, 1080   # из видео, не 3840x2160
#   fps = 59.94
#   name = format_name_for_fcpxml(width, height, fps)  # FFVideoFormat1920x1080p5994
#   frame_duration = get_frame_duration(fps)           # 1001/60000s
#
#   <format id="r0" name="FFVideoFormat1920x1080p5994" frameDuration="1001/60000s" width="1920" height="1080"/>
#   <sequence format="r0" ...>
#
# Версию fcpxml лучше ставить 1.9: <fcpxml version="1.9">
