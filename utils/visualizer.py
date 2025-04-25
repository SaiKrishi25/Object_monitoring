import cv2
import numpy as np
from typing import Tuple, Optional

def draw_box(frame: np.ndarray, track_id: int, x: int, y: int, w: int, h: int, label: str = "Object", color: Optional[Tuple[int, int, int]] = None) -> None:
    if color is None or not isinstance(color, tuple) or len(color) != 3:
        color = ((track_id * 57) % 255, (track_id * 121) % 255, (track_id * 233) % 255)
    color = tuple(map(int, color))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = f"{label} #{track_id}"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - text_height - baseline - 5), (x + text_width + 5, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_fps(frame: np.ndarray, fps: float) -> None:
    fps_text = f"FPS: {fps:.1f}"
    (text_width, text_height), baseline = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + 10), (0, 0, 0), -1)
    cv2.putText(frame, fps_text, (15, 10 + text_height + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def draw_status_info(frame: np.ndarray, active_count: int, new_count: int, missing_count: int) -> None:
    height, width = frame.shape[:2]
    panel_width = 250
    panel_height = 130
    cv2.rectangle(frame, (width - panel_width - 10, 10), (width - 10, panel_height + 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (width - panel_width - 10, 10), (width - 10, panel_height + 10), (255, 255, 255), 1)
    cv2.putText(frame, "OBJECT MONITOR", (width - panel_width + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Active Objects: {active_count}", (width - panel_width + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"New Objects: {new_count}", (width - panel_width + 10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    cv2.putText(frame, f"Missing Objects: {missing_count}", (width - panel_width + 10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)