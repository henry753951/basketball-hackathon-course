from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import cv2
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

_MATPLOTLIB_FONT_CONFIGURED = False


def configure_matplotlib_fonts() -> None:
    """Use a CJK-capable font when the runtime provides one."""
    global _MATPLOTLIB_FONT_CONFIGURED
    if _MATPLOTLIB_FONT_CONFIGURED:
        return
    preferred_fonts = [
        "Microsoft JhengHei",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False
    _MATPLOTLIB_FONT_CONFIGURED = True


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_image_rgb(path: str | Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(path: str | Path, image_rgb: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR))
    return path


def show_image(
    image_rgb: np.ndarray, title: str | None = None, figsize=(10, 6)
) -> None:
    configure_matplotlib_fonts()
    plt.figure(figsize=figsize)
    plt.imshow(image_rgb)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def bottom_center(box_xyxy: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return ((x1 + x2) / 2.0, y2)


def draw_boxes(
    image_rgb: np.ndarray,
    boxes: Iterable[Sequence[float]],
    labels: Iterable[str] | None = None,
    color=(255, 80, 80),
    thickness: int = 3,
) -> np.ndarray:
    image = image_rgb.copy()
    labels = list(labels) if labels is not None else [""] * len(list(boxes))
    boxes = list(boxes)
    if len(labels) != len(boxes):
        labels = [""] * len(boxes)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
    return image


def draw_points(
    image_rgb: np.ndarray,
    points: Iterable[Sequence[float]],
    labels: Iterable[str] | None = None,
    color=(40, 120, 255),
    radius: int = 7,
) -> np.ndarray:
    image = image_rgb.copy()
    image_height, image_width = image.shape[:2]
    points = list(points)
    labels = list(labels) if labels is not None else [""] * len(points)
    for p, label in zip(points, labels):
        x, y = int(round(float(p[0]))), int(round(float(p[1])))
        cv2.circle(image, (x, y), radius, color, -1)
        if label:
            text = str(label)
            (text_width, text_height), baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2,
            )
            text_x = x + 8
            if text_x + text_width > image_width - 8:
                text_x = max(8, x - text_width - 8)
            text_y = y - 8
            if text_y - text_height < 8:
                text_y = min(image_height - 8, y + text_height + baseline + 8)
            cv2.putText(
                image,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
    return image


def side_by_side(
    left_rgb: np.ndarray, right_rgb: np.ndarray, max_width: int = 1500
) -> np.ndarray:
    left = np.asarray(left_rgb)
    right = np.asarray(right_rgb)
    h = max(left.shape[0], right.shape[0])

    def resize_to_height(img):
        if img.shape[0] == h:
            return img
        w = int(img.shape[1] * h / img.shape[0])
        return cv2.resize(img, (w, h))

    left = resize_to_height(left)
    right = resize_to_height(right)
    canvas = np.ones((h, left.shape[1] + right.shape[1], 3), dtype=np.uint8) * 255
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] : left.shape[1] + right.shape[1]] = right
    if canvas.shape[1] > max_width:
        new_h = int(canvas.shape[0] * max_width / canvas.shape[1])
        canvas = cv2.resize(canvas, (max_width, new_h))
    return canvas
