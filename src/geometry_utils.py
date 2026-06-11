from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .cv_utils import load_json, read_image_rgb, draw_points, save_image_rgb


def compute_homography(
    camera_points: Sequence[Sequence[float]], bev_points: Sequence[Sequence[float]]
) -> np.ndarray:
    src = np.asarray(camera_points, dtype=np.float32)
    dst = np.asarray(bev_points, dtype=np.float32)
    if len(src) < 4 or len(dst) < 4:
        raise ValueError("Homography 至少需要 4 組對應點。")
    H, mask = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography 失敗，請確認點位順序與座標是否正確。")
    return H


def project_points(points_xy: Sequence[Sequence[float]], H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out


def load_sample_homography(course_root: str | Path):
    course_root = Path(course_root)
    data = load_json(
        course_root / "assets" / "samples" / "sample_homography_points.json"
    )
    H = compute_homography(data["camera_points"], data["bev_points"])
    return H, data


def draw_projection_pair(
    course_root: str | Path,
    camera_point: Sequence[float],
    bev_point: Sequence[float],
    output_path: str | Path | None = None,
):
    from .cv_utils import side_by_side, show_image

    course_root = Path(course_root)
    frame = read_image_rgb(
        course_root / "assets" / "samples" / "sample_court_frame.png"
    )
    bev = read_image_rgb(course_root / "assets" / "samples" / "sample_bev_court.png")
    frame_vis = draw_points(frame, [camera_point], ["camera point"])
    bev_vis = draw_points(bev, [bev_point], ["BEV point"])
    combo = side_by_side(frame_vis, bev_vis)
    if output_path is not None:
        save_image_rgb(output_path, combo)
    show_image(combo, "camera point → BEV point")
    return combo
