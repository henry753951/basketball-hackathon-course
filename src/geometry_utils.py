from __future__ import annotations

from pathlib import Path
from typing import Sequence, TypedDict

import cv2
import numpy as np

from .cv_utils import load_json, read_image_rgb, draw_points, save_image_rgb


class HomographyPair(TypedDict):
    keypoint_name: str
    camera_xy: list[float]
    bev_xy: list[float]


DEFAULT_HOMOGRAPHY_PAIRS: list[HomographyPair] = [
    {
        "keypoint_name": "sample_right_lane_top_left",
        "camera_xy": [818.0, 480.0],
        "bev_xy": [620.0, 180.0],
    },
    {
        "keypoint_name": "sample_right_lane_top_right",
        "camera_xy": [1216.0, 453.0],
        "bev_xy": [820.0, 180.0],
    },
    {
        "keypoint_name": "sample_right_lane_bottom_right",
        "camera_xy": [1410.0, 657.0],
        "bev_xy": [820.0, 350.0],
    },
    {
        "keypoint_name": "sample_right_lane_bottom_left",
        "camera_xy": [672.0, 623.0],
        "bev_xy": [620.0, 350.0],
    },
    {
        "keypoint_name": "sample_right_sideline_top_left",
        "camera_xy": [674.0, 423.0],
        "bev_xy": [620.0, 90.0],
    },
    {
        "keypoint_name": "sample_right_sideline_top_right",
        "camera_xy": [1434.0, 424.0],
        "bev_xy": [820.0, 90.0],
    },
]

DEFAULT_SAMPLE_PLAYER_BBOX_XYXY: list[float] = [545.8197, 469.2618, 627.8652, 714.5692]


def default_homography_pairs() -> list[HomographyPair]:
    return [
        {
            "keypoint_name": pair["keypoint_name"],
            "camera_xy": pair["camera_xy"].copy(),
            "bev_xy": pair["bev_xy"].copy(),
        }
        for pair in DEFAULT_HOMOGRAPHY_PAIRS
    ]


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


def render_bev_court(spec_path: str | Path) -> np.ndarray:
    spec = load_json(spec_path)
    canvas = spec["canvas"]
    style = spec.get("style", {})
    width = int(canvas["width"])
    height = int(canvas["height"])
    background = tuple(int(v) for v in canvas.get("background_rgb", [248, 248, 248]))
    default_color = tuple(int(v) for v in style.get("line_rgb", [40, 40, 40]))
    default_thickness = int(style.get("line_thickness", 2))

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = background

    for item in spec.get("polylines", []):
        points = np.asarray(item["points"], dtype=np.int32).reshape(-1, 1, 2)
        color = tuple(int(v) for v in item.get("rgb", default_color))
        thickness = int(item.get("thickness", default_thickness))
        cv2.polylines(
            image,
            [points],
            isClosed=bool(item.get("closed", False)),
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    for item in spec.get("circles", []):
        center = tuple(int(v) for v in item["center"])
        radius = int(item["radius"])
        color = tuple(int(v) for v in item.get("rgb", default_color))
        thickness = int(item.get("thickness", default_thickness))
        cv2.circle(
            image,
            center,
            radius,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    return image


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
    bev = render_bev_court(
        course_root / "assets" / "samples" / "sample_bev_court.json"
    )
    frame_vis = draw_points(frame, [camera_point], ["camera point"])
    bev_vis = draw_points(bev, [bev_point], ["BEV point"])
    combo = side_by_side(frame_vis, bev_vis)
    if output_path is not None:
        save_image_rgb(output_path, combo)
    show_image(combo, "camera point → BEV point")
    return combo
