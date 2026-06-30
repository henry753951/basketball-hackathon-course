from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, TypedDict

import cv2
import numpy as np

from .cv_utils import load_json, read_image_rgb, draw_points, save_image_rgb


class HomographyPair(TypedDict):
    keypoint_name: str
    camera_xy: list[float]
    bev_xy: list[float]


COURT_PRESETS_CM: dict[str, dict[str, float]] = {
    "nba": {
        "court_width": 1524.0,
        "court_length": 2865.0,
        "three_point_arc_radius": 724.0,
        "straight_section_three_point_line": 424.0,
        "sideline_to_three_point_line": 91.0,
        "paint_width": 488.0,
        "paint_length": 579.0,
        "free_throw_line_distance": 457.0,
        "center_circle_radius": 183.0,
        "restricted_area_radius": 122.0,
        "rim_diameter": 46.0,
        "baseline_to_rim_center": 160.0,
        "baseline_to_throw_line": 835.0,
        "baseline_to_throw_line_length": 50.0,
    },
    "fiba": {
        "court_width": 1500.0,
        "court_length": 2800.0,
        "three_point_arc_radius": 675.0,
        "straight_section_three_point_line": 330.0,
        "sideline_to_three_point_line": 90.0,
        "paint_width": 490.0,
        "paint_length": 580.0,
        "free_throw_line_distance": 460.0,
        "center_circle_radius": 180.0,
        "restricted_area_radius": 125.0,
        "rim_diameter": 45.0,
        "baseline_to_rim_center": 157.0,
        "baseline_to_throw_line": 830.0,
        "baseline_to_throw_line_length": 50.0,
    },
}

COURT_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (2, 9),
    (11, 3),
    (9, 10),
    (10, 11),
    (1, 4),
    (0, 12),
    (12, 15),
    (15, 18),
    (15, 16),
    (16, 17),
    (5, 14),
    (14, 17),
    (17, 20),
    (27, 28),
    (28, 29),
    (29, 30),
    (30, 31),
    (31, 32),
    (28, 31),
    (29, 21),
    (21, 22),
    (22, 23),
    (23, 30),
]

COURT_EDGE_GROUPS: dict[str, set[int]] = {
    "outer": {0, 1, 2, 3, 4, 9, 18, 19, 20, 21, 22, 23},
    "paint": {5, 6, 7, 8, 24, 25, 26, 27},
    "lane": {10, 11, 12, 15, 16, 17},
    "center": {13, 14},
}

COURT_SPECIAL_SEGMENTS: list[tuple[str, tuple[int, int]]] = [
    ("lane", (1, 7)),
    ("lane", (4, 8)),
    ("lane", (28, 24)),
    ("lane", (31, 25)),
    ("outer", (18, 27)),
    ("outer", (20, 32)),
    ("outer", (28, 31)),
]

DEFAULT_COURT_TEMPLATE: dict[str, Any] = {
    "league": "nba",
    "measurement_unit": "feet",
    "scale": 20.0,
    "padding": 50,
    "line_thickness": 4,
    "background_hex": "#0F172A",
    "paint_fill_hex": "#1E293B",
    "colors": {
        "base": "#334155",
        "outer": "#22D3EE",
        "paint": "#F97316",
        "lane": "#A855F7",
        "center": "#FDE047",
        "rim": "#FB7185",
        "restricted": "#34D399",
        "backboard": "#60A5FA",
    },
}

BACKBOARD_TO_RIM_CM = 63.5
BACKBOARD_SPAN_CM = 183.0

DEFAULT_HOMOGRAPHY_PAIRS: list[HomographyPair] = [
    {
        "keypoint_name": "right_paint_ft_top",
        "camera_xy": [818.0, 480.0],
        "bev_xy": [1516.95, 390.68],
    },
    {
        "keypoint_name": "right_free_throw",
        "camera_xy": [1216.0, 453.0],
        "bev_xy": [1720.03, 550.0],
    },
    {
        "keypoint_name": "right_paint_ft_bottom",
        "camera_xy": [1410.0, 657.0],
        "bev_xy": [1720.03, 709.32],
    },
    {
        "keypoint_name": "right_paint_baseline_bottom",
        "camera_xy": [672.0, 623.0],
        "bev_xy": [1829.15, 709.32],
    },
    {
        "keypoint_name": "right_baseline_top",
        "camera_xy": [674.0, 423.0],
        "bev_xy": [1930.0, 50.0],
    },
    {
        "keypoint_name": "right_baseline_bottom",
        "camera_xy": [1434.0, 424.0],
        "bev_xy": [1930.0, 1050.0],
    },
]

DEFAULT_DEMO_PLAYER_BBOX_XYXY: list[float] = [545.8197, 469.2618, 627.8652, 714.5692]


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


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Invalid hex color: {value}")
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
    )


def _rgb_to_bgr(color_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return color_rgb[2], color_rgb[1], color_rgb[0]


def _to_pixel(point_xy: tuple[float, float], scale: float, padding: int) -> tuple[int, int]:
    return (
        int(round(point_xy[0] * scale + padding)),
        int(round(point_xy[1] * scale + padding)),
    )


def _court_measurements(
    league: str = "nba",
    measurement_unit: str = "feet",
) -> dict[str, float]:
    preset = COURT_PRESETS_CM.get(league.lower())
    if preset is None:
        raise ValueError(f"Unsupported basketball court league: {league}")
    if measurement_unit not in {"feet", "centimeters"}:
        raise ValueError(f"Unsupported measurement unit: {measurement_unit}")
    factor = 30.48 if measurement_unit == "feet" else 1.0
    return {key: value / factor for key, value in preset.items()}


def _court_vertices(
    league: str = "nba",
    measurement_unit: str = "feet",
) -> list[tuple[float, float]]:
    m = _court_measurements(league=league, measurement_unit=measurement_unit)
    paint_start = (m["court_width"] - m["paint_width"]) / 2.0
    center_y = m["court_width"] / 2.0
    length = m["court_length"]
    return [
        (0.0, 0.0),
        (0.0, m["sideline_to_three_point_line"]),
        (0.0, paint_start),
        (0.0, paint_start + m["paint_width"]),
        (0.0, m["court_width"] - m["sideline_to_three_point_line"]),
        (0.0, m["court_width"]),
        (m["baseline_to_rim_center"], center_y),
        (m["straight_section_three_point_line"], m["sideline_to_three_point_line"]),
        (m["straight_section_three_point_line"], m["court_width"] - m["sideline_to_three_point_line"]),
        (m["paint_length"], paint_start),
        (m["paint_length"], center_y),
        (m["paint_length"], paint_start + m["paint_width"]),
        (m["baseline_to_throw_line"], 0.0),
        (m["baseline_to_rim_center"] + m["three_point_arc_radius"], center_y),
        (m["baseline_to_throw_line"], m["court_width"]),
        (length / 2.0, 0.0),
        (length / 2.0, center_y),
        (length / 2.0, m["court_width"]),
        (length - m["baseline_to_throw_line"], 0.0),
        (length - m["baseline_to_rim_center"] - m["three_point_arc_radius"], center_y),
        (length - m["baseline_to_throw_line"], m["court_width"]),
        (length - m["paint_length"], paint_start),
        (length - m["paint_length"], center_y),
        (length - m["paint_length"], paint_start + m["paint_width"]),
        (length - m["straight_section_three_point_line"], m["sideline_to_three_point_line"]),
        (length - m["straight_section_three_point_line"], m["court_width"] - m["sideline_to_three_point_line"]),
        (length - m["baseline_to_rim_center"], center_y),
        (length, 0.0),
        (length, m["sideline_to_three_point_line"]),
        (length, paint_start),
        (length, paint_start + m["paint_width"]),
        (length, m["court_width"] - m["sideline_to_three_point_line"]),
        (length, m["court_width"]),
    ]


def _template_with_defaults(template: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {
        **DEFAULT_COURT_TEMPLATE,
        **(template or {}),
    }
    merged["colors"] = {
        **DEFAULT_COURT_TEMPLATE["colors"],
        **dict((template or {}).get("colors", {})),
    }
    return merged


def court_template_bounds(template: dict[str, Any] | None = None) -> tuple[float, float, float, float]:
    resolved = _template_with_defaults(template)
    scale = float(resolved["scale"])
    padding = int(resolved["padding"])
    m = _court_measurements(
        league=str(resolved["league"]),
        measurement_unit=str(resolved["measurement_unit"]),
    )
    return (
        float(padding),
        float(padding),
        float(padding + round(m["court_length"] * scale)),
        float(padding + round(m["court_width"] * scale)),
    )


def _edge_group(edge_index: int) -> str:
    for name, indexes in COURT_EDGE_GROUPS.items():
        if edge_index in indexes:
            return name
    return "base"


def court_edge_color_rgb(edge_index: int, template: dict[str, Any] | None = None) -> tuple[int, int, int]:
    resolved = _template_with_defaults(template)
    color_hex = str(resolved["colors"][_edge_group(edge_index)])
    return _hex_to_rgb(color_hex)


def court_group_color_rgb(group_name: str, template: dict[str, Any] | None = None) -> tuple[int, int, int]:
    resolved = _template_with_defaults(template)
    color_hex = str(resolved["colors"][group_name])
    return _hex_to_rgb(color_hex)


def _draw_circular_arc_from_three_points(
    image_bgr: np.ndarray,
    first_point: tuple[int, int],
    middle_point: tuple[int, int],
    last_point: tuple[int, int],
    color_bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    p1 = np.asarray(first_point, dtype=np.float64)
    p2 = np.asarray(middle_point, dtype=np.float64)
    p3 = np.asarray(last_point, dtype=np.float64)
    sum_sq_p2 = p2[0] ** 2 + p2[1] ** 2
    term1 = (p1[0] ** 2 + p1[1] ** 2 - sum_sq_p2) / 2.0
    term2 = (sum_sq_p2 - p3[0] ** 2 - p3[1] ** 2) / 2.0
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    if abs(det) < 1e-10:
        return
    center_x = (term1 * (p2[1] - p3[1]) - term2 * (p1[1] - p2[1])) / det
    center_y = ((p1[0] - p2[0]) * term2 - (p2[0] - p3[0]) * term1) / det
    center = (float(center_x), float(center_y))
    radius = float(np.hypot(center_x - p1[0], center_y - p1[1]))

    def angle_deg(point_xy: np.ndarray) -> float:
        return float(np.degrees(np.arctan2(point_xy[1] - center[1], point_xy[0] - center[0])))

    start_angle = angle_deg(p1)
    end_angle = angle_deg(p3)
    if end_angle < start_angle:
        end_angle += 360.0
    cv2.ellipse(
        image_bgr,
        center=(int(round(center[0])), int(round(center[1]))),
        axes=(int(round(radius)), int(round(radius))),
        angle=0,
        startAngle=int(round(start_angle)),
        endAngle=int(round(end_angle)),
        color=color_bgr,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _ellipse_point(
    center: tuple[int, int],
    axes: tuple[int, int],
    rotation_degrees: float,
    theta_degrees: float,
) -> tuple[int, int]:
    cx, cy = center
    a, b = axes
    phi = np.deg2rad(rotation_degrees)
    t = np.deg2rad(theta_degrees)
    x = cx + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
    y = cy + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
    return int(round(x)), int(round(y))


def _draw_dashed_ellipse(
    image_bgr: np.ndarray,
    center: tuple[int, int],
    axes: tuple[int, int],
    rotation_degrees: float,
    start_degrees: float,
    end_degrees: float,
    color_bgr: tuple[int, int, int],
    thickness: int,
    dash_length_degrees: float = 12.0,
    gap_length_degrees: float = 8.0,
    detail_degrees: float = 2.0,
) -> None:
    angle = start_degrees
    while angle < end_degrees:
        dash_start = angle
        dash_end = min(angle + dash_length_degrees, end_degrees)
        theta = dash_start
        prev_point = _ellipse_point(center, axes, rotation_degrees, theta)
        theta += detail_degrees
        while theta <= dash_end:
            cur_point = _ellipse_point(center, axes, rotation_degrees, theta)
            cv2.line(image_bgr, prev_point, cur_point, color_bgr, thickness, cv2.LINE_AA)
            prev_point = cur_point
            theta += detail_degrees
        angle = dash_end + gap_length_degrees


def _render_colorful_template_court(template: dict[str, Any]) -> np.ndarray:
    resolved = _template_with_defaults(template)
    colors = {name: _hex_to_rgb(hex_value) for name, hex_value in resolved["colors"].items()}
    background_rgb = _hex_to_rgb(str(resolved["background_hex"]))
    paint_fill_rgb = _hex_to_rgb(str(resolved["paint_fill_hex"]))
    scale = float(resolved["scale"])
    padding = int(resolved["padding"])
    thickness = int(resolved["line_thickness"])
    league = str(resolved["league"])
    measurement_unit = str(resolved["measurement_unit"])
    measurements = _court_measurements(league=league, measurement_unit=measurement_unit)
    vertices = _court_vertices(league=league, measurement_unit=measurement_unit)

    canvas_width = int(round(measurements["court_length"] * scale)) + padding * 2
    canvas_height = int(round(measurements["court_width"] * scale)) + padding * 2
    image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    image[:, :] = background_rgb
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    left_paint = np.asarray([_to_pixel(vertices[i], scale, padding) for i in [2, 3, 11, 9]], dtype=np.int32)
    right_paint = np.asarray([_to_pixel(vertices[i], scale, padding) for i in [29, 30, 23, 21]], dtype=np.int32)
    cv2.fillPoly(image_bgr, [left_paint], _rgb_to_bgr(paint_fill_rgb))
    cv2.fillPoly(image_bgr, [right_paint], _rgb_to_bgr(paint_fill_rgb))

    base_bgr = _rgb_to_bgr(colors["base"])
    for start_idx, end_idx in COURT_EDGES:
        cv2.line(
            image_bgr,
            _to_pixel(vertices[start_idx], scale, padding),
            _to_pixel(vertices[end_idx], scale, padding),
            base_bgr,
            thickness,
            cv2.LINE_AA,
        )

    for start_idx, end_idx in [(1, 7), (4, 8), (28, 24), (31, 25), (18, 27), (20, 32), (28, 31)]:
        cv2.line(
            image_bgr,
            _to_pixel(vertices[start_idx], scale, padding),
            _to_pixel(vertices[end_idx], scale, padding),
            base_bgr,
            thickness,
            cv2.LINE_AA,
        )

    throw_line_specs = [
        (vertices[12], (vertices[12][0], vertices[12][1] + measurements["baseline_to_throw_line_length"])),
        (vertices[14], (vertices[14][0], vertices[14][1] - measurements["baseline_to_throw_line_length"])),
        (vertices[18], (vertices[18][0], vertices[18][1] + measurements["baseline_to_throw_line_length"])),
        (vertices[20], (vertices[20][0], vertices[20][1] - measurements["baseline_to_throw_line_length"])),
    ]
    for start_point, end_point in throw_line_specs:
        cv2.line(
            image_bgr,
            _to_pixel(start_point, scale, padding),
            _to_pixel(end_point, scale, padding),
            base_bgr,
            thickness,
            cv2.LINE_AA,
        )

    center_circle_radius_px = int(round(measurements["center_circle_radius"] * scale))
    cv2.circle(
        image_bgr,
        _to_pixel(vertices[16], scale, padding),
        center_circle_radius_px,
        base_bgr,
        thickness,
        cv2.LINE_AA,
    )

    if measurement_unit == "feet":
        backboard_to_rim = BACKBOARD_TO_RIM_CM / 30.48
        backboard_span = BACKBOARD_SPAN_CM / 30.48
    else:
        backboard_to_rim = BACKBOARD_TO_RIM_CM
        backboard_span = BACKBOARD_SPAN_CM

    left_baseline_x = _to_pixel((0.0, 0.0), scale, padding)[0]
    right_baseline_x = _to_pixel((measurements["court_length"], 0.0), scale, padding)[0]
    restricted_radius_px = int(round(measurements["restricted_area_radius"] * scale))
    for side, basket_idx, free_idx, arc_indices, start_angle, end_angle, dashed_start, dashed_end in [
        ("left", 6, 10, [7, 13, 8], 180, 360, 0, 180),
        ("right", 26, 22, [25, 19, 24], 0, 180, 180, 360),
    ]:
        basket_xy = vertices[basket_idx]
        basket_px = _to_pixel(basket_xy, scale, padding)
        rim_radius_px = int(round((measurements["rim_diameter"] / 2.0) * scale))
        cv2.circle(image_bgr, basket_px, rim_radius_px, base_bgr, thickness, cv2.LINE_AA)
        cv2.ellipse(
            image_bgr,
            center=basket_px,
            angle=90,
            axes=(restricted_radius_px, restricted_radius_px),
            startAngle=start_angle,
            endAngle=end_angle,
            color=base_bgr,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        free_px = _to_pixel(vertices[free_idx], scale, padding)
        cv2.ellipse(
            image_bgr,
            center=free_px,
            angle=90,
            axes=(center_circle_radius_px, center_circle_radius_px),
            startAngle=start_angle,
            endAngle=end_angle,
            color=base_bgr,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        _draw_dashed_ellipse(
            image_bgr,
            center=free_px,
            axes=(center_circle_radius_px, center_circle_radius_px),
            rotation_degrees=90,
            start_degrees=dashed_start,
            end_degrees=dashed_end,
            color_bgr=base_bgr,
            thickness=thickness,
        )

        arc_points = [_to_pixel(vertices[index], scale, padding) for index in arc_indices]
        _draw_circular_arc_from_three_points(
            image_bgr,
            arc_points[0],
            arc_points[1],
            arc_points[2],
            base_bgr,
            thickness,
        )

        x_free = free_px[0]
        y_top = free_px[1] - center_circle_radius_px
        y_bottom = free_px[1] + center_circle_radius_px
        if side == "left":
            cv2.line(image_bgr, (left_baseline_x, y_top), (x_free, y_top), base_bgr, thickness, cv2.LINE_AA)
            cv2.line(image_bgr, (left_baseline_x, y_bottom), (x_free, y_bottom), base_bgr, thickness, cv2.LINE_AA)
            backboard_x = basket_xy[0] - backboard_to_rim
        else:
            cv2.line(image_bgr, (x_free, y_top), (right_baseline_x, y_top), base_bgr, thickness, cv2.LINE_AA)
            cv2.line(image_bgr, (x_free, y_bottom), (right_baseline_x, y_bottom), base_bgr, thickness, cv2.LINE_AA)
            backboard_x = basket_xy[0] + backboard_to_rim
        half_span = backboard_span / 2.0
        cv2.line(
            image_bgr,
            _to_pixel((backboard_x, basket_xy[1] - half_span), scale, padding),
            _to_pixel((backboard_x, basket_xy[1] + half_span), scale, padding),
            base_bgr,
            max(2, thickness * 2),
            cv2.LINE_AA,
        )

    for edge_index, (start_idx, end_idx) in enumerate(COURT_EDGES):
        cv2.line(
            image_bgr,
            _to_pixel(vertices[start_idx], scale, padding),
            _to_pixel(vertices[end_idx], scale, padding),
            _rgb_to_bgr(court_edge_color_rgb(edge_index, resolved)),
            thickness,
            cv2.LINE_AA,
        )

    for group_name, (start_idx, end_idx) in COURT_SPECIAL_SEGMENTS:
        cv2.line(
            image_bgr,
            _to_pixel(vertices[start_idx], scale, padding),
            _to_pixel(vertices[end_idx], scale, padding),
            _rgb_to_bgr(court_group_color_rgb(group_name, resolved)),
            thickness,
            cv2.LINE_AA,
        )

    cv2.circle(
        image_bgr,
        _to_pixel(vertices[16], scale, padding),
        center_circle_radius_px,
        _rgb_to_bgr(colors["center"]),
        thickness,
        cv2.LINE_AA,
    )
    for basket_idx, free_idx in [(6, 10), (26, 22)]:
        basket_px = _to_pixel(vertices[basket_idx], scale, padding)
        free_px = _to_pixel(vertices[free_idx], scale, padding)
        rim_radius_px = int(round((measurements["rim_diameter"] / 2.0) * scale))
        cv2.circle(image_bgr, basket_px, rim_radius_px, _rgb_to_bgr(colors["rim"]), thickness, cv2.LINE_AA)
        cv2.circle(image_bgr, basket_px, restricted_radius_px, _rgb_to_bgr(colors["restricted"]), thickness, cv2.LINE_AA)
        cv2.circle(image_bgr, free_px, center_circle_radius_px, _rgb_to_bgr(colors["lane"]), thickness, cv2.LINE_AA)
        cv2.ellipse(
            image_bgr,
            center=free_px,
            angle=90,
            axes=(center_circle_radius_px, center_circle_radius_px),
            startAngle=180 if basket_idx == 6 else 0,
            endAngle=360 if basket_idx == 6 else 180,
            color=_rgb_to_bgr(colors["lane"]),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def render_bev_court(spec_path: str | Path) -> np.ndarray:
    spec = load_json(spec_path)
    template = spec.get("template")
    if template is not None:
        return _render_colorful_template_court(template)
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
