from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .cv_utils import bottom_center, draw_boxes, draw_points, ensure_dir, load_json, save_json
from .geometry_utils import project_points, render_bev_court

BASKETBALL_CLASSES = [
    "ball",
    "ball-in-basket",
    "number",
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
    "referee",
    "rim",
]

COURT_LABELS = [
    "01",
    "02",
    "04",
    "05",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "19",
    "21",
    "23",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "37",
    "38",
    "40",
    "41",
]

COURT_KEYPOINT_DISPLAY_NAMES = [
    "left_baseline_top",
    "left_corner_three_top",
    "left_paint_baseline_top",
    "left_paint_baseline_bottom",
    "left_corner_three_bottom",
    "left_baseline_bottom",
    "left_rim",
    "left_wing_three_top",
    "left_wing_three_bottom",
    "left_paint_ft_top",
    "left_free_throw",
    "left_paint_ft_bottom",
    "left_throw_line_top",
    "left_arc_apex",
    "left_throw_line_bottom",
    "midcourt_top",
    "center_court",
    "midcourt_bottom",
    "right_throw_line_top",
    "right_arc_apex",
    "right_throw_line_bottom",
    "right_paint_ft_top",
    "right_free_throw",
    "right_paint_ft_bottom",
    "right_wing_three_top",
    "right_wing_three_bottom",
    "right_rim",
    "right_baseline_top",
    "right_corner_three_top",
    "right_paint_baseline_top",
    "right_paint_baseline_bottom",
    "right_corner_three_bottom",
    "right_baseline_bottom",
]

COURT_KEYPOINT_NAME_BY_LABEL = dict(zip(COURT_LABELS, COURT_KEYPOINT_DISPLAY_NAMES))

PLAYER_CLASS_NAMES = {
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
}


def display_keypoint_name(label: str) -> str:
    return COURT_KEYPOINT_NAME_BY_LABEL.get(str(label), str(label))


@dataclass(frozen=True)
class DetectionRecord:
    frame_index: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    track_id: int | None = None


@dataclass(frozen=True)
class CourtKeypointRecord:
    frame_index: int
    index: int
    label: str
    confidence: float
    image_xy: list[float]
    bev_xy: list[float]


def detector_model_path(course_root: str | Path) -> Path:
    return Path(course_root) / "assets" / "models" / "detectors" / "yolo26n_basketball_player_best.pt"


def court_keypoint_model_path(course_root: str | Path) -> Path:
    return (
        Path(course_root)
        / "assets"
        / "models"
        / "court_keypoints"
        / "yolo26n_basketball_court_pose_best.pt"
    )


def reference_videos(course_root: str | Path) -> list[Path]:
    folder = Path(course_root) / "assets" / "raw" / "reference_videos"
    if not folder.exists():
        return []
    exts = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in exts)


def first_reference_video(course_root: str | Path) -> Path:
    videos = reference_videos(course_root)
    if not videos:
        raise FileNotFoundError(
            "找不到參考影片。請確認 assets/raw/reference_videos/ 內至少有一支 mp4。"
        )
    return videos[0]


def load_yolo_model(model_path: str | Path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    from ultralytics import YOLO

    return YOLO(str(model_path))


def read_video_frame(video_path: str | Path, frame_index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"無法讀取 frame {frame_index}: {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def mp4_fourcc() -> int:
    return int(getattr(cv2, "VideoWriter_fourcc")(*"mp4v"))


def rgb_from_ultralytics_plot(result: Any) -> np.ndarray:
    plotted_bgr = result.plot()
    return cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)


def _result_names(result: Any) -> dict[int, str]:
    names = getattr(result, "names", {}) or {}
    return {int(k): str(v) for k, v in dict(names).items()}


def detections_from_result(result: Any, frame_index: int = 0) -> list[DetectionRecord]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    names = _result_names(result)
    records: list[DetectionRecord] = []
    for i in range(len(boxes)):
        box = boxes[i]
        xyxy = [float(v) for v in box.xyxy[0].tolist()]
        class_id = int(box.cls[0].item())
        records.append(
            DetectionRecord(
                frame_index=frame_index,
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
                confidence=float(box.conf[0].item()),
                bbox_xyxy=xyxy,
            )
        )
    return records


def run_detector_on_image(
    model_path: str | Path,
    image_rgb: np.ndarray,
    *,
    conf: float = 0.25,
    imgsz: int = 960,
    frame_index: int = 0,
) -> tuple[list[DetectionRecord], Any]:
    model = load_yolo_model(model_path)
    result = model.predict(image_rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
    return detections_from_result(result, frame_index=frame_index), result


def draw_detection_records(
    image_rgb: np.ndarray,
    detections: Sequence[DetectionRecord],
    *,
    class_names: Iterable[str] | None = None,
    max_items: int | None = None,
) -> np.ndarray:
    keep = list(detections)
    if class_names is not None:
        allowed = set(class_names)
        keep = [det for det in keep if det.class_name in allowed]
    if max_items is not None:
        keep = keep[:max_items]
    labels = [
        (
            f"ID {det.track_id} {det.class_name} {det.confidence:.2f}"
            if det.track_id is not None
            else f"{det.class_name} {det.confidence:.2f}"
        )
        for det in keep
    ]
    return draw_boxes(image_rgb, [det.bbox_xyxy for det in keep], labels)


def court_vertices_normalized(league: str = "nba") -> list[tuple[float, float]]:
    preset = {
        "nba": (2865.0, 1524.0, 724.0, 424.0, 91.0, 488.0, 579.0, 160.0, 835.0),
        "fiba": (2800.0, 1500.0, 675.0, 330.0, 90.0, 490.0, 580.0, 157.0, 830.0),
    }.get(league)
    if preset is None:
        raise ValueError(f"Unsupported basketball court league: {league}")
    length, width, arc, straight, corner, paint_width, paint_length, rim, throw = preset
    paint_start = (width - paint_width) / 2.0
    center_y = width / 2.0
    raw = [
        (0.0, 0.0),
        (0.0, corner),
        (0.0, paint_start),
        (0.0, paint_start + paint_width),
        (0.0, width - corner),
        (0.0, width),
        (rim, center_y),
        (straight, corner),
        (straight, width - corner),
        (paint_length, paint_start),
        (paint_length, center_y),
        (paint_length, paint_start + paint_width),
        (throw, 0.0),
        (rim + arc, center_y),
        (throw, width),
        (length / 2.0, 0.0),
        (length / 2.0, center_y),
        (length / 2.0, width),
        (length - throw, 0.0),
        (length - rim - arc, center_y),
        (length - throw, width),
        (length - paint_length, paint_start),
        (length - paint_length, center_y),
        (length - paint_length, paint_start + paint_width),
        (length - straight, corner),
        (length - straight, width - corner),
        (length - rim, center_y),
        (length, 0.0),
        (length, corner),
        (length, paint_start),
        (length, paint_start + paint_width),
        (length, width - corner),
        (length, width),
    ]
    return [(x / length, y / width) for x, y in raw]


def court_vertices_bev(bev_shape: tuple[int, int, int], league: str = "nba") -> np.ndarray:
    height, width = bev_shape[:2]
    return np.asarray(
        [(x * (width - 1), y * (height - 1)) for x, y in court_vertices_normalized(league)],
        dtype=np.float32,
    )


def bev_court_bounds(spec_path: str | Path) -> tuple[float, float, float, float]:
    spec = load_json(spec_path)
    for item in spec.get("polylines", []):
        if item.get("name") == "outer_boundary":
            points = np.asarray(item["points"], dtype=np.float32)
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            return float(x1), float(y1), float(x2), float(y2)
    canvas = spec["canvas"]
    return 0.0, 0.0, float(canvas["width"] - 1), float(canvas["height"] - 1)


def court_vertices_bev_in_bounds(
    bounds_xyxy: tuple[float, float, float, float],
    league: str = "nba",
) -> np.ndarray:
    x1, y1, x2, y2 = bounds_xyxy
    return np.asarray(
        [(x1 + x * (x2 - x1), y1 + y * (y2 - y1)) for x, y in court_vertices_normalized(league)],
        dtype=np.float32,
    )


def court_keypoints_from_result(
    result: Any,
    bev_shape: tuple[int, int, int],
    *,
    frame_index: int = 0,
    anchor_confidence: float = 0.35,
    league: str = "nba",
    court_bounds: tuple[float, float, float, float] | None = None,
) -> list[CourtKeypointRecord]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or getattr(keypoints, "xy", None) is None:
        return []
    xy = keypoints.xy.detach().cpu().numpy()
    conf = None
    if getattr(keypoints, "conf", None) is not None:
        conf = keypoints.conf.detach().cpu().numpy()
    if len(xy) == 0:
        return []

    vertices = (
        court_vertices_bev_in_bounds(court_bounds, league=league)
        if court_bounds is not None
        else court_vertices_bev(bev_shape, league=league)
    )
    records: list[CourtKeypointRecord] = []
    for index, point in enumerate(xy[0]):
        if index >= len(COURT_LABELS) or index >= len(vertices):
            break
        confidence = float(conf[0][index]) if conf is not None else 1.0
        if confidence < anchor_confidence:
            continue
        records.append(
            CourtKeypointRecord(
                frame_index=frame_index,
                index=index,
                label=display_keypoint_name(COURT_LABELS[index]),
                confidence=confidence,
                image_xy=[float(point[0]), float(point[1])],
                bev_xy=[float(vertices[index, 0]), float(vertices[index, 1])],
            )
        )
    return records


def estimate_homography_from_keypoints(
    keypoints: Sequence[CourtKeypointRecord],
    *,
    ransac_threshold: float = 5.0,
    min_points: int = 6,
    max_median_error: float = 35.0,
) -> np.ndarray | None:
    if len(keypoints) < min_points:
        return None
    src = np.asarray([kp.image_xy for kp in keypoints], dtype=np.float32)
    dst = np.asarray([kp.bev_xy for kp in keypoints], dtype=np.float32)
    if np.linalg.matrix_rank(src - src.mean(axis=0)) < 2:
        return None
    if np.linalg.matrix_rank(dst - dst.mean(axis=0)) < 2:
        return None
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold)
    if H is None or mask is None:
        return None
    inlier_count = int(mask.ravel().sum())
    if inlier_count < min_points:
        return None
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    inlier_errors = errors[mask.ravel().astype(bool)]
    if len(inlier_errors) == 0 or float(np.median(inlier_errors)) > max_median_error:
        return None
    return H


def run_court_keypoints_on_image(
    model_path: str | Path,
    image_rgb: np.ndarray,
    bev_shape: tuple[int, int, int],
    *,
    conf: float = 0.25,
    anchor_confidence: float = 0.35,
    imgsz: int = 960,
    frame_index: int = 0,
    court_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[list[CourtKeypointRecord], np.ndarray | None, Any]:
    model = load_yolo_model(model_path)
    result = model.predict(image_rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
    keypoints = court_keypoints_from_result(
        result,
        bev_shape,
        frame_index=frame_index,
        anchor_confidence=anchor_confidence,
        court_bounds=court_bounds,
    )
    return keypoints, estimate_homography_from_keypoints(keypoints), result


def records_to_dicts(records: Sequence[DetectionRecord | CourtKeypointRecord]) -> list[dict[str, Any]]:
    return [record.__dict__ for record in records]


def write_detection_preview_video(
    *,
    video_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    max_frames: int = 45,
    conf: float = 0.25,
    imgsz: int = 960,
) -> tuple[Path, list[dict[str, Any]]]:
    model = load_yolo_model(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    writer = cv2.VideoWriter(
        str(output_path),
        mp4_fourcc(),
        fps,
        (width, height),
    )
    all_records: list[dict[str, Any]] = []
    frame_index = 0
    while frame_index < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = model.predict(frame_rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
        detections = detections_from_result(result, frame_index=frame_index)
        all_records.extend(records_to_dicts(detections))
        vis_rgb = rgb_from_ultralytics_plot(result)
        writer.write(cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
        frame_index += 1
    cap.release()
    writer.release()
    save_json(all_records, output_path.with_suffix(".json"))
    return output_path, all_records


def _player_detections(records: Sequence[DetectionRecord]) -> list[DetectionRecord]:
    return [det for det in records if det.class_name in PLAYER_CLASS_NAMES]


def _draw_bev_players(
    bev_rgb: np.ndarray,
    points_xy: np.ndarray,
    labels: Sequence[str],
    paths: dict[int, list[tuple[float, float]]] | None = None,
) -> np.ndarray:
    canvas = bev_rgb.copy()
    overlay = canvas.copy()
    if paths:
        for tid, pts in paths.items():
            if len(pts) < 2:
                continue
            color = _track_color(tid)
            pts_int = np.asarray(pts, dtype=np.float32).round().astype(np.int32)
            cv2.polylines(overlay, [pts_int.reshape(-1, 1, 2)], False, color, 3, cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0)
    for i, (point, label) in enumerate(zip(points_xy, labels)):
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        tid = int(label) if str(label).isdigit() else i
        color = _track_color(tid)
        cv2.circle(canvas, (x, y), 11, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 8, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(label),
            (x + 12, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(label),
            (x + 12, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _track_color(track_id: int) -> tuple[int, int, int]:
    palette = [
        (41, 128, 255),
        (255, 99, 72),
        (46, 204, 113),
        (155, 89, 182),
        (241, 196, 15),
        (26, 188, 156),
        (231, 76, 60),
        (52, 152, 219),
    ]
    return palette[abs(int(track_id)) % len(palette)]


def _draw_keypoints_overlay(
    image_rgb: np.ndarray,
    keypoints: Sequence[CourtKeypointRecord],
) -> np.ndarray:
    image = image_rgb.copy()
    for kp in keypoints:
        x, y = int(round(kp.image_xy[0])), int(round(kp.image_xy[1]))
        cv2.circle(image, (x, y), 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 5, (46, 204, 113), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            kp.label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            kp.label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (46, 204, 113),
            1,
            cv2.LINE_AA,
        )
    return image


def write_detector_keypoint_bev_video(
    *,
    video_path: str | Path,
    detector_path: str | Path,
    court_model_path: str | Path,
    bev_spec_path: str | Path,
    output_path: str | Path,
    max_frames: int = 30,
    detector_conf: float = 0.25,
    keypoint_conf: float = 0.25,
    anchor_confidence: float = 0.35,
    imgsz: int = 960,
    start_frame: int = 0,
) -> tuple[Path, list[dict[str, Any]]]:
    detector = load_yolo_model(detector_path)
    court_model = load_yolo_model(court_model_path)
    bev_base = render_bev_court(bev_spec_path)
    court_bounds = bev_court_bounds(bev_spec_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    writer = cv2.VideoWriter(
        str(output_path),
        mp4_fourcc(),
        fps,
        (frame_width * 2, frame_height),
    )
    rows: list[dict[str, Any]] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    local_frame_index = 0
    last_H: np.ndarray | None = None
    while local_frame_index < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_index = start_frame + local_frame_index
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        det_result = detector.predict(frame_rgb, conf=detector_conf, imgsz=imgsz, verbose=False)[0]
        detections = detections_from_result(det_result, frame_index=frame_index)
        players = _player_detections(detections)
        key_result = court_model.predict(frame_rgb, conf=keypoint_conf, imgsz=imgsz, verbose=False)[0]
        keypoints = court_keypoints_from_result(
            key_result,
            bev_base.shape,
            frame_index=frame_index,
            anchor_confidence=anchor_confidence,
            court_bounds=court_bounds,
        )
        current_H = estimate_homography_from_keypoints(keypoints)
        H = current_H if current_H is not None else last_H
        if H is not None:
            last_H = H
        feet = [bottom_center(det.bbox_xyxy) for det in players]
        bev_points = project_points(feet, H) if H is not None and feet else np.empty((0, 2))
        labels = [str(i) for i in range(len(players))]
        frame_vis = rgb_from_ultralytics_plot(det_result)
        frame_vis = _draw_keypoints_overlay(frame_vis, keypoints)
        if feet:
            frame_vis = draw_points(frame_vis, feet, labels=labels, color=(40, 120, 255), radius=6)
        bev_vis = _draw_bev_players(bev_base, bev_points, labels)
        bev_vis = cv2.resize(bev_vis, (frame_width, frame_height))
        writer.write(cv2.cvtColor(np.hstack([frame_vis, bev_vis]), cv2.COLOR_RGB2BGR))
        for label, det, foot, bev_xy in zip(labels, players, feet, bev_points):
            rows.append(
                {
                    "frame": frame_index,
                    "label": label,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "bbox_xyxy": det.bbox_xyxy,
                    "foot_x": float(foot[0]),
                    "foot_y": float(foot[1]),
                    "bev_x": float(bev_xy[0]),
                    "bev_y": float(bev_xy[1]),
                    "keypoint_count": len(keypoints),
                }
            )
        local_frame_index += 1
    cap.release()
    writer.release()
    save_json(rows, output_path.with_suffix(".json"))
    return output_path, rows


def _detections_to_supervision(result: Any) -> Any:
    import supervision as sv

    detections = sv.Detections.from_ultralytics(result)
    names = _result_names(result)
    if len(detections) == 0:
        detections.data["class_name"] = np.asarray([], dtype=str)
        return detections
    class_ids = detections.class_id if detections.class_id is not None else np.zeros(len(detections), dtype=int)
    class_names = np.asarray([names.get(int(class_id), str(class_id)) for class_id in class_ids])
    detections.data["class_name"] = class_names
    mask = np.isin(class_names, list(PLAYER_CLASS_NAMES))
    return detections[mask]


def write_bytetrack_bev_video(
    *,
    video_path: str | Path,
    detector_path: str | Path,
    court_model_path: str | Path,
    bev_spec_path: str | Path,
    output_path: str | Path,
    max_frames: int = 45,
    detector_conf: float = 0.25,
    keypoint_conf: float = 0.25,
    anchor_confidence: float = 0.35,
    imgsz: int = 960,
    start_frame: int = 0,
) -> tuple[Path, list[dict[str, Any]]]:
    detector = load_yolo_model(detector_path)
    court_model = load_yolo_model(court_model_path)
    bev_base = render_bev_court(bev_spec_path)
    court_bounds = bev_court_bounds(bev_spec_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    writer = cv2.VideoWriter(
        str(output_path),
        mp4_fourcc(),
        fps,
        (frame_width * 2, frame_height),
    )
    records: list[dict[str, Any]] = []
    paths: dict[int, list[tuple[float, float]]] = {}
    local_frame_index = 0
    last_H: np.ndarray | None = None
    for source_frame_index, det_result in enumerate(
        detector.track(
            source=str(video_path),
            tracker="bytetrack.yaml",
            stream=True,
            persist=True,
            conf=detector_conf,
            imgsz=imgsz,
            verbose=False,
        )
    ):
        if source_frame_index < start_frame:
            continue
        if local_frame_index >= max_frames:
            break
        frame_index = source_frame_index
        frame_rgb = cv2.cvtColor(det_result.orig_img, cv2.COLOR_BGR2RGB)
        tracked = _detections_to_supervision(det_result)
        key_result = court_model.predict(frame_rgb, conf=keypoint_conf, imgsz=imgsz, verbose=False)[0]
        keypoints = court_keypoints_from_result(
            key_result,
            bev_base.shape,
            frame_index=frame_index,
            anchor_confidence=anchor_confidence,
            court_bounds=court_bounds,
        )
        current_H = estimate_homography_from_keypoints(keypoints)
        H = current_H if current_H is not None else last_H
        if H is not None:
            last_H = H

        xyxy = np.asarray(tracked.xyxy, dtype=float)
        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else np.arange(len(tracked))
        feet = [bottom_center(box) for box in xyxy]
        bev_points = project_points(feet, H) if H is not None and feet else np.empty((0, 2))
        labels = [str(int(tid)) for tid in tracker_ids]

        frame_records = [
            DetectionRecord(
                frame_index=frame_index,
                class_id=int(tracked.class_id[i]) if tracked.class_id is not None else -1,
                class_name=str(tracked.data.get("class_name", ["player"] * len(tracked))[i]),
                confidence=float(tracked.confidence[i]) if tracked.confidence is not None else 1.0,
                bbox_xyxy=[float(v) for v in xyxy[i].tolist()],
                track_id=int(tracker_ids[i]),
            )
            for i in range(len(tracked))
        ]
        frame_vis = rgb_from_ultralytics_plot(det_result)
        frame_vis = _draw_keypoints_overlay(frame_vis, keypoints)
        if feet:
            frame_vis = draw_points(frame_vis, feet, labels=labels, color=(40, 120, 255), radius=6)

        for tid, point in zip(tracker_ids, bev_points):
            paths.setdefault(int(tid), []).append((float(point[0]), float(point[1])))
        bev_vis = _draw_bev_players(bev_base, bev_points, labels, paths=paths)
        bev_vis = cv2.resize(bev_vis, (frame_width, frame_height))
        writer.write(cv2.cvtColor(np.hstack([frame_vis, bev_vis]), cv2.COLOR_RGB2BGR))

        for det, foot, bev_xy in zip(frame_records, feet, bev_points):
            records.append(
                {
                    "frame": frame_index,
                    "track_id": det.track_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "bbox_xyxy": det.bbox_xyxy,
                    "foot_x": float(foot[0]),
                    "foot_y": float(foot[1]),
                    "bev_x": float(bev_xy[0]),
                    "bev_y": float(bev_xy[1]),
                    "keypoint_count": len(keypoints),
                }
            )
        local_frame_index += 1
    cap.release()
    writer.release()
    save_json(records, output_path.with_suffix(".json"))
    return output_path, records
