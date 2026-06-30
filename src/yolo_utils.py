from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .cv_utils import bottom_center, draw_boxes, draw_points, ensure_dir, save_json
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

PLAYER_CLASS_NAMES = {
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
}


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


def court_keypoints_from_result(
    result: Any,
    bev_shape: tuple[int, int, int],
    *,
    frame_index: int = 0,
    anchor_confidence: float = 0.35,
    league: str = "nba",
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

    vertices = court_vertices_bev(bev_shape, league=league)
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
                label=COURT_LABELS[index],
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
) -> np.ndarray | None:
    if len(keypoints) < 4:
        return None
    src = np.asarray([kp.image_xy for kp in keypoints], dtype=np.float32)
    dst = np.asarray([kp.bev_xy for kp in keypoints], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold)
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
) -> tuple[list[CourtKeypointRecord], np.ndarray | None, Any]:
    model = load_yolo_model(model_path)
    result = model.predict(image_rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
    keypoints = court_keypoints_from_result(
        result,
        bev_shape,
        frame_index=frame_index,
        anchor_confidence=anchor_confidence,
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
        vis_rgb = draw_detection_records(frame_rgb, detections, max_items=20)
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
    if paths:
        for tid, pts in paths.items():
            if len(pts) < 2:
                continue
            pts_int = np.asarray(pts, dtype=np.float32).round().astype(np.int32)
            cv2.polylines(canvas, [pts_int.reshape(-1, 1, 2)], False, (80, 80, 80), 2, cv2.LINE_AA)
    if len(points_xy):
        canvas = draw_points(canvas, points_xy, labels=labels, color=(255, 80, 80), radius=7)
    return canvas


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
) -> tuple[Path, list[dict[str, Any]]]:
    detector = load_yolo_model(detector_path)
    court_model = load_yolo_model(court_model_path)
    bev_base = render_bev_court(bev_spec_path)
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
    frame_index = 0
    last_H: np.ndarray | None = None
    while frame_index < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
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
        )
        current_H = estimate_homography_from_keypoints(keypoints)
        H = current_H if current_H is not None else last_H
        if H is not None:
            last_H = H
        feet = [bottom_center(det.bbox_xyxy) for det in players]
        bev_points = project_points(feet, H) if H is not None and feet else np.empty((0, 2))
        labels = [str(i) for i in range(len(players))]
        frame_vis = draw_detection_records(frame_rgb, players)
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
        frame_index += 1
    cap.release()
    writer.release()
    save_json(rows, output_path.with_suffix(".json"))
    return output_path, rows


def _detections_to_supervision(result: Any):
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
) -> tuple[Path, list[dict[str, Any]]]:
    import supervision as sv

    detector = load_yolo_model(detector_path)
    court_model = load_yolo_model(court_model_path)
    tracker = sv.ByteTrack()
    bev_base = render_bev_court(bev_spec_path)
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
    frame_index = 0
    last_H: np.ndarray | None = None
    while frame_index < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        det_result = detector.predict(frame_rgb, conf=detector_conf, imgsz=imgsz, verbose=False)[0]
        tracked = tracker.update_with_detections(_detections_to_supervision(det_result))
        key_result = court_model.predict(frame_rgb, conf=keypoint_conf, imgsz=imgsz, verbose=False)[0]
        keypoints = court_keypoints_from_result(
            key_result,
            bev_base.shape,
            frame_index=frame_index,
            anchor_confidence=anchor_confidence,
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
        frame_vis = draw_detection_records(frame_rgb, frame_records)
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
        frame_index += 1
    cap.release()
    writer.release()
    save_json(records, output_path.with_suffix(".json"))
    return output_path, records
