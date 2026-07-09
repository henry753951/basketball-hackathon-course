from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, cast

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import requests

from src.cv_utils import side_by_side
from src.video_utils import ensure_notebook_playable_mp4, open_mp4_video_writer

_POSE_LANDMARKER_LITE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

_SKELETON_KEYS = ("hip", "shoulder", "elbow", "wrist", "knee", "ankle")
_SKELETON_EDGES = (
    ("hip", "shoulder"),
    ("shoulder", "elbow"),
    ("elbow", "wrist"),
    ("hip", "knee"),
    ("knee", "ankle"),
)


def _row_float(row: pd.Series, key: str) -> float:
    return float(cast(float, row.at[key]))


def angle_3pt(a, b, c) -> float:
    """Return angle ABC in degrees."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return float("nan")
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def synthetic_pose_sequence(n: int = 80) -> pd.DataFrame:
    """Create a tiny synthetic shooting-pose table so every notebook can run without a real video."""
    rows = []
    for f in range(n):
        t = f / max(1, n - 1)
        lift = math.sin(math.pi * min(1, t * 1.3))
        # simplified side-view joints
        hip = np.array([300, 250])
        shoulder = np.array([300, 160 - 25 * lift])
        elbow = np.array([345 + 35 * lift, 145 - 70 * lift])
        wrist = np.array([385 + 80 * lift, 130 - 115 * lift])
        knee = np.array([285 + 5 * lift, 330 - 20 * lift])
        ankle = np.array([285, 430])
        rows.append(
            {
                "frame": f,
                "hip_x": hip[0],
                "hip_y": hip[1],
                "shoulder_x": shoulder[0],
                "shoulder_y": shoulder[1],
                "elbow_x": elbow[0],
                "elbow_y": elbow[1],
                "wrist_x": wrist[0],
                "wrist_y": wrist[1],
                "knee_x": knee[0],
                "knee_y": knee[1],
                "ankle_x": ankle[0],
                "ankle_y": ankle[1],
            }
        )
    df = pd.DataFrame(rows)
    return add_pose_angles(df)


def mediapipe_pose_model_path(course_root: str | Path) -> Path:
    return Path(course_root) / "assets" / "models" / "mediapipe" / "pose_landmarker_lite.task"


def ensure_mediapipe_pose_model(course_root: str | Path) -> Path:
    model_path = mediapipe_pose_model_path(course_root)
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(_POSE_LANDMARKER_LITE_URL, timeout=120)
    response.raise_for_status()
    model_path.write_bytes(response.content)
    return model_path


def extract_pose_sequence_mediapipe_tasks(
    video_path: str | Path,
    *,
    course_root: str | Path,
    stride: int = 5,
    side: Literal["right", "left"] = "right",
    delegate: Literal["auto", "cpu", "gpu"] = "auto",
    min_pose_detection_confidence: float = 0.4,
    min_pose_presence_confidence: float = 0.4,
    min_tracking_confidence: float = 0.4,
) -> pd.DataFrame:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    if stride <= 0:
        raise ValueError(f"stride must be >= 1, got {stride}")

    model_path = ensure_mediapipe_pose_model(course_root)
    def make_options(mp_delegate: "BaseOptions.Delegate | None"):
        base_options = BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp_delegate,
        )
        return vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    delegate_candidates: list[tuple[str, "BaseOptions.Delegate | None"]]
    if delegate == "gpu":
        delegate_candidates = [("gpu", BaseOptions.Delegate.GPU)]
    elif delegate == "cpu":
        delegate_candidates = [("cpu", BaseOptions.Delegate.CPU)]
    else:
        delegate_candidates = [
            ("gpu", BaseOptions.Delegate.GPU),
            ("cpu", BaseOptions.Delegate.CPU),
        ]

    side_prefix = side.upper()
    landmark_names = {
        "shoulder": f"{side_prefix}_SHOULDER",
        "elbow": f"{side_prefix}_ELBOW",
        "wrist": f"{side_prefix}_WRIST",
        "hip": f"{side_prefix}_HIP",
        "knee": f"{side_prefix}_KNEE",
        "ankle": f"{side_prefix}_ANKLE",
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    rows: list[dict[str, float | int]] = []
    landmarker_errors: list[str] = []
    used_delegate = "cpu"
    for delegate_name, mp_delegate in delegate_candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        rows.clear()
        frame_idx = 0
        try:
            options = make_options(mp_delegate)
            with vision.PoseLandmarker.create_from_options(options) as landmarker:
                while cap.isOpened():
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    if frame_idx % stride != 0:
                        frame_idx += 1
                        continue

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    height, width = frame_rgb.shape[:2]
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    timestamp_ms = int(round(frame_idx * 1000.0 / max(fps, 1e-6)))
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks[0]

                        def pt(name: str) -> tuple[float, float]:
                            point = landmarks[int(getattr(vision.PoseLandmark, name))]
                            return point.x * width, point.y * height

                        shoulder = pt(landmark_names["shoulder"])
                        elbow = pt(landmark_names["elbow"])
                        wrist = pt(landmark_names["wrist"])
                        hip = pt(landmark_names["hip"])
                        knee = pt(landmark_names["knee"])
                        ankle = pt(landmark_names["ankle"])
                        rows.append(
                            {
                                "frame": frame_idx,
                                "shoulder_x": shoulder[0],
                                "shoulder_y": shoulder[1],
                                "elbow_x": elbow[0],
                                "elbow_y": elbow[1],
                                "wrist_x": wrist[0],
                                "wrist_y": wrist[1],
                                "hip_x": hip[0],
                                "hip_y": hip[1],
                                "knee_x": knee[0],
                                "knee_y": knee[1],
                                "ankle_x": ankle[0],
                                "ankle_y": ankle[1],
                            }
                        )
                    frame_idx += 1
            used_delegate = delegate_name
            break
        except RuntimeError as exc:
            landmarker_errors.append(f"{delegate_name}: {exc}")
            if delegate != "auto":
                raise
            continue
    cap.release()

    if not rows:
        if landmarker_errors:
            print("Pose delegate fallback log:")
            for msg in landmarker_errors:
                print("-", msg)
        raise RuntimeError(
            "MediaPipe Tasks Pose Landmarker 沒有從影片偵測到任何 pose landmarks。"
            "請確認 side.mp4 為清楚的側拍投籃影片，或調整拍攝角度、距離與光線。"
        )
    print(f"Pose landmarker delegate: {used_delegate}; stride={stride}")
    return add_pose_angles(pd.DataFrame(rows))


def add_pose_angles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    elbow_angles = []
    knee_angles = []
    shoulder_angles = []
    for _, r in out.iterrows():
        shoulder = (r["shoulder_x"], r["shoulder_y"])
        elbow = (r["elbow_x"], r["elbow_y"])
        wrist = (r["wrist_x"], r["wrist_y"])
        hip = (r["hip_x"], r["hip_y"])
        knee = (r["knee_x"], r["knee_y"])
        ankle = (r["ankle_x"], r["ankle_y"])
        elbow_angles.append(angle_3pt(shoulder, elbow, wrist))
        knee_angles.append(angle_3pt(hip, knee, ankle))
        shoulder_angles.append(angle_3pt(hip, shoulder, elbow))
    out["elbow_angle"] = elbow_angles
    out["knee_angle"] = knee_angles
    out["shoulder_angle"] = shoulder_angles
    return out


def draw_skeleton(width: int, height: int, row: pd.Series) -> np.ndarray:
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    pts = _skeleton_points(row)
    for a, b in _SKELETON_EDGES:
        cv2.line(img, pts[a], pts[b], (50, 50, 50), 3)
    for name, p in pts.items():
        cv2.circle(img, p, 6, (30, 120, 255), -1)
        cv2.putText(
            img,
            name,
            (p[0] + 8, p[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 30, 30),
            1,
        )
    elbow_angle = (
        _row_float(row, "elbow_angle") if "elbow_angle" in row.index else float("nan")
    )
    knee_angle = (
        _row_float(row, "knee_angle") if "knee_angle" in row.index else float("nan")
    )
    txt = f"elbow={elbow_angle:.1f}, knee={knee_angle:.1f}"
    cv2.putText(img, txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _skeleton_points(row: pd.Series) -> dict[str, tuple[int, int]]:
    return {
        "hip": (int(_row_float(row, "hip_x")), int(_row_float(row, "hip_y"))),
        "shoulder": (
            int(_row_float(row, "shoulder_x")),
            int(_row_float(row, "shoulder_y")),
        ),
        "elbow": (int(_row_float(row, "elbow_x")), int(_row_float(row, "elbow_y"))),
        "wrist": (int(_row_float(row, "wrist_x")), int(_row_float(row, "wrist_y"))),
        "knee": (int(_row_float(row, "knee_x")), int(_row_float(row, "knee_y"))),
        "ankle": (int(_row_float(row, "ankle_x")), int(_row_float(row, "ankle_y"))),
    }


def _maybe_import_supervision():
    try:
        import supervision as sv
    except ImportError:
        return None
    return sv


def _annotate_supervision(
    scene: np.ndarray,
    *,
    detections,
    labels: list[str] | None = None,
    trace_annotator=None,
    dot_annotator=None,
    label_annotator=None,
) -> np.ndarray:
    scene_out = scene

    def call_annotator(annotator, current_scene: np.ndarray, *, text_labels=None) -> np.ndarray:
        attempts: list[tuple[tuple[object, ...], dict[str, object]]] = []
        if text_labels is not None:
            attempts.extend(
                [
                    ((), {"scene": current_scene, "detections": detections, "labels": text_labels}),
                    ((current_scene, detections, text_labels), {}),
                ]
            )
        attempts.extend(
            [
                ((), {"scene": current_scene, "detections": detections}),
                ((current_scene, detections), {}),
            ]
        )
        for args, kwargs in attempts:
            try:
                return annotator.annotate(*args, **kwargs)
            except TypeError:
                continue
        return current_scene

    if trace_annotator is not None:
        scene_out = call_annotator(trace_annotator, scene_out)
    if dot_annotator is not None:
        scene_out = call_annotator(dot_annotator, scene_out)
    if label_annotator is not None and labels is not None:
        scene_out = call_annotator(label_annotator, scene_out, text_labels=labels)
    return scene_out


def _trace_detection(x: float, y: float, *, tracker_id: int = 1, box_size: int = 18):
    sv = _maybe_import_supervision()
    if sv is None:
        return None
    half = box_size / 2.0
    return sv.Detections(
        xyxy=np.array([[x - half, y - half, x + half, y + half]], dtype=np.float32),
        tracker_id=np.array([tracker_id], dtype=np.int32),
        class_id=np.array([0], dtype=np.int32),
    )


def _draw_pose_history(
    scene_rgb: np.ndarray,
    history_rows: list[pd.Series],
    *,
    current_row: pd.Series | None,
) -> np.ndarray:
    overlay = scene_rgb.copy()
    faded_rows = history_rows[:-1]
    for idx, row in enumerate(faded_rows[-3:]):
        intensity = 0.08 + 0.06 * idx
        line_color = (
            int(48 + 28 * idx),
            int(175 * intensity),
            int(220 * intensity),
        )
        point_color = (
            int(220 * intensity),
            int(160 * intensity),
            int(90 * intensity),
        )
        pts = _skeleton_points(row)
        for a, b in _SKELETON_EDGES:
            cv2.line(overlay, pts[a], pts[b], line_color, 1, cv2.LINE_AA)
        for point in pts.values():
            cv2.circle(overlay, point, 3, point_color, -1, cv2.LINE_AA)

    if current_row is not None:
        pts = _skeleton_points(current_row)
        for a, b in _SKELETON_EDGES:
            cv2.line(overlay, pts[a], pts[b], (40, 255, 190), 3, cv2.LINE_AA)
        for name, point in pts.items():
            radius = 7 if name == "wrist" else 5
            cv2.circle(overlay, point, radius, (255, 190, 70), -1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.76, scene_rgb, 0.24, 0)


def _draw_angle_label(
    image: np.ndarray,
    point: tuple[int, int],
    text: str,
    *,
    offset: tuple[int, int],
    bg_color: tuple[int, int, int],
    fg_color: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.58,
    thickness: int = 2,
) -> None:
    x = int(point[0] + offset[0])
    y = int(point[1] + offset[1])
    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    left = max(8, x - 8)
    top = max(8, y - text_h - 10)
    right = min(image.shape[1] - 8, left + text_w + 16)
    bottom = min(image.shape[0] - 8, top + text_h + baseline + 12)
    cv2.rectangle(image, (left, top), (right, bottom), bg_color, -1, cv2.LINE_AA)
    cv2.rectangle(image, (left, top), (right, bottom), (32, 32, 32), 1, cv2.LINE_AA)
    text_y = bottom - baseline - 6
    cv2.putText(
        image,
        text,
        (left + 8, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        fg_color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_pose_angle_labels(
    image: np.ndarray,
    row: pd.Series,
    pts: dict[str, tuple[int, int]],
    *,
    compact: bool = False,
) -> np.ndarray:
    elbow_angle = _row_float(row, "elbow_angle")
    knee_angle = _row_float(row, "knee_angle")
    shoulder_angle = _row_float(row, "shoulder_angle")
    labels = [
        ("elbow", f"E {elbow_angle:.1f}", (-18, -16), (52, 114, 255)),
        ("knee", f"K {knee_angle:.1f}", (-18, 34), (31, 166, 112)),
        ("shoulder", f"S {shoulder_angle:.1f}", (-88, -14), (240, 132, 48)),
    ]
    for joint_name, text, offset, bg_color in labels:
        joint_point = pts[joint_name]
        draw_offset = offset
        if compact:
            draw_offset = (int(offset[0] * 0.82), int(offset[1] * 0.82))
        _draw_angle_label(
            image,
            joint_point,
            text,
            offset=draw_offset,
            bg_color=bg_color,
            font_scale=0.52 if compact else 0.58,
            thickness=1 if compact else 2,
        )
    return image


def _draw_pose_side_panel(
    width: int,
    height: int,
    history_rows: list[pd.Series],
    *,
    current_row: pd.Series | None,
    trail_joint: str,
) -> np.ndarray:
    panel = np.ones((height, width, 3), dtype=np.uint8) * 255
    panel[:, :] = (250, 248, 243)
    cv2.putText(
        panel,
        "live skeleton",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )

    if current_row is None:
        cv2.putText(
            panel,
            "pose not detected",
            (24, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (120, 120, 120),
            2,
            cv2.LINE_AA,
        )
        return panel

    current_pts = _skeleton_points(current_row)
    xs = np.array([point[0] for point in current_pts.values()], dtype=np.float32)
    ys = np.array([point[1] for point in current_pts.values()], dtype=np.float32)
    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())
    src_w = max(1.0, max_x - min_x)
    src_h = max(1.0, max_y - min_y)

    content_left = 28
    content_right = width - 28
    content_top = 96
    content_bottom = height - 28
    content_w = max(1, content_right - content_left)
    content_h = max(1, content_bottom - content_top)

    scale = min(content_w / src_w, content_h / src_h) * 0.88
    target_cx = content_left + content_w * 0.50
    target_cy = content_top + content_h * 0.58
    src_cx = (min_x + max_x) / 2.0
    src_cy = (min_y + max_y) / 2.0

    def transform_point(point: tuple[int, int]) -> tuple[int, int]:
        x = int(round((point[0] - src_cx) * scale + target_cx))
        y = int(round((point[1] - src_cy) * scale + target_cy))
        return x, y

    normalized_pts = {name: transform_point(point) for name, point in current_pts.items()}

    faded_rows = history_rows[:-1]
    for idx, row in enumerate(faded_rows[-3:]):
        pts = {name: transform_point(point) for name, point in _skeleton_points(row).items()}
        intensity = 0.07 + 0.05 * idx
        line_color = (
            int(122 + 18 * idx),
            int(168 + 8 * idx),
            int(202 + 6 * idx),
        )
        point_color = (
            int(235 * intensity),
            int(175 * intensity),
            int(118 * intensity),
        )
        for a, b in _SKELETON_EDGES:
            cv2.line(panel, pts[a], pts[b], line_color, 1, cv2.LINE_AA)
        for point in pts.values():
            cv2.circle(panel, point, 3, point_color, -1, cv2.LINE_AA)

    for a, b in _SKELETON_EDGES:
        cv2.line(panel, normalized_pts[a], normalized_pts[b], (35, 50, 65), 5, cv2.LINE_AA)
        cv2.line(panel, normalized_pts[a], normalized_pts[b], (62, 214, 190), 3, cv2.LINE_AA)
    for name, point in normalized_pts.items():
        radius = 10 if name == "wrist" else 8
        cv2.circle(panel, point, radius, (255, 184, 72), -1, cv2.LINE_AA)
        cv2.circle(panel, point, radius + 1, (90, 72, 40), 1, cv2.LINE_AA)
    panel = _draw_pose_angle_labels(panel, current_row, normalized_pts)

    cv2.putText(
        panel,
        "live skeleton",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"frame {int(_row_float(current_row, 'frame'))}",
        (24, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (70, 70, 70),
        2,
        cv2.LINE_AA,
    )

    trail_points: list[tuple[int, int]] = []
    for row in history_rows[-10:]:
        trail_points.append(
            transform_point(
                (
                    int(_row_float(row, f"{trail_joint}_x")),
                    int(_row_float(row, f"{trail_joint}_y")),
                )
            )
        )

    if len(trail_points) >= 2:
        cv2.polylines(
            panel,
            [np.asarray(trail_points, dtype=np.int32).reshape(-1, 1, 2)],
            False,
            (255, 140, 60),
            3,
            cv2.LINE_AA,
        )
    if trail_points:
        cv2.circle(panel, trail_points[-1], 8, (255, 90, 40), -1, cv2.LINE_AA)

    return panel


def densify_pose_frames(pose_df: pd.DataFrame) -> pd.DataFrame:
    if pose_df.empty or "frame" not in pose_df.columns:
        return pose_df.copy()

    ordered = pose_df.sort_values("frame").drop_duplicates(subset="frame").reset_index(drop=True)
    full_frames = np.arange(
        int(cast(int, ordered["frame"].min())),
        int(cast(int, ordered["frame"].max())) + 1,
        dtype=int,
    )
    dense = ordered.set_index("frame").reindex(full_frames)
    numeric_columns = [col for col in dense.columns if pd.api.types.is_numeric_dtype(dense[col])]
    dense[numeric_columns] = dense[numeric_columns].interpolate(
        method="linear",
        limit_direction="both",
    )
    dense.index.name = "frame"
    dense = dense.reset_index()
    return dense


def render_pose_overlay_video(
    video_path: str | Path,
    pose_df: pd.DataFrame,
    output_path: str | Path,
    *,
    max_frames: int | None = None,
    trail_joint: str = "wrist",
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = None

    dense_pose_df = densify_pose_frames(pose_df)
    rows_by_frame = {
        int(cast(int, row["frame"])): row for _, row in dense_pose_df.iterrows()
    }
    history_rows: list[pd.Series] = []
    frame_idx = 0

    sv = _maybe_import_supervision()
    trace_annotator = None
    dot_annotator = None
    label_annotator = None
    if sv is not None:
        if hasattr(sv, "TraceAnnotator"):
            try:
                trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=20)
            except TypeError:
                trace_annotator = sv.TraceAnnotator()
        dot_annotator_cls = getattr(sv, "DotAnnotator", None)
        if dot_annotator_cls is not None:
            try:
                dot_annotator = dot_annotator_cls(radius=7)
            except TypeError:
                dot_annotator = dot_annotator_cls()
        label_annotator_cls = getattr(sv, "LabelAnnotator", None)
        if label_annotator_cls is not None:
            try:
                label_annotator = label_annotator_cls(text_scale=0.45, text_thickness=1)
            except TypeError:
                label_annotator = label_annotator_cls()

    while cap.isOpened():
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        current_row = rows_by_frame.get(frame_idx)
        if current_row is not None:
            history_rows.append(current_row)
            history_rows = history_rows[-5:]

        scene_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        scene_rgb = _draw_pose_history(scene_rgb, history_rows, current_row=current_row)
        if current_row is not None:
            joint_x = _row_float(current_row, f"{trail_joint}_x")
            joint_y = _row_float(current_row, f"{trail_joint}_y")
            detections = _trace_detection(joint_x, joint_y, tracker_id=11, box_size=18)
            if detections is not None:
                scene_rgb = _annotate_supervision(
                    scene_rgb,
                    detections=detections,
                    labels=[trail_joint],
                    trace_annotator=trace_annotator,
                    dot_annotator=dot_annotator,
                    label_annotator=label_annotator,
                )

        side_panel_width = max(320, int(height * 0.72))
        side_panel_rgb = _draw_pose_side_panel(
            side_panel_width,
            height,
            history_rows,
            current_row=current_row,
            trail_joint=trail_joint,
        )
        combined_rgb = side_by_side(scene_rgb, side_panel_rgb, max_width=width * 2)

        if writer is None:
            writer, _ = open_mp4_video_writer(
                output_path,
                fps=fps,
                frame_size=(combined_rgb.shape[1], combined_rgb.shape[0]),
            )
        writer.write(cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR))
        frame_idx += 1

    if writer is None:
        raise RuntimeError(f"無法從影片產生 overlay 預覽：{video_path}")
    writer.release()
    cap.release()
    return ensure_notebook_playable_mp4(output_path, overwrite=True)


def track_orange_ball(
    video_path: str | Path, max_frames: int | None = None
) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    rows = []
    frame_idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # A broad orange range. Students can tune these values for their own video.
        lower = np.array([5, 80, 80])
        upper = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 20:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    x = M["m10"] / M["m00"]
                    y = M["m01"] / M["m00"]
                    rows.append({"frame": frame_idx, "x": x, "y": y, "area": area})
        frame_idx += 1
    cap.release()
    df = pd.DataFrame(rows)
    if len(df) > 1:
        df["vx"] = df["x"].diff()
        df["vy"] = df["y"].diff()
        df["speed"] = np.sqrt(df["vx"].fillna(0) ** 2 + df["vy"].fillna(0) ** 2)
    return df


def estimate_release_frame(ball_df: pd.DataFrame) -> int | None:
    if ball_df.empty:
        return None
    has_speed = "speed" in ball_df.columns and bool(
        ball_df["speed"].notna().to_numpy().any()
    )
    if has_speed:
        # A simple proxy: release often occurs near first fast upward movement.
        candidate = ball_df.sort_values("speed", ascending=False).iloc[0]
        return int(_row_float(candidate, "frame"))
    # fallback: highest point in image coordinate means smallest y
    return int(_row_float(ball_df.sort_values("y").iloc[0], "frame"))


def render_ball_tracking_overlay_video(
    video_path: str | Path,
    ball_df: pd.DataFrame,
    output_path: str | Path,
    *,
    release_frame: int | None = None,
    max_frames: int | None = None,
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer, _ = open_mp4_video_writer(
        output_path,
        fps=fps,
        frame_size=(width, height),
    )

    rows_by_frame = {
        int(cast(int, row["frame"])): row for _, row in ball_df.sort_values("frame").iterrows()
    }
    sv = _maybe_import_supervision()
    trace_annotator = None
    dot_annotator = None
    label_annotator = None
    if sv is not None:
        if hasattr(sv, "TraceAnnotator"):
            try:
                trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=24)
            except TypeError:
                trace_annotator = sv.TraceAnnotator()
        dot_annotator_cls = getattr(sv, "DotAnnotator", None)
        if dot_annotator_cls is not None:
            try:
                dot_annotator = dot_annotator_cls(radius=7)
            except TypeError:
                dot_annotator = dot_annotator_cls()
        label_annotator_cls = getattr(sv, "LabelAnnotator", None)
        if label_annotator_cls is not None:
            try:
                label_annotator = label_annotator_cls(text_scale=0.45, text_thickness=1)
            except TypeError:
                label_annotator = label_annotator_cls()

    frame_idx = 0
    while cap.isOpened():
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        scene_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        row = rows_by_frame.get(frame_idx)
        if row is not None:
            x = _row_float(row, "x")
            y = _row_float(row, "y")
            detections = _trace_detection(x, y, tracker_id=21, box_size=20)
            if detections is not None:
                speed = _row_float(row, "speed") if "speed" in row.index and not pd.isna(row["speed"]) else 0.0
                scene_rgb = _annotate_supervision(
                    scene_rgb,
                    detections=detections,
                    labels=[f"ball {speed:.1f}px/f"],
                    trace_annotator=trace_annotator,
                    dot_annotator=dot_annotator,
                    label_annotator=label_annotator,
                )
            cv2.circle(scene_rgb, (int(round(x)), int(round(y))), 12, (255, 170, 0), 2, cv2.LINE_AA)

        if release_frame is not None and frame_idx == release_frame:
            cv2.putText(
                scene_rgb,
                "release frame",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(scene_rgb, (12, 12), (220, 52), (255, 90, 90), 2, cv2.LINE_AA)

        writer.write(cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR))
        frame_idx += 1

    writer.release()
    cap.release()
    return ensure_notebook_playable_mp4(output_path, overwrite=True)


def render_pose_and_ball_overlay_video(
    video_path: str | Path,
    pose_df: pd.DataFrame,
    ball_df: pd.DataFrame,
    output_path: str | Path,
    *,
    release_frame: int | None = None,
    max_frames: int | None = None,
    trail_joint: str = "wrist",
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = None

    dense_pose_df = densify_pose_frames(pose_df)
    pose_rows_by_frame = {
        int(cast(int, row["frame"])): row for _, row in dense_pose_df.iterrows()
    }
    ball_rows_by_frame = {
        int(cast(int, row["frame"])): row
        for _, row in ball_df.sort_values("frame").iterrows()
    }
    pose_history_rows: list[pd.Series] = []
    frame_idx = 0

    sv = _maybe_import_supervision()
    trace_annotator = None
    dot_annotator = None
    label_annotator = None
    if sv is not None:
        if hasattr(sv, "TraceAnnotator"):
            try:
                trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=24)
            except TypeError:
                trace_annotator = sv.TraceAnnotator()
        dot_annotator_cls = getattr(sv, "DotAnnotator", None)
        if dot_annotator_cls is not None:
            try:
                dot_annotator = dot_annotator_cls(radius=7)
            except TypeError:
                dot_annotator = dot_annotator_cls()
        label_annotator_cls = getattr(sv, "LabelAnnotator", None)
        if label_annotator_cls is not None:
            try:
                label_annotator = label_annotator_cls(text_scale=0.45, text_thickness=1)
            except TypeError:
                label_annotator = label_annotator_cls()

    while cap.isOpened():
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        current_pose_row = pose_rows_by_frame.get(frame_idx)
        current_ball_row = ball_rows_by_frame.get(frame_idx)
        if current_pose_row is not None:
            pose_history_rows.append(current_pose_row)
            pose_history_rows = pose_history_rows[-5:]

        scene_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        scene_rgb = _draw_pose_history(
            scene_rgb, pose_history_rows, current_row=current_pose_row
        )

        if current_pose_row is not None:
            joint_x = _row_float(current_pose_row, f"{trail_joint}_x")
            joint_y = _row_float(current_pose_row, f"{trail_joint}_y")
            joint_detections = _trace_detection(
                joint_x, joint_y, tracker_id=11, box_size=18
            )
            if joint_detections is not None:
                scene_rgb = _annotate_supervision(
                    scene_rgb,
                    detections=joint_detections,
                    labels=[trail_joint],
                    trace_annotator=trace_annotator,
                    dot_annotator=dot_annotator,
                    label_annotator=label_annotator,
                )

        if current_ball_row is not None:
            ball_x = _row_float(current_ball_row, "x")
            ball_y = _row_float(current_ball_row, "y")
            ball_detections = _trace_detection(
                ball_x, ball_y, tracker_id=21, box_size=20
            )
            if ball_detections is not None:
                speed = (
                    _row_float(current_ball_row, "speed")
                    if "speed" in current_ball_row.index and not pd.isna(current_ball_row["speed"])
                    else 0.0
                )
                scene_rgb = _annotate_supervision(
                    scene_rgb,
                    detections=ball_detections,
                    labels=[f"ball {speed:.1f}px/f"],
                    trace_annotator=trace_annotator,
                    dot_annotator=dot_annotator,
                    label_annotator=label_annotator,
                )
            cv2.circle(
                scene_rgb,
                (int(round(ball_x)), int(round(ball_y))),
                12,
                (255, 170, 0),
                2,
                cv2.LINE_AA,
            )

        if release_frame is not None and frame_idx == release_frame:
            cv2.putText(
                scene_rgb,
                "release frame",
                (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(scene_rgb, (12, 44), (220, 84), (255, 90, 90), 2, cv2.LINE_AA)

        side_panel_width = max(320, int(height * 0.72))
        side_panel_rgb = _draw_pose_side_panel(
            side_panel_width,
            height,
            pose_history_rows,
            current_row=current_pose_row,
            trail_joint=trail_joint,
        )
        combined_rgb = side_by_side(scene_rgb, side_panel_rgb, max_width=width * 2)

        if writer is None:
            writer, _ = open_mp4_video_writer(
                output_path,
                fps=fps,
                frame_size=(combined_rgb.shape[1], combined_rgb.shape[0]),
            )
        writer.write(cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR))
        frame_idx += 1

    if writer is None:
        raise RuntimeError(f"無法從影片產生 pose+ball overlay 預覽：{video_path}")
    writer.release()
    cap.release()
    return ensure_notebook_playable_mp4(output_path, overwrite=True)
