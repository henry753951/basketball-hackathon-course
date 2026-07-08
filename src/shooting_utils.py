from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, cast

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import requests

_POSE_LANDMARKER_LITE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
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
    min_pose_detection_confidence: float = 0.4,
    min_pose_presence_confidence: float = 0.4,
    min_tracking_confidence: float = 0.4,
) -> pd.DataFrame:
    from mediapipe.tasks.python import BaseOptions, vision

    if stride <= 0:
        raise ValueError(f"stride must be >= 1, got {stride}")

    model_path = ensure_mediapipe_pose_model(course_root)
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

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
    frame_idx = 0
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
    cap.release()

    if not rows:
        raise RuntimeError(
            "MediaPipe Tasks Pose Landmarker 沒有從影片偵測到任何 pose landmarks。"
            "請確認 side.mp4 為清楚的側拍投籃影片，或調整拍攝角度、距離與光線。"
        )
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
    pts = {
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
    edges = [
        ("hip", "shoulder"),
        ("shoulder", "elbow"),
        ("elbow", "wrist"),
        ("hip", "knee"),
        ("knee", "ankle"),
    ]
    for a, b in edges:
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
