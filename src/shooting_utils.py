from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _row_float(row: pd.Series, key: str) -> float:
    return float(row.at[key])


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
