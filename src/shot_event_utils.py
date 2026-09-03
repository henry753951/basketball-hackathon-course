from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import pandas as pd

from .cv_utils import ensure_dir, save_image_rgb, side_by_side
from .shooting_utils import angle_3pt
from .video_utils import ensure_notebook_playable_mp4, open_mp4_video_writer
from .yolo_utils import load_yolo_model, preferred_inference_device


_COCO_SIDE_KEYPOINTS = {
    "left": {
        "shoulder": 5,
        "elbow": 7,
        "wrist": 9,
        "hip": 11,
        "knee": 13,
        "ankle": 15,
    },
    "right": {
        "shoulder": 6,
        "elbow": 8,
        "wrist": 10,
        "hip": 12,
        "knee": 14,
        "ankle": 16,
    },
}
_SHOT_POSE_LOCK_THRESHOLD = 0.30


def _row_float(row: pd.Series, key: str) -> float:
    return float(cast(Any, row.at[key]))


def _row_int(row: pd.Series, key: str) -> int:
    return int(round(_row_float(row, key)))


def shot_detector_model_path(course_root: str | Path) -> Path:
    return Path(course_root) / "assets" / "models" / "detectors" / "shot_detection.pt"


def yolo_pose_model_path(course_root: str | Path) -> Path:
    return Path(course_root) / "assets" / "models" / "pose" / "yolov8n-pose.pt"


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-6)


def _top_box(result: Any) -> tuple[float, np.ndarray | None]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0.0, None
    confidences = boxes.conf.detach().cpu().numpy()
    index = int(np.argmax(confidences))
    return float(boxes.conf[index].item()), boxes.xyxy[index].detach().cpu().numpy()


def _select_pose_index(
    result: Any,
    *,
    shot_box: np.ndarray | None,
    shot_score: float,
    previous_center: np.ndarray | None,
) -> int | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()

    # Low-confidence shot boxes frequently alternate between nearby people.
    # Only let a strong shot signal override temporal pose continuity.
    if shot_box is not None and shot_score >= _SHOT_POSE_LOCK_THRESHOLD:
        overlaps = np.asarray([_bbox_iou(shot_box, box) for box in xyxy])
        if float(overlaps.max(initial=0.0)) > 0.01:
            return int(np.argmax(overlaps))

    centers = np.column_stack(
        [
            (xyxy[:, 0] + xyxy[:, 2]) / 2.0,
            (xyxy[:, 1] + xyxy[:, 3]) / 2.0,
        ]
    )
    if previous_center is not None:
        distances = np.linalg.norm(centers - previous_center.reshape(1, 2), axis=1)
        return int(np.argmin(distances - confidences * 20.0))

    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return int(np.argmax(areas * np.maximum(confidences, 0.01)))


def extract_yolo_shot_pose_sequence(
    video_path: str | Path,
    *,
    pose_model_path: str | Path,
    shot_model_path: str | Path,
    stride: int = 1,
    max_frames: int | None = 180,
    side: Literal["right", "left"] = "right",
    pose_conf: float = 0.20,
    shot_conf: float = 0.01,
    imgsz: int = 1280,
    device: str | int | None = None,
) -> pd.DataFrame:
    """Extract the active shooter's YOLO pose and a per-frame shot score.

    The shot detector selects the relevant person when it is confident enough;
    otherwise the previous person center keeps the same athlete selected.
    """
    if stride <= 0:
        raise ValueError(f"stride must be >= 1, got {stride}")

    pose_model = load_yolo_model(pose_model_path)
    shot_model = load_yolo_model(shot_model_path)
    inference_device = device if device is not None else preferred_inference_device()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    rows: list[dict[str, float | int | str]] = []
    previous_center: np.ndarray | None = None
    frame_index = 0
    while cap.isOpened():
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_index >= max_frames:
            break
        if frame_index % stride != 0:
            frame_index += 1
            continue

        shot_result = shot_model.predict(
            frame_bgr,
            conf=shot_conf,
            imgsz=imgsz,
            device=inference_device,
            verbose=False,
        )[0]
        shot_score, shot_box = _top_box(shot_result)
        pose_result = pose_model.predict(
            frame_bgr,
            conf=pose_conf,
            imgsz=imgsz,
            device=inference_device,
            verbose=False,
        )[0]
        pose_index = _select_pose_index(
            pose_result,
            shot_box=shot_box,
            shot_score=shot_score,
            previous_center=previous_center,
        )
        if pose_index is None:
            frame_index += 1
            continue

        pose_boxes = pose_result.boxes
        pose_keypoints = pose_result.keypoints
        if pose_boxes is None or pose_keypoints is None:
            frame_index += 1
            continue
        pose_box = pose_boxes.xyxy[pose_index].detach().cpu().numpy()
        pose_confidence = float(pose_boxes.conf[pose_index].item())
        previous_center = np.asarray(
            [
                (pose_box[0] + pose_box[2]) / 2.0,
                (pose_box[1] + pose_box[3]) / 2.0,
            ],
            dtype=float,
        )
        keypoints = pose_keypoints.xy[pose_index].detach().cpu().numpy()
        keypoint_confidences = pose_keypoints.conf
        keypoint_conf = (
            keypoint_confidences[pose_index].detach().cpu().numpy()
            if keypoint_confidences is not None
            else np.ones(17, dtype=float)
        )

        row: dict[str, float | int | str] = {
            "frame": frame_index,
            "pose_confidence": pose_confidence,
            "person_x1": float(pose_box[0]),
            "person_y1": float(pose_box[1]),
            "person_x2": float(pose_box[2]),
            "person_y2": float(pose_box[3]),
            "shot_score": shot_score,
            "shot_x1": float(shot_box[0]) if shot_box is not None else float("nan"),
            "shot_y1": float(shot_box[1]) if shot_box is not None else float("nan"),
            "shot_x2": float(shot_box[2]) if shot_box is not None else float("nan"),
            "shot_y2": float(shot_box[3]) if shot_box is not None else float("nan"),
            "pose_side": side,
        }
        for side_name, landmark_indices in _COCO_SIDE_KEYPOINTS.items():
            points: dict[str, np.ndarray] = {}
            for joint_name, landmark_index in landmark_indices.items():
                point = keypoints[landmark_index]
                points[joint_name] = point
                row[f"{side_name}_{joint_name}_x"] = float(point[0])
                row[f"{side_name}_{joint_name}_y"] = float(point[1])
                row[f"{side_name}_{joint_name}_confidence"] = float(
                    keypoint_conf[landmark_index]
                )
            row[f"{side_name}_elbow_angle"] = angle_3pt(
                points["shoulder"], points["elbow"], points["wrist"]
            )
            row[f"{side_name}_shoulder_angle"] = angle_3pt(
                points["hip"], points["shoulder"], points["elbow"]
            )
            row[f"{side_name}_knee_angle"] = angle_3pt(
                points["hip"], points["knee"], points["ankle"]
            )

        for joint_name in _COCO_SIDE_KEYPOINTS[side]:
            row[f"{joint_name}_x"] = float(row[f"{side}_{joint_name}_x"])
            row[f"{joint_name}_y"] = float(row[f"{side}_{joint_name}_y"])
        row["elbow_angle"] = float(row[f"{side}_elbow_angle"])
        row["shoulder_angle"] = float(row[f"{side}_shoulder_angle"])
        row["knee_angle"] = float(row[f"{side}_knee_angle"])
        rows.append(row)
        frame_index += 1

    cap.release()
    if not rows:
        raise RuntimeError("YOLO pose model 沒有偵測到可用的人體姿態。")
    return pd.DataFrame(rows)


def plot_shot_event_analysis(
    frame_features: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(
        frame_features["frame"],
        frame_features["shot_score"],
        color="#8b5cf6",
        alpha=0.45,
        label="shot score",
    )
    axes[0].plot(
        frame_features["frame"],
        frame_features["shot_score_smooth"],
        color="#6d28d9",
        linewidth=2.2,
        label="5-frame mean",
    )
    axes[0].axhline(0.30, color="#ef4444", linestyle="--", linewidth=1, label="event threshold")
    axes[0].set_ylabel("shot confidence")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.2)

    axes[1].plot(frame_features["frame"], frame_features["elbow_angle"], label="elbow")
    axes[1].plot(frame_features["frame"], frame_features["shoulder_angle"], label="shoulder")
    axes[1].plot(frame_features["frame"], frame_features["knee_angle"], label="knee")
    axes[1].set_ylabel("joint angle (degrees)")
    axes[1].set_xlabel("frame")
    axes[1].set_ylim(0, 190)
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.2)

    for _, event in events_df.iterrows():
        start = _row_int(event, "start_frame")
        end = _row_int(event, "end_frame")
        release = _row_int(event, "release_frame")
        for axis in axes:
            axis.axvspan(start, end, color="#f59e0b", alpha=0.12)
            axis.axvline(release, color="#ef4444", linewidth=1.8)
        axes[0].annotate(
            f"shot {_row_int(event, 'event_id')}\nrelease {release}",
            (release, min(0.95, _row_float(event, "peak_score") + 0.08)),
            ha="center",
            fontsize=9,
        )
        axes[1].annotate(
            f"launch {_row_float(event, 'launch_angle_deg'):.1f}°\nelbow {_row_float(event, 'release_elbow_angle'):.1f}°",
            (release, _row_float(event, "release_elbow_angle")),
            xytext=(8, -38),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.8},
        )
    fig.suptitle("Day 4 shot event and release-angle analysis")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _float_from_row(row: pd.Series, key: str) -> float:
    return _row_float(row, key)


def _draw_pose(scene_rgb: np.ndarray, row: pd.Series) -> None:
    points: dict[tuple[str, str], tuple[int, int]] = {}
    for side in ("left", "right"):
        for joint in _COCO_SIDE_KEYPOINTS[side]:
            points[(side, joint)] = (
                int(round(_float_from_row(row, f"{side}_{joint}_x"))),
                int(round(_float_from_row(row, f"{side}_{joint}_y"))),
            )
        for first, second in (
            ("shoulder", "elbow"),
            ("elbow", "wrist"),
            ("shoulder", "hip"),
            ("hip", "knee"),
            ("knee", "ankle"),
        ):
            cv2.line(scene_rgb, points[(side, first)], points[(side, second)], (44, 180, 255), 3, cv2.LINE_AA)
    cv2.line(scene_rgb, points[("left", "shoulder")], points[("right", "shoulder")], (44, 180, 255), 3, cv2.LINE_AA)
    cv2.line(scene_rgb, points[("left", "hip")], points[("right", "hip")], (44, 180, 255), 3, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(scene_rgb, point, 4, (255, 255, 255), -1, cv2.LINE_AA)


def _event_for_frame(events_df: pd.DataFrame, frame: int) -> pd.Series | None:
    for _, event in events_df.iterrows():
        if _row_int(event, "start_frame") - 12 <= frame <= _row_int(event, "end_frame") + 14:
            return event
    return None


def _draw_shot_overlay(
    frame_bgr: np.ndarray,
    *,
    frame: int,
    pose_row: pd.Series | None,
    ball_row: pd.Series | None,
    event: pd.Series | None,
) -> np.ndarray:
    scene = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if pose_row is not None:
        _draw_pose(scene, pose_row)
        shot_score = _float_from_row(pose_row, "shot_score")
        if shot_score >= 0.05:
            x1 = int(round(_float_from_row(pose_row, "shot_x1")))
            y1 = int(round(_float_from_row(pose_row, "shot_y1")))
            x2 = int(round(_float_from_row(pose_row, "shot_x2")))
            y2 = int(round(_float_from_row(pose_row, "shot_y2")))
            cv2.rectangle(scene, (x1, y1), (x2, y2), (255, 196, 64), 2, cv2.LINE_AA)
        cv2.putText(
            scene,
            f"shot score {shot_score:.2f}",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if ball_row is not None:
        x = int(round(_float_from_row(ball_row, "x")))
        y = int(round(_float_from_row(ball_row, "y")))
        cv2.circle(scene, (x, y), 13, (255, 64, 64), 3, cv2.LINE_AA)
        cv2.putText(scene, "event ball", (x + 16, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 64, 64), 2, cv2.LINE_AA)

    if event is not None:
        event_id = _row_int(event, "event_id")
        release_frame = _row_int(event, "release_frame")
        launch_angle = _row_float(event, "launch_angle_deg")
        elbow_angle = _row_float(event, "release_elbow_angle")
        cv2.rectangle(scene, (12, 46), (390, 118), (20, 20, 20), -1)
        cv2.putText(scene, f"SHOT EVENT #{event_id}", (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 205, 64), 2, cv2.LINE_AA)
        cv2.putText(scene, f"launch {launch_angle:.1f} deg | elbow {elbow_angle:.1f} deg", (24, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        if frame == release_frame:
            cv2.rectangle(scene, (10, 10), (520, 135), (255, 64, 64), 4, cv2.LINE_AA)
            cv2.putText(scene, "RELEASE", (410, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 64, 64), 3, cv2.LINE_AA)
            if ball_row is not None:
                x = int(round(_float_from_row(ball_row, "x")))
                y = int(round(_float_from_row(ball_row, "y")))
                direction = 1.0 if _row_float(event, "launch_velocity_x_px_per_frame") >= 0 else -1.0
                radians = math.radians(launch_angle)
                endpoint = (
                    int(round(x + direction * 130.0 * math.cos(radians))),
                    int(round(y - 130.0 * math.sin(radians))),
                )
                cv2.arrowedLine(scene, (x, y), endpoint, (64, 255, 128), 4, cv2.LINE_AA, tipLength=0.18)
    return scene


def render_shot_event_overlay_video(
    video_path: str | Path,
    frame_features: pd.DataFrame,
    events_df: pd.DataFrame,
    event_ball_df: pd.DataFrame,
    output_path: str | Path,
    *,
    max_frames: int = 180,
) -> Path:
    pose_rows = {
        _row_int(row, "frame"): row
        for _, row in frame_features.iterrows()
    }
    ball_rows = {
        (_row_int(row, "event_id"), _row_int(row, "frame")): row
        for _, row in event_ball_df.iterrows()
    }
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer, _ = open_mp4_video_writer(output_path, fps=fps, frame_size=(width, height))
    frame = 0
    while cap.isOpened() and frame < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        event = _event_for_frame(events_df, frame)
        event_id = _row_int(event, "event_id") if event is not None else -1
        ball_row = ball_rows.get((event_id, frame))
        scene = _draw_shot_overlay(
            frame_bgr,
            frame=frame,
            pose_row=pose_rows.get(frame),
            ball_row=ball_row,
            event=event,
        )
        writer.write(cv2.cvtColor(scene, cv2.COLOR_RGB2BGR))
        frame += 1
    writer.release()
    cap.release()
    return ensure_notebook_playable_mp4(output_path, overwrite=True)


def render_shot_event_check_image(
    video_path: str | Path,
    frame_features: pd.DataFrame,
    events_df: pd.DataFrame,
    event_ball_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    pose_rows = {
        _row_int(row, "frame"): row
        for _, row in frame_features.iterrows()
    }
    ball_rows = {
        (_row_int(row, "event_id"), _row_int(row, "frame")): row
        for _, row in event_ball_df.iterrows()
    }
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    tiles: list[np.ndarray] = []
    for _, event_series in events_df.iterrows():
        event_id = _row_int(event_series, "event_id")
        for offset in (-2, 0, 4):
            frame = max(0, _row_int(event_series, "release_frame") + offset)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            scene = _draw_shot_overlay(
                frame_bgr,
                frame=frame,
                pose_row=pose_rows.get(frame),
                ball_row=ball_rows.get((event_id, frame)),
                event=event_series,
            )
            cv2.putText(scene, f"frame {frame}", (18, scene.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(scene)
    cap.release()
    if not tiles:
        raise RuntimeError("沒有可產生投籃事件檢查圖的 frame。")
    rows: list[np.ndarray] = []
    for start in range(0, len(tiles), 3):
        row = tiles[start]
        for tile in tiles[start + 1 : start + 3]:
            row = side_by_side(row, tile, max_width=2400)
        rows.append(row)
    target_width = max(row.shape[1] for row in rows)
    normalized_rows: list[np.ndarray] = []
    for row in rows:
        if row.shape[1] == target_width:
            normalized_rows.append(row)
            continue
        scale = target_width / row.shape[1]
        normalized_rows.append(
            cv2.resize(
                row,
                (target_width, int(round(row.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        )
    contact_sheet = np.vstack(normalized_rows)
    output_path = Path(output_path)
    save_image_rgb(output_path, contact_sheet)
    return output_path
