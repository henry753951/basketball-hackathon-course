from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd

from .cv_utils import draw_boxes, ensure_dir, save_json
from .video_utils import ensure_notebook_playable_mp4, open_mp4_video_writer


def draw_detection_records(
    image_rgb: np.ndarray,
    detections: Sequence[Any],
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
    if not keep:
        return image_rgb.copy()
    try:
        import supervision as sv
    except ImportError:
        labels = _labels_from_detection_records(keep)
        return draw_boxes(image_rgb, [det.bbox_xyxy for det in keep], labels, color=(255, 184, 77), thickness=4)

    xyxy = np.asarray([det.bbox_xyxy for det in keep], dtype=np.float32)
    class_id = np.asarray([det.class_id for det in keep], dtype=int)
    confidence = np.asarray([det.confidence for det in keep], dtype=np.float32)
    tracker_ids = None
    if keep and any(det.track_id is not None for det in keep):
        tracker_ids = np.asarray([int(det.track_id or -1) for det in keep], dtype=int)
    sv_detections = sv.Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        tracker_id=tracker_ids,
        data={"class_name": np.asarray([det.class_name for det in keep], dtype=str)},
    )
    return _annotate_detection_scene(
        image_rgb,
        sv_detections,
        labels=_labels_from_detection_records(keep),
        trace=tracker_ids is not None,
    )


def write_detection_preview_video(
    *,
    video_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    max_frames: int = 45,
    conf: float = 0.25,
    imgsz: int = 960,
    class_names_override: Sequence[str] | dict[int, str] | None = None,
    keep_class_names: Iterable[str] | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    from .yolo_utils import detections_from_result, load_yolo_model, records_to_dicts

    model = load_yolo_model(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, first_frame_bgr = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"無法從影片讀取第一個 frame：{video_path}")
    height, width = first_frame_bgr.shape[:2]
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    writer, _ = open_mp4_video_writer(output_path, fps=fps, frame_size=(width, height))
    all_records: list[dict[str, Any]] = []
    frame_index = 0
    frame_bgr = first_frame_bgr
    while frame_index < max_frames:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = model.predict(frame_rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
        detections = detections_from_result(
            result,
            frame_index=frame_index,
            class_names_override=class_names_override,
        )
        if keep_class_names is not None:
            keep_names = {str(name) for name in keep_class_names}
            detections = [det for det in detections if det.class_name in keep_names]
        all_records.extend(records_to_dicts(detections))
        vis_rgb = draw_detection_records(frame_rgb, detections)
        writer.write(cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
        frame_index += 1
        ok, frame_bgr = cap.read()
        if not ok:
            break
    cap.release()
    writer.release()
    if frame_index == 0:
        raise RuntimeError(f"預覽影片沒有寫出任何 frame：{output_path}")
    ensure_notebook_playable_mp4(output_path)
    save_json(all_records, output_path.with_suffix(".json"))
    return output_path, all_records


def _detections_to_supervision(
    result: Any,
    class_names_override: Sequence[str] | dict[int, str] | None = None,
    keep_class_names: Iterable[str] | None = None,
) -> Any:
    from .yolo_utils import _class_name_lookup
    import supervision as sv

    detections = sv.Detections.from_ultralytics(result)
    names = _class_name_lookup(result, class_names_override)
    if len(detections) == 0:
        detections.data["class_name"] = np.asarray([], dtype=str)
        return detections
    class_ids = (
        detections.class_id
        if detections.class_id is not None
        else np.zeros(len(detections), dtype=int)
    )
    class_names = np.asarray([names.get(int(class_id), str(class_id)) for class_id in class_ids])
    detections.data["class_name"] = class_names
    if keep_class_names is None:
        from .yolo_utils import PLAYER_CLASS_NAMES

        mask = np.isin(class_names, list(PLAYER_CLASS_NAMES))
    else:
        mask = np.isin(class_names, [str(name) for name in keep_class_names])
    return detections[mask]


def _annotate_detection_scene(
    image_rgb: np.ndarray,
    detections: Any,
    *,
    labels: Sequence[str] | None = None,
    trace: bool = False,
) -> np.ndarray:
    try:
        import supervision as sv
    except ImportError:
        xyxy = np.asarray(getattr(detections, "xyxy", np.empty((0, 4))), dtype=float)
        fallback_labels = list(labels) if labels is not None else [""] * len(xyxy)
        if len(xyxy) == 0:
            return image_rgb.copy()
        return draw_boxes(image_rgb, xyxy.tolist(), fallback_labels, color=(255, 184, 77), thickness=4)

    scene = image_rgb.copy()
    if len(detections) == 0:
        return scene

    def call_annotate(annotator: Any, current_scene: np.ndarray, *, text_labels: Sequence[str] | None = None) -> np.ndarray:
        attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
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

    if trace and getattr(detections, "tracker_id", None) is not None and hasattr(sv, "TraceAnnotator"):
        try:
            trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=20)
        except TypeError:
            trace_annotator = sv.TraceAnnotator()
        scene = call_annotate(trace_annotator, scene)

    box_annotator_cls = (
        getattr(sv, "RoundBoxAnnotator", None)
        or getattr(sv, "BoxAnnotator", None)
        or getattr(sv, "BoundingBoxAnnotator", None)
    )
    label_annotator_cls = getattr(sv, "LabelAnnotator", None)
    if box_annotator_cls is not None:
        try:
            box_annotator = box_annotator_cls(thickness=4)
        except TypeError:
            box_annotator = box_annotator_cls()
        scene = call_annotate(box_annotator, scene)
    if label_annotator_cls is not None and labels is not None:
        try:
            label_annotator = label_annotator_cls(text_scale=0.7)
        except TypeError:
            label_annotator = label_annotator_cls()
        scene = call_annotate(label_annotator, scene, text_labels=labels)
    return scene


def _labels_from_detection_records(records: Sequence[Any]) -> list[str]:
    labels: list[str] = []
    for det in records:
        if det.track_id is not None:
            labels.append(f"#{det.track_id} {det.class_name} {det.confidence:.2f}")
        else:
            labels.append(f"{det.class_name} {det.confidence:.2f}")
    return labels


def _labels_from_supervision_detections(detections: Any) -> list[str]:
    class_names = list(detections.data.get("class_name", []))
    tracker_ids = getattr(detections, "tracker_id", None)
    confidences = getattr(detections, "confidence", None)
    labels: list[str] = []
    for i in range(len(detections)):
        class_name = str(class_names[i]) if i < len(class_names) else "player"
        confidence = float(confidences[i]) if confidences is not None else 1.0
        if tracker_ids is not None:
            labels.append(f"#{int(tracker_ids[i])} {class_name} {confidence:.2f}")
        else:
            labels.append(f"{class_name} {confidence:.2f}")
    return labels


def _supervision_from_detection_records(detections: Sequence[Any]) -> Any:
    import supervision as sv

    if not detections:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            class_id=np.empty((0,), dtype=int),
            confidence=np.empty((0,), dtype=np.float32),
            tracker_id=None,
            data={"class_name": np.asarray([], dtype=str)},
        )

    xyxy = np.asarray([det.bbox_xyxy for det in detections], dtype=np.float32)
    class_id = np.asarray([int(det.class_id) for det in detections], dtype=int)
    confidence = np.asarray([float(det.confidence) for det in detections], dtype=np.float32)
    tracker_ids = None
    if any(det.track_id is not None for det in detections):
        tracker_ids = np.asarray(
            [int(det.track_id) if det.track_id is not None else -1 for det in detections],
            dtype=int,
        )
    return sv.Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        tracker_id=tracker_ids,
        data={"class_name": np.asarray([str(det.class_name) for det in detections], dtype=str)},
    )


def write_bytetrack_preview_video(
    *,
    video_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    max_frames: int = 120,
    conf: float = 0.25,
    imgsz: int = 960,
    start_frame: int = 0,
    class_names_override: Sequence[str] | dict[int, str] | None = None,
    keep_class_names: Iterable[str] | None = None,
    hold_last_ball_frames: int = 3,
) -> tuple[Path, list[dict[str, Any]]]:
    from .yolo_utils import detections_from_result, load_yolo_model
    import supervision as sv

    model = load_yolo_model(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    writer, _ = open_mp4_video_writer(output_path, fps=fps, frame_size=(frame_width, frame_height))
    records: list[dict[str, Any]] = []
    local_frame_index = 0
    last_ball_record: dict[str, Any] | None = None
    try:
        tracker = sv.ByteTrack(
            frame_rate=float(fps),
            track_activation_threshold=max(0.001, min(conf * 2.0, 0.1)),
            lost_track_buffer=max(20, int(round(fps * 0.6))),
            minimum_matching_threshold=0.65,
        )
    except TypeError:
        tracker = sv.ByteTrack()

    source_frame_index = 0
    while local_frame_index < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if source_frame_index < start_frame:
            source_frame_index += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = model.predict(frame_rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
        detections = detections_from_result(
            result,
            frame_index=source_frame_index,
            class_names_override=class_names_override,
        )
        if keep_class_names is not None:
            keep_names = {str(name) for name in keep_class_names}
            detections = [det for det in detections if det.class_name in keep_names]
        sv_detections = _supervision_from_detection_records(detections)
        tracked = tracker.update_with_detections(sv_detections)
        display_detections = tracked
        labels = _labels_from_supervision_detections(tracked)
        trace_enabled = True
        record_source = "track"

        if len(display_detections) == 0 and len(sv_detections) > 0:
            display_detections = sv_detections
            labels = _labels_from_supervision_detections(display_detections)
            trace_enabled = False
            record_source = "detect"
        elif (
            len(display_detections) == 0
            and last_ball_record is not None
            and hold_last_ball_frames > 0
            and source_frame_index - int(last_ball_record["frame"]) <= hold_last_ball_frames
        ):
            carry_record = dict(last_ball_record)
            carry_record["frame"] = int(source_frame_index)
            carry_record["source"] = "carry"
            display_detections = _supervision_from_detection_records(
                [
                    type(
                        "CarryDetection",
                        (),
                        {
                            "bbox_xyxy": carry_record["bbox_xyxy"],
                            "class_id": carry_record["class_id"],
                            "confidence": carry_record["confidence"],
                            "track_id": carry_record["track_id"],
                            "class_name": carry_record["class_name"],
                        },
                    )()
                ]
            )
            labels = [f"#{carry_record['track_id']} {carry_record['class_name']} hold"]
            trace_enabled = False
            record_source = "carry"

        frame_vis = _annotate_detection_scene(
            frame_rgb,
            display_detections,
            labels=labels,
            trace=trace_enabled,
        )
        writer.write(cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR))

        xyxy = np.asarray(display_detections.xyxy, dtype=float)
        tracker_ids = (
            display_detections.tracker_id
            if display_detections.tracker_id is not None
            else np.full(len(display_detections), -1, dtype=int)
        )
        class_ids = (
            display_detections.class_id
            if display_detections.class_id is not None
            else np.full(len(display_detections), -1, dtype=int)
        )
        class_names = display_detections.data.get("class_name", ["player"] * len(display_detections))
        confidences = (
            display_detections.confidence
            if display_detections.confidence is not None
            else np.ones(len(display_detections), dtype=float)
        )
        frame_records: list[dict[str, Any]] = []
        for i in range(len(display_detections)):
            frame_records.append(
                {
                    "frame": int(source_frame_index),
                    "track_id": int(tracker_ids[i]),
                    "class_id": int(class_ids[i]),
                    "class_name": str(class_names[i]),
                    "confidence": float(confidences[i]),
                    "bbox_xyxy": [float(v) for v in xyxy[i].tolist()],
                    "source": record_source,
                }
            )
        if frame_records:
            best_record = max(frame_records, key=lambda item: float(item["confidence"]))
            last_ball_record = dict(best_record)
        records.extend(frame_records)
        local_frame_index += 1
        source_frame_index += 1

    cap.release()
    writer.release()
    if local_frame_index == 0:
        raise RuntimeError(f"ByteTrack 預覽影片沒有寫出任何 frame：{output_path}")
    ensure_notebook_playable_mp4(output_path)
    save_json(records, output_path.with_suffix(".json"))
    return output_path, records


def ball_track_dataframe_from_tracking_records(
    records: Sequence[dict[str, Any]],
    *,
    class_names: Sequence[str] = ("basketball", "ball"),
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    keep_names = {str(name) for name in class_names}
    for record in records:
        class_name = str(record.get("class_name", ""))
        if class_name not in keep_names:
            continue
        bbox_xyxy = record.get("bbox_xyxy")
        if not isinstance(bbox_xyxy, Sequence) or len(bbox_xyxy) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        rows.append(
            {
                "frame": int(record.get("frame", 0)),
                "track_id": int(record.get("track_id", -1)),
                "x": center_x,
                "y": center_y,
                "w": x2 - x1,
                "h": y2 - y1,
                "confidence": float(record.get("confidence", 0.0)),
                "class_name": class_name,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["frame", "track_id", "x", "y", "w", "h", "confidence", "class_name"]
        )

    df = pd.DataFrame(rows).sort_values(["frame", "confidence"], ascending=[True, False])
    df = df.drop_duplicates(subset="frame", keep="first").reset_index(drop=True)
    if len(df) > 1:
        df["vx"] = df["x"].diff()
        df["vy"] = df["y"].diff()
        df["speed"] = np.sqrt(df["vx"].fillna(0.0) ** 2 + df["vy"].fillna(0.0) ** 2)
    return df
