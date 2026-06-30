from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DETECTION_CLASSES = [
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


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def dataset_status(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    data_yaml = root / "data.yaml"
    status: dict[str, Any] = {
        "path": str(root),
        "exists": root.exists(),
        "data_yaml": str(data_yaml),
        "data_yaml_exists": data_yaml.exists(),
        "splits": {},
    }
    if not data_yaml.exists():
        return status
    data = load_yaml(data_yaml)
    base = Path(data.get("path", root))
    if not base.is_absolute():
        base = root / base
    for split_key in ("train", "val", "valid", "test"):
        split_value = data.get(split_key)
        if not split_value:
            continue
        split_path = Path(split_value)
        if not split_path.is_absolute():
            split_path = base / split_path
        image_count = (
            sum(1 for p in split_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
            if split_path.exists()
            else 0
        )
        label_path = Path(str(split_path).replace("images", "labels"))
        label_count = len(list(label_path.rglob("*.txt"))) if label_path.exists() else 0
        status["splits"][split_key] = {
            "images": str(split_path),
            "labels": str(label_path),
            "image_count": image_count,
            "label_count": label_count,
        }
    status["names"] = data.get("names")
    status["kpt_shape"] = data.get("kpt_shape")
    return status


def find_coco_annotation(split_dir: Path) -> Path | None:
    for name in ("_annotations.coco.json", "_annotations.json", "annotations.json"):
        candidate = split_dir / name
        if candidate.exists():
            return candidate
    matches = sorted(split_dir.glob("*.json"))
    return matches[0] if matches else None


def _load_coco(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"COCO annotation root must be an object: {path}")
    return data


def _find_image(split_dir: Path, file_name: str) -> Path | None:
    raw = Path(file_name)
    candidates = [
        split_dir / raw,
        split_dir / raw.name,
        split_dir / "images" / raw,
        split_dir / "images" / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(split_dir.rglob(raw.name))
    return matches[0] if matches else None


def _infer_keypoint_metadata(coco: dict[str, Any]) -> tuple[list[str], list[list[int]], dict[int, int], list[str]]:
    categories = sorted(coco.get("categories") or [], key=lambda c: int(c.get("id", 0)))
    if not categories:
        raise ValueError("COCO categories are missing.")
    class_id_map = {int(cat["id"]): idx for idx, cat in enumerate(categories)}
    names = [str(cat.get("name", f"class_{idx}")) for idx, cat in enumerate(categories)]
    keypoint_names: list[str] = []
    skeleton: list[list[int]] = []
    for cat in categories:
        raw_names = cat.get("keypoints") or []
        if isinstance(raw_names, list) and len(raw_names) > len(keypoint_names):
            keypoint_names = [str(name) for name in raw_names]
            skeleton = []
            for edge in cat.get("skeleton") or []:
                if isinstance(edge, list) and len(edge) >= 2:
                    skeleton.append([int(edge[0]) - 1, int(edge[1]) - 1])
    if not keypoint_names:
        raise ValueError("COCO categories do not contain keypoint names.")
    return keypoint_names, skeleton, class_id_map, names


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalized_bbox(ann: dict[str, Any], img_w: float, img_h: float) -> tuple[float, float, float, float]:
    bbox = ann.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        raise ValueError("COCO keypoint annotation must contain bbox.")
    x, y, w, h = [float(v) for v in bbox[:4]]
    return (
        _clamp01((x + w / 2.0) / img_w),
        _clamp01((y + h / 2.0) / img_h),
        _clamp01(w / img_w),
        _clamp01(h / img_h),
    )


def _pose_label_line(
    ann: dict[str, Any],
    class_id: int,
    img_w: float,
    img_h: float,
    keypoint_count: int,
) -> str:
    xc, yc, bw, bh = _normalized_bbox(ann, img_w, img_h)
    values = [str(class_id), f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    keypoints = list(ann.get("keypoints") or [])
    keypoints.extend([0] * max(0, keypoint_count * 3 - len(keypoints)))
    for index in range(keypoint_count):
        x = float(keypoints[index * 3])
        y = float(keypoints[index * 3 + 1])
        v = int(float(keypoints[index * 3 + 2]))
        if v <= 0:
            values.extend(["0.000000", "0.000000", "0"])
        else:
            values.extend([f"{_clamp01(x / img_w):.6f}", f"{_clamp01(y / img_h):.6f}", str(v)])
    return " ".join(values)


def convert_roboflow_coco_keypoints_to_yolo_pose(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_ann = None
    for split in ("train", "valid", "test"):
        first_ann = find_coco_annotation(dataset_dir / split)
        if first_ann is not None:
            break
    if first_ann is None:
        raise FileNotFoundError(f"No COCO annotation file found under {dataset_dir}")

    keypoint_names, skeleton, class_id_map, names = _infer_keypoint_metadata(_load_coco(first_ann))
    stats: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        split_dir = dataset_dir / split
        ann_path = find_coco_annotation(split_dir)
        if ann_path is None:
            continue
        coco = _load_coco(ann_path)
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in coco.get("annotations") or []:
            if ann.get("iscrowd", 0):
                continue
            annotations_by_image[int(ann["image_id"])].append(ann)

        out_image_dir = output_dir / "images" / split
        out_label_dir = output_dir / "labels" / split
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        image_count = 0
        label_count = 0
        for image in coco.get("images") or []:
            src = _find_image(split_dir, str(image["file_name"]))
            if src is None or src.suffix.lower() not in IMAGE_EXTS:
                continue
            dst = out_image_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            image_count += 1
            img_w = float(image["width"])
            img_h = float(image["height"])
            lines = []
            for ann in annotations_by_image.get(int(image["id"]), []):
                category_id = int(ann["category_id"])
                if category_id not in class_id_map:
                    continue
                lines.append(
                    _pose_label_line(
                        ann,
                        class_id_map[category_id],
                        img_w,
                        img_h,
                        len(keypoint_names),
                    )
                )
            (out_label_dir / f"{dst.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
            label_count += len(lines)
        stats[split] = {"images": image_count, "labels": label_count}

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "kpt_shape": [len(keypoint_names), 3],
        "flip_idx": list(range(len(keypoint_names))),
        "names": {idx: name for idx, name in enumerate(names)},
        "keypoint_names": keypoint_names,
        "keypoint_skeleton": skeleton,
        "conversion_stats": stats,
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    return output_dir / "data.yaml"
