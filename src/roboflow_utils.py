from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
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


def has_yolo_dataset(path: str | Path) -> bool:
    return (Path(path) / "data.yaml").exists()


def has_coco_keypoint_dataset(path: str | Path) -> bool:
    root = Path(path)
    return any(find_coco_annotation(root / split) is not None for split in ("train", "valid", "test"))


def _clean_api_key(api_key: str | None = None) -> str:
    value = (api_key or os.getenv("ROBOFLOW_API_KEY") or "").strip()
    if not value or value == "YOUR_API_KEY":
        raise ValueError("請填入 Roboflow API key，或設定環境變數 ROBOFLOW_API_KEY。")
    return value


def _copy_directory_contents(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        elif overwrite or not target.exists():
            shutil.copy2(child, target)


def _find_dataset_root(extract_dir: Path, *, expected: str) -> Path:
    if expected == "yolo":
        matches = sorted(extract_dir.rglob("data.yaml"))
        if matches:
            return matches[0].parent
    elif expected == "coco-keypoints":
        for candidate in [extract_dir, *sorted(p for p in extract_dir.rglob("*") if p.is_dir())]:
            if has_coco_keypoint_dataset(candidate):
                return candidate
    else:
        raise ValueError(f"Unsupported expected dataset type: {expected}")
    raise FileNotFoundError(f"Downloaded archive did not contain a {expected} dataset.")


def download_roboflow_dataset(
    *,
    workspace: str,
    project: str,
    version: int,
    export_format: str,
    output_dir: str | Path,
    api_key: str | None = None,
    expected: str = "yolo",
    overwrite: bool = False,
) -> Path:
    """Download a generated Roboflow dataset version and extract it into output_dir.

    Uses Roboflow's REST export endpoint so students only need their API key and
    project identifiers. The API key is never written to disk.
    """
    output_dir = Path(output_dir)
    if not overwrite:
        if expected == "yolo" and has_yolo_dataset(output_dir):
            return output_dir
        if expected == "coco-keypoints" and has_coco_keypoint_dataset(output_dir):
            return output_dir

    key = _clean_api_key(api_key)
    endpoint = f"https://api.roboflow.com/{workspace}/{project}/{int(version)}/{export_format}"
    response = requests.get(endpoint, params={"api_key": key}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    export_info = payload.get("export") if isinstance(payload, dict) else None
    download_url = None
    if isinstance(export_info, dict):
        download_url = export_info.get("link")
    if download_url is None and isinstance(payload, dict):
        download_url = payload.get("link") or payload.get("download")
    if not isinstance(download_url, str) or not download_url:
        raise RuntimeError("Roboflow export response did not include a download link.")

    cache_dir = output_dir.parent / ".roboflow_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / f"{workspace}_{project}_v{int(version)}_{export_format}.zip"
    if overwrite or not archive_path.exists():
        with requests.get(download_url, stream=True, timeout=120) as download:
            download.raise_for_status()
            with archive_path.open("wb") as f:
                for chunk in download.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    extract_dir = cache_dir / archive_path.stem
    if overwrite and extract_dir.exists():
        shutil.rmtree(extract_dir)
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)

    dataset_root = _find_dataset_root(extract_dir, expected=expected)
    _copy_directory_contents(dataset_root, output_dir, overwrite=overwrite)
    metadata = {
        "workspace": workspace,
        "project": project,
        "version": int(version),
        "format": export_format,
        "expected": expected,
    }
    (output_dir / ".roboflow_download.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return output_dir


def ensure_roboflow_detection_dataset(
    output_dir: str | Path,
    *,
    workspace: str,
    project: str,
    version: int,
    api_key: str | None = None,
    export_format: str = "yolov8",
    overwrite: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    if has_yolo_dataset(output_dir) and not overwrite:
        return output_dir / "data.yaml"
    download_roboflow_dataset(
        workspace=workspace,
        project=project,
        version=version,
        export_format=export_format,
        output_dir=output_dir,
        api_key=api_key,
        expected="yolo",
        overwrite=overwrite,
    )
    return output_dir / "data.yaml"


def ensure_roboflow_court_pose_dataset(
    *,
    coco_dir: str | Path,
    yolo_pose_dir: str | Path,
    workspace: str,
    project: str,
    version: int,
    api_key: str | None = None,
    export_format: str = "coco",
    overwrite_download: bool = False,
    overwrite_conversion: bool = False,
) -> Path:
    coco_dir = Path(coco_dir)
    yolo_pose_dir = Path(yolo_pose_dir)
    if not has_coco_keypoint_dataset(coco_dir) or overwrite_download:
        download_roboflow_dataset(
            workspace=workspace,
            project=project,
            version=version,
            export_format=export_format,
            output_dir=coco_dir,
            api_key=api_key,
            expected="coco-keypoints",
            overwrite=overwrite_download,
        )
    data_yaml = yolo_pose_dir / "data.yaml"
    if data_yaml.exists() and not overwrite_conversion:
        return data_yaml
    return convert_roboflow_coco_keypoints_to_yolo_pose(
        coco_dir,
        yolo_pose_dir,
        overwrite=overwrite_conversion,
    )


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
    used_category_ids = {
        int(ann["category_id"])
        for ann in coco.get("annotations") or []
        if not ann.get("iscrowd", 0) and "category_id" in ann
    }
    if not used_category_ids:
        used_category_ids = {int(cat["id"]) for cat in categories}
    class_id_map = {category_id: idx for idx, category_id in enumerate(sorted(used_category_ids))}
    names_by_id = {
        class_id_map[int(cat["id"])]: str(cat.get("name", f"class_{idx}"))
        for idx, cat in enumerate(categories)
        if int(cat["id"]) in class_id_map
    }
    names = [names_by_id[idx] for idx in sorted(names_by_id)]
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
