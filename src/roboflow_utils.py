from __future__ import annotations

import json
import os
import random
import shutil
import time
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


def _coerce_detection_label_line(line: str) -> str | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return None

    if len(coords) == 4:
        cx, cy, w, h = coords
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        w = max(0.0, x_max - x_min)
        h = max(0.0, y_max - y_min)
    else:
        return None

    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _write_detection_only_label(src_label_path: Path, dst_label_path: Path) -> bool:
    lines = src_label_path.read_text(encoding="utf-8").splitlines()
    normalized = []
    for line in lines:
        if not line.strip():
            continue
        normalized_line = _coerce_detection_label_line(line)
        if normalized_line is not None:
            normalized.append(normalized_line)
    dst_label_path.write_text(
        "\n".join(normalized) + ("\n" if normalized else ""),
        encoding="utf-8",
    )
    return bool(normalized)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _normalize_relative_dataset_path(root: Path, value: str) -> str:
    raw = Path(value)
    if raw.is_absolute():
        return value

    candidates = [raw]
    parts = list(raw.parts)
    while parts and parts[0] in {".", ".."}:
        parts = parts[1:]
        if parts:
            candidates.append(Path(*parts))

    for candidate in candidates:
        if (root / candidate).exists():
            return candidate.as_posix()
    return value


def _normalize_yolo_data_yaml(root: Path) -> None:
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        return

    data = load_yaml(data_yaml)
    changed = False
    for key in ("path", "train", "val", "valid", "test"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = _normalize_relative_dataset_path(root, value)
        if normalized != value:
            data[key] = normalized
            changed = True

    if changed:
        data_yaml.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


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
        base = root / Path(_normalize_relative_dataset_path(root, str(base)))
    for split_key in ("train", "val", "valid", "test"):
        split_value = data.get(split_key)
        if not split_value:
            continue
        split_path = Path(split_value)
        if not split_path.is_absolute():
            split_path = base / Path(_normalize_relative_dataset_path(root, str(split_path)))
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


def build_yolo_detection_subset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    train_limit: int,
    seed: int = 0,
    copy_val: bool = True,
    copy_test: bool = True,
    overwrite: bool = False,
) -> Path:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if train_limit <= 0:
        raise ValueError(f"train_limit must be >= 1, got {train_limit}")

    data_yaml = source_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"找不到 YOLO data.yaml：{data_yaml}")

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_yaml(data_yaml)
    train_rel = str(data.get("train") or "train/images")
    val_rel = str(data.get("val") or data.get("valid") or "valid/images")
    test_rel = str(data.get("test") or "test/images")

    def resolve_split(rel_path: str) -> Path:
        split_path = Path(_normalize_relative_dataset_path(source_dir, rel_path))
        if split_path.is_absolute():
            return split_path
        return source_dir / split_path

    def labels_dir_for(images_dir: Path) -> Path:
        return Path(str(images_dir).replace("images", "labels"))

    def sample_images(images_dir: Path, limit: int) -> list[Path]:
        image_paths = sorted(
            p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if len(image_paths) <= limit:
            return image_paths
        rng = random.Random(seed)
        chosen = rng.sample(image_paths, k=limit)
        return sorted(chosen)

    def copy_subset(images_dir: Path, out_images_dir: Path, *, limit: int | None) -> dict[str, int]:
        out_labels_dir = Path(str(out_images_dir).replace("images", "labels"))
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)
        src_labels_dir = labels_dir_for(images_dir)

        selected = sample_images(images_dir, limit) if limit is not None else sorted(
            p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        label_count = 0
        for image_path in selected:
            shutil.copy2(image_path, out_images_dir / image_path.name)
            label_path = src_labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                wrote_label = _write_detection_only_label(
                    label_path, out_labels_dir / label_path.name
                )
                if wrote_label:
                    label_count += 1
        return {"images": len(selected), "labels": label_count}

    train_src = resolve_split(train_rel)
    if not train_src.exists():
        raise FileNotFoundError(f"找不到 train images：{train_src}")

    train_out_rel = Path("train") / "images"
    stats = {
        "train": copy_subset(train_src, output_dir / train_out_rel, limit=train_limit),
    }

    if copy_val:
        val_src = resolve_split(val_rel)
        if val_src.exists():
            stats["val"] = copy_subset(val_src, output_dir / Path("valid") / "images", limit=None)
    if copy_test:
        test_src = resolve_split(test_rel)
        if test_src.exists():
            stats["test"] = copy_subset(test_src, output_dir / Path("test") / "images", limit=None)

    subset_data = dict(data)
    subset_data.pop("path", None)
    subset_data["train"] = "train/images"
    if "val" in subset_data or "valid" in subset_data:
        subset_data["val"] = "valid/images"
        subset_data.pop("valid", None)
    if "test" in subset_data and copy_test:
        subset_data["test"] = "test/images"
    elif not copy_test:
        subset_data.pop("test", None)

    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(subset_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output_dir / ".subset_meta.json").write_text(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "train_limit": int(train_limit),
                "seed": int(seed),
                "copy_val": bool(copy_val),
                "copy_test": bool(copy_test),
                "stats": stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir / "data.yaml"


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


def _roboflow_download_failure_message(
    *,
    workspace: str,
    project: str,
    version: int,
    export_format: str,
    output_dir: Path,
    download_dir: Path,
) -> str:
    return (
        "無法從 Roboflow 下載資料集。"
        f" workspace={workspace!r}, project={project!r}, version={int(version)}, format={export_format!r}。"
        " 請確認 Colab 有外網、API key 正確、且 Roboflow 版本已經 Publish。"
        f" 若不走下載流程，也可以先把已匯出的資料集放到 {output_dir}。"
        f" SDK 下載目錄預期會建立在 {download_dir}。"
    )


def _load_roboflow_sdk() -> type:
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise ImportError(
            "目前環境還沒有載入 Roboflow 官方 SDK。"
            "如果你在 Colab，請重新執行 notebook 最前面的 bootstrap cell。"
        ) from exc
    return Roboflow


def _download_roboflow_dataset_legacy(
    *,
    workspace: str,
    project: str,
    version: int,
    export_format: str,
    extract_dir: Path,
    api_key: str,
) -> None:
    export_url = (
        f"https://api.roboflow.com/{workspace}/{project}/{int(version)}/{export_format}"
        f"?api_key={api_key}&nocache=true"
    )
    export_info: dict[str, Any] | None = None
    for _ in range(60):
        response = requests.get(export_url, timeout=120)
        if response.status_code not in (200, 202):
            raise RuntimeError(response.text)
        payload = response.json()
        if response.status_code == 202 or payload.get("ready") is False:
            time.sleep(2)
            continue
        if isinstance(payload, dict):
            export_info = payload
            break
        raise RuntimeError(f"Unexpected Roboflow export response: {payload!r}")

    if not export_info:
        raise RuntimeError("Timed out while waiting for Roboflow dataset export to become ready.")

    download_url = export_info.get("export", {}).get("link")
    if not download_url:
        raise RuntimeError(f"Roboflow export response did not include a download link: {export_info!r}")

    extract_dir.mkdir(parents=True, exist_ok=True)
    archive_path = extract_dir.parent / f"{extract_dir.name}.zip"
    with requests.get(download_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with archive_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_dir)
    archive_path.unlink(missing_ok=True)


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
    """Download a Roboflow dataset version with the official SDK into output_dir."""
    output_dir = Path(output_dir)
    if not overwrite:
        if expected == "yolo" and has_yolo_dataset(output_dir):
            return output_dir
        if expected == "coco-keypoints" and has_coco_keypoint_dataset(output_dir):
            return output_dir

    cache_dir = output_dir.parent / ".roboflow_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = cache_dir / f"{workspace}_{project}_v{int(version)}_{export_format}"

    if not overwrite and extract_dir.exists():
        try:
            dataset_root = _find_dataset_root(extract_dir, expected=expected)
            _copy_directory_contents(dataset_root, output_dir, overwrite=False)
            return output_dir
        except FileNotFoundError:
            pass

    if overwrite and extract_dir.exists():
        shutil.rmtree(extract_dir)

    Roboflow = _load_roboflow_sdk()
    key = _clean_api_key(api_key)

    try:
        rf = Roboflow(api_key=key)
        rf_project = rf.project(f"{workspace}/{project}")
        rf_version = rf_project.version(int(version))
        version_download = getattr(rf_version, "download", None)
        if callable(version_download):
            try:
                version_download(
                    model_format=export_format,
                    location=str(extract_dir),
                    overwrite=overwrite,
                )
            except TypeError:
                try:
                    version_download(export_format, str(extract_dir), overwrite)
                except TypeError:
                    version_download(export_format, str(extract_dir))
        else:
            _download_roboflow_dataset_legacy(
                workspace=workspace,
                project=project,
                version=version,
                export_format=export_format,
                extract_dir=extract_dir,
                api_key=key,
            )
    except Exception as exc:
        raise RuntimeError(
            _roboflow_download_failure_message(
                workspace=workspace,
                project=project,
                version=version,
                export_format=export_format,
                output_dir=output_dir,
                download_dir=extract_dir,
            )
        ) from exc

    dataset_root = _find_dataset_root(extract_dir, expected=expected)
    _copy_directory_contents(dataset_root, output_dir, overwrite=overwrite)
    if expected == "yolo":
        _normalize_yolo_data_yaml(output_dir)
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
