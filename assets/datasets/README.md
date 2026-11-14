# Roboflow Datasets

Store student or instructor Roboflow exports here. All notebook languages use the same locations.

| Path | Format | Exercise |
| --- | --- | --- |
| `assets/datasets/roboflow_bbox_yolo/` | YOLO detection export with `data.yaml`, `train/`, `valid/`, `test/`, or equivalent image/label directories. | Day 1-03. |
| `assets/datasets/roboflow_court_coco/` | COCO keypoint export, usually containing `train/_annotations.coco.json`, `valid/_annotations.coco.json` and `test/_annotations.coco.json`. | Day 1-02. |
| `assets/datasets/roboflow_court_yolo_pose/` | Ultralytics YOLO pose data converted from the COCO export. | Day 1-02. |

Large datasets may remain on your computer or Google Drive rather than being committed to Git. Day 4 uses the supplied ball model directly; the old `roboflow_ball_yolo/` training location is not required by the current exercises.

## Download Through the Roboflow API

The Day 1-02 and Day 1-03 notebooks can download your dataset through the official Roboflow Python SDK. Select a language in the [notebook index](../../README.md#notebooks-in-teaching-order).

```python
USE_ROBOFLOW_DOWNLOAD = True
ROBOFLOW_WORKSPACE = "your-workspace"
ROBOFLOW_PROJECT = "your-project"
ROBOFLOW_VERSION = 1
ROBOFLOW_API_KEY = ""  # Leave empty for secure getpass input at runtime.
```

- BBOX detection downloads a `yolov8` export to `assets/datasets/roboflow_bbox_yolo/`.
- Court keypoints download a `coco` export to `assets/datasets/roboflow_court_coco/`, then convert it to `assets/datasets/roboflow_court_yolo_pose/`.
- Existing `data.yaml` files or COCO annotations are reused by default. Set `FORCE_DOWNLOAD = True` to download again.
- Set `FORCE_CONVERSION = True` to repeat COCO-to-YOLO Pose conversion.
- If the SDK is unavailable in Colab, rerun the initial setup cell to install the course requirements.
- After adding or editing annotations on the Roboflow website, use `Generate New Version` / `Publish` under `Versions` before downloading. Unpublished edits are not included in an existing version.
