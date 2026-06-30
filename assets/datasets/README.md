# Roboflow Dataset 放置位置

本資料夾保留學生或教師從 Roboflow 匯出的資料集。Notebook 預設讀取下列位置：

| 路徑 | 格式 | 使用 Notebook |
| --- | --- | --- |
| `assets/datasets/roboflow_bbox_yolo/` | YOLO detection export，需包含 `data.yaml`、`train/`、`valid/`、`test/` 或對應的 `images/labels` 結構。 | `day1/d1_03_bbox_homework_setup.ipynb`、`day2/d2_03_roboflow_bbox_training.ipynb` |
| `assets/datasets/roboflow_court_coco/` | Roboflow COCO keypoint export，通常包含 `train/_annotations.coco.json`、`valid/_annotations.coco.json`、`test/_annotations.coco.json`。 | `day1/d1_02_keypoint_annotation_roboflow_lab.ipynb` |
| `assets/datasets/roboflow_court_yolo_pose/` | 由 COCO keypoint export 轉出的 Ultralytics YOLO pose 格式。 | `day1/d1_02_keypoint_annotation_roboflow_lab.ipynb` |

大型資料集可只放在本機或 Google Drive，不一定要提交到 Git。
