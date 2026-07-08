# Roboflow Dataset 放置位置

本資料夾保留學生或教師從 Roboflow 匯出的資料集。Notebook 預設讀取下列位置：

| 路徑 | 格式 | 使用 Notebook |
| --- | --- | --- |
| `assets/datasets/roboflow_bbox_yolo/` | YOLO detection export，需包含 `data.yaml`、`train/`、`valid/`、`test/` 或對應的 `images/labels` 結構。 | `day1/d1_03_bbox_homework_setup.ipynb`、`day2/d2_03_roboflow_bbox_training.ipynb` |
| `assets/datasets/roboflow_court_coco/` | Roboflow COCO keypoint export，通常包含 `train/_annotations.coco.json`、`valid/_annotations.coco.json`、`test/_annotations.coco.json`。 | `day1/d1_02_keypoint_annotation_roboflow_lab.ipynb` |
| `assets/datasets/roboflow_court_yolo_pose/` | 由 COCO keypoint export 轉出的 Ultralytics YOLO pose 格式。 | `day1/d1_02_keypoint_annotation_roboflow_lab.ipynb` |

大型資料集可只放在本機或 Google Drive，不一定要提交到 Git。

## Roboflow API 下載

`day1/d1_02_keypoint_annotation_roboflow_lab.ipynb`、`day1/d1_03_bbox_homework_setup.ipynb`、`day2/d2_03_roboflow_bbox_training.ipynb` 都可以改用 Roboflow 官方 Python SDK 下載學生自己的 dataset。

在 notebook 內填入：

```python
USE_ROBOFLOW_DOWNLOAD = True
ROBOFLOW_WORKSPACE = "your-workspace"
ROBOFLOW_PROJECT = "your-project"
ROBOFLOW_VERSION = 1
ROBOFLOW_API_KEY = ""  # 留空會在執行時用 getpass 輸入
```

- BBOX detection 會下載 `yolov8` export 到 `assets/datasets/roboflow_bbox_yolo/`。
- Court keypoint 會下載 `coco` export 到 `assets/datasets/roboflow_court_coco/`，再自動轉成 `assets/datasets/roboflow_court_yolo_pose/`。
- 如果目標資料夾已經有 `data.yaml` 或 COCO annotations，預設會直接沿用；需要重抓時把 notebook 裡的 `FORCE_DOWNLOAD = True`。
- 需要重跑 COCO-to-YOLO Pose 轉換時，把 `FORCE_CONVERSION = True`。
- 若 Colab 還沒安裝 SDK，請重新執行 notebook 前面的 requirements 安裝 cell，或手動執行 `pip install roboflow`。
- 學生在 Roboflow 網頁上新增或修正標註後，必須先到 `Versions` 頁建立新的 dataset version，API 才能下載到最新結果。只改圖片標註但沒有 `Generate New Version` 時，notebook 仍然只會抓到舊版本。
