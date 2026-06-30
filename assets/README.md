# Assets 說明

本資料夾存放課程使用的輸入資料、範例素材與 Notebook 輸出結果。

## 子資料夾

| 路徑 | 說明 |
| --- | --- |
| `raw/` | 學生自行上傳的原始影片或壓縮檔。 |
| `raw/reference_videos/` | 課程提供的籃球比賽參考片段；供 Day 1 - Day 3 的 YOLO、court keypoint、ByteTrack 與 BEV 流程使用。 |
| `converted/` | Notebook 轉檔後的 MP4 影片。Day 4 與 Day 5 會優先讀取此資料夾。 |
| `datasets/` | Roboflow 匯出的 detection / keypoint dataset；詳細格式見 `assets/datasets/README.md`。 |
| `models/` | 課程提供的已訓練 YOLO detector 與 court keypoint 權重。 |
| `samples/` | 課程內建範例資料。用於課堂示範、本機驗證，以及尚未取得模型輸出時的替代輸入。 |
| `results/` | Notebook 執行後產生的圖片、CSV、JSON 與壓縮檔。此資料夾內容不納入版本控制。 |

## samples 檔案與使用位置

| 檔案 | 說明 | 使用 Notebook |
| --- | --- | --- |
| `sample_court_frame.png` | 球場相機視角範例圖；用於 keypoint 配對、Homography 與 Day 2-01 BBOX-to-BEV 互動。 | `day1/d1_01_keypoint_pairing_homography.ipynb`、`day2/d2_01_manual_detection_box_to_bev.ipynb` |
| `sample_bev_court.json` | Reference-style colorful BEV court template；由 `src.geometry_utils.render_bev_court` 產生投影底圖。 | `day1/d1_01_keypoint_pairing_homography.ipynb`、`day1/d1_03_keypoint_annotation_roboflow_lab.ipynb`、`day2/d2_01_manual_detection_box_to_bev.ipynb`、`day2/d2_04_bbox_to_bev_integration.ipynb`、`day3/d3_03_tracking_to_bev_mini_project.ipynb` |

## 球追蹤模型建議

Day 4-03 的 `track_orange_ball` 是顏色式基準方法，用於說明球中心點、速度與出手 frame 的資料欄位。在正式專案中，建議使用 Ultralytics YOLO 類型的 object detector 偵測籃球，類別至少包含 `ball`；若任務包含進球判斷，可加入 `ball-in-basket`、rim 或 backboard。

偵測結果再交由 ByteTrack 或 BoT-SORT 做跨 frame 關聯；短暫漏偵可用插值補齊。球體尺寸小、移動快且容易遮擋，因此資料集應涵蓋不同拍攝角度、場地光線、球衣顏色與壓縮品質。

## 使用規範

- 學生自行拍攝或下載的原始影片放入 `raw/`。
- 課程參考比賽片段放入 `raw/reference_videos/`。
- Roboflow 匯出的資料集放入 `datasets/` 對應子資料夾。
- 已訓練模型權重放入 `models/` 對應子資料夾。
- 經 Notebook 轉檔後的影片放入 `converted/`。
- Notebook 產生的結果放入 `results/`。
- `samples/` 內檔案為課程範例資料，除非教師另有指示，請勿覆蓋。
