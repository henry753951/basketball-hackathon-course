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
| `student_uploads/` | 請學生上傳到 Roboflow 的資料。

## samples 檔案與使用位置

| 檔案 | 說明 | 使用 Notebook |
| --- | --- | --- |
| `sample_court_frame.png` | 球場相機視角範例圖；用於 keypoint 配對、Homography 與 Day 2-02 YOLO player footpoint → BEV 投影。 | `day1/d1_01_keypoint_pairing_homography.ipynb`、`day2/d2_02_yolo_players_to_bev.ipynb` |
| `sample_bev_court.json` | Reference-style colorful BEV court template；由 `src.geometry_utils.render_bev_court` 產生投影底圖。 | `day1/d1_01_keypoint_pairing_homography.ipynb`、`day1/d1_02_keypoint_annotation_roboflow_lab.ipynb`、`day2/d2_02_yolo_players_to_bev.ipynb`、`day2/d2_04_bbox_to_bev_integration.ipynb`、`day3/d3_03_tracking_to_bev_mini_project.ipynb` |

## Day 4 投籃分析模型

Day 4 直接使用三個課程提供的 Ultralytics 權重，學生不需要自行訓練：

- `models/detectors/ball_rimV8.pt`：偵測 `ball` 與 `rim`。
- `models/detectors/shot_detection.pt`：產生逐 frame 的投籃候選分數。
- `models/pose/yolov8n-pose.pt`：偵測投籃者的人體 keypoints。

Day 4-03 會用 shot detector 鎖定事件區段，從投籃者手腕附近建立該次出手專屬球軌跡，避免場上多顆球時只取最高 confidence 而切錯球。短暫漏偵以軌跡預測或插值補齊；confidence 不是正確率，release frame 與角度仍需用輸出的檢查圖人工確認。

執行 `init_colab.ipynb` 時會逐一檢查以上三個模型檔，並測試 `assets/results/` 是否可寫入。Colab 中的 Day 4 notebook 只接受位於 `/content/drive/MyDrive/basketball_hackathon/course/` 的課程根目錄，確保所有分析結果會保留在 Google Drive，而不是留在重啟後會消失的 `/content/`。

## 使用規範

- 學生自行拍攝或下載的原始影片放入 `raw/`。在 Google Drive 中的完整位置是
  `我的雲端硬碟/basketball_hackathon/course/assets/raw/`；也可以由 Day 4-01 的瀏覽器上傳格寫入。
- 課程參考比賽片段放入 `raw/reference_videos/`。
- Roboflow 匯出的資料集放入 `datasets/` 對應子資料夾。
- 已訓練模型權重放入 `models/` 對應子資料夾。
- 經 Notebook 轉檔後的影片放入 `converted/`。
- Notebook 產生的結果放入 `results/`。
- `samples/` 內檔案為課程範例資料，除非教師另有指示，請勿覆蓋。
