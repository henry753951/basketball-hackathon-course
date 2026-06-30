# Python 籃球運動資料分析課程

本 repo 為五日課程教材，適用於具備基礎 Python 語法、尚未系統學習電腦視覺或運動資料分析的修課者。課程內容涵蓋影像座標、Homography、物件偵測、追蹤、鳥瞰圖投影與投籃動作分析。

課程編排：

- Day 1 - Day 3：影像座標、Homography、Detection、ByteTrack 與 BEV 位置投影。
- Day 4 - Day 5：近距離投籃影片、人體姿態、球軌跡、分析摘要與成果匯出。
- Notebook 保留主要實作流程；`src/` 放置重複使用的工具函式。

## 本機環境

本專案已使用 `uv` 管理 Python 3.12 環境與 notebook 驗證工具。

```powershell
uv sync
uv run python -m compileall src
uv run ruff check .
uv run pyright src
```

執行 notebook 驗證時，請使用已同步的 `uv` 環境，避免混用系統 Python：

```powershell
$env:PYTHONUTF8 = "1"
uv run jupyter execute <notebook-path> --timeout=180 --kernel_name=python3
```

## Colab 使用流程

學生第一次使用時，先開啟 `init_colab.ipynb`：

1. 掛載自己的 Google Drive。
2. 將整個課程 repo 複製到 `MyDrive/basketball_hackathon/course/`。
3. 安裝課程需要的 Python 套件。

之後開啟任一課程 Notebook 時，第一個 code cell 會自動掛載 Drive、定位課程資料夾、安裝 `requirements.txt`，並把 repo root 加入 Python import path，讓 Notebook 可以直接引用 `src/` 裡的共用工具。

## 結構

```text
course/
├── init_colab.ipynb
├── day1/
├── day2/
├── day3/
├── day4/
├── day5/
├── assets/
│   ├── raw/          # 學生原始影片或壓縮檔
│   ├── converted/    # ffmpeg 轉成 mp4 後的影片
│   ├── datasets/     # Roboflow 匯出資料集
│   ├── models/       # 課程提供的已訓練權重
│   ├── samples/      # 課程範例圖片與 json data
│   └── results/      # 分析輸出
├── src/
├── requirements.txt
└── workshop_schedule_updated.tex
```

## 資料與範例素材

`assets/` 存放課程輸入資料、學生影片與 Notebook 輸出結果。

| 路徑 | 用途 |
| --- | --- |
| `assets/raw/` | 學生自行上傳的原始影片或壓縮檔。 |
| `assets/raw/reference_videos/` | 課程提供的籃球比賽參考片段，供 Day 1 - Day 3 的 detector、keypoint、tracking 與 BEV 流程使用。 |
| `assets/converted/` | 經 Notebook 轉檔後的 MP4 影片，供 Day 4、Day 5 分析使用。 |
| `assets/datasets/` | 學生或教師從 Roboflow 匯出的 detection / keypoint dataset。 |
| `assets/models/` | 課程提供的已訓練 YOLO detector 與 court keypoint model。 |
| `assets/samples/` | 課程內建範例資料；用於沒有模型權重或標註資料時的課堂執行與驗證。 |
| `assets/results/` | Notebook 產生的圖檔、CSV、JSON 與 showcase zip。此資料夾內容不納入版本控制，僅保留 `.gitkeep`。 |

`assets/samples/` 內容如下：

| 檔案 | 使用單元 | 說明 |
| --- | --- | --- |
| `sample_court_frame.png` | Day 1 - Day 3 | 球場相機視角範例圖，用於座標點選、Homography、Detection 與 Tracking 視覺化。 |
| `sample_bev_court.json` | Day 1 - Day 3 | Reference-style colorful BEV court template；由 `src.geometry_utils.render_bev_court` 產生投影底圖。 |

完整使用位置請見 `assets/README.md`。

## 球追蹤模型建議

Day 4-03 的 `track_orange_ball` 是顏色式基準方法，僅用於建立球中心點、速度與出手 frame 的資料格式。在正式專案中，建議使用 Ultralytics YOLO 類型的 object detector 偵測籃球，類別至少包含 `ball`；若任務包含進球判斷，可加入 `ball-in-basket`、rim 或 backboard。

偵測結果再交由 ByteTrack 或 BoT-SORT 做跨 frame 關聯；短暫漏偵可用插值補齊。球體尺寸小、移動快且容易遮擋，訓練資料應涵蓋不同拍攝角度、場地光線、球衣顏色與壓縮品質。

## 建議上課順序

### Day 1：座標、點選工具、Homography、Roboflow keypoint / bbox 作業準備

1. `day1/d1_01_keypoint_pairing_homography.ipynb`
2. `day1/d1_02_keypoint_annotation_roboflow_lab.ipynb`
3. `day1/d1_03_bbox_homework_setup.ipynb`

### Day 2：Detection 與 BBOX-to-BEV

1. `day2/d2_01_manual_detection_box_to_bev.ipynb`
2. `day2/d2_02_yolo26_detection.ipynb`
3. `day2/d2_03_roboflow_bbox_training.ipynb`
4. `day2/d2_04_bbox_to_bev_integration.ipynb`

### Day 3：ByteTrack 與位置數據化

1. `day3/d3_01_tracking_concept_iou_association.ipynb`
2. `day3/d3_02_yolo_bytetrack_tracking.ipynb`
3. `day3/d3_03_tracking_to_bev_mini_project.ipynb`

### Day 4：近距離投籃影片、人體姿態與球軌跡

1. `day4/d4_01_upload_and_convert_shooting_video.ipynb`
2. `day4/d4_02_mediapipe_pose_angle_lab.ipynb`
3. `day4/d4_03_ball_tracking_and_release_point_lab.ipynb`

### Day 5：整合、報表與展示

1. `day5/d5_01_shooting_analysis_pipeline_runner.ipynb`
2. `day5/d5_02_report_visualization_builder.ipynb`
3. `day5/d5_03_showcase_export.ipynb`

## 學生影片上傳

Day 4 / Day 5 可以用兩種方式：

1. 把影片上傳到 `assets/raw/`
2. 使用 Buzzheavier 上傳影片後，把下載連結貼到 Notebook

Notebook 會把影片轉成：

```text
assets/converted/video_001.mp4
```

後續分析都讀 `assets/converted/*.mp4`。

## 路徑提醒

請不要使用中文資料夾名稱或中文影片檔名。中文課名只用於顯示；實際路徑使用：

```text
basketball_hackathon/course/
```
