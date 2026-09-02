# Python 籃球運動資料分析課程

本 repo 為五日課程教材，適用於具備基礎 Python 語法、尚未系統學習電腦視覺或運動資料分析的修課者。課程內容涵蓋影像座標、Homography、物件偵測、追蹤、鳥瞰圖投影與投籃動作分析。

課程編排：

- Day 1 - Day 3：影像座標、Homography、Detection、ByteTrack 與 BEV 位置投影。
- Day 4：近距離投籃影片、人體姿態、球軌跡與出手時間觀察。
- Day 5：以 Proposal 或完成專案 Demo 報告整合所學，說明問題、方法、評估、限制與延伸方向。
- Notebook 保留主要實作流程；`src/` 放置重複使用的工具函式。

## 本機環境

本專案使用 `uv` 管理 Python 環境與 notebook 驗證工具；Apple Silicon macOS 已用 PyTorch MPS（Metal GPU）驗證。NVIDIA 機器則會自動使用 CUDA，沒有 GPU 時才退回 CPU。

```bash
uv sync --dev
uv run python -m compileall src
uv run ruff check .
uv run pyright src
```

確認目前 kernel 能看到 Mac GPU：

```bash
uv run python -c 'import torch; print(torch.__version__); print("MPS:", torch.backends.mps.is_available())'
```

執行 notebook 驗證時，請使用已同步的 `uv` 環境，避免混用系統 Python：

```bash
PYTHONUTF8=1 uv run jupyter execute <notebook-path> --timeout=180 --kernel_name=python3
```

Day 4 不需要訓練或 Roboflow API key，直接使用課程提供的：

```text
assets/models/detectors/ball_rimV8.pt
assets/models/detectors/shot_detection.pt
assets/models/pose/yolov8n-pose.pt
```

## Colab 使用流程

學生第一次使用時，先開啟 `init_colab.ipynb`：

1. 掛載自己的 Google Drive。
2. 將整個課程 repo 複製到 `MyDrive/basketball_hackathon/course/`。
3. 安裝課程需要的 Python 套件。
4. 驗證 Day 4 的三個模型權重都已同步，並測試 `assets/results/` 可以寫入 Drive。

之後開啟任一課程 Notebook 時，第一個 code cell 會自動掛載 Drive、定位課程資料夾、安裝 `requirements.txt`，並把 repo root 加入 Python import path，讓 Notebook 可以直接引用 `src/` 裡的共用工具。

Day 4 請從 Drive 裡的 `basketball_hackathon/course/day4/` 開啟。第一個 code cell 會再次檢查模型，並把 `RESULTS` 固定指向 `/content/drive/MyDrive/basketball_hackathon/course/assets/results/`；CSV、圖片、JSON 與預覽影片會直接保留在 Google Drive。

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
| `assets/converted/` | 經 Notebook 轉檔後的 MP4 影片，供 Day 4 分析與 Day 5 自選專案展示使用。 |
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

## Day 4 投籃分析模型

Day 4 直接使用課程提供的 `ball_rimV8.pt`、`shot_detection.pt` 與 `yolov8n-pose.pt`。Day 4-02 以 shot detector 鎖定投籃者並輸出雙側姿態角度；Day 4-03 再從手腕附近建立事件專屬球軌跡，估計 release frame、release 關節角度與 2D 起飛角。課堂會依序檢查單幀球框、人物骨架、事件球軌跡與 release 畫面，再解讀分析數值與限制。

### Roboflow Dataset 自動下載

Day 1 的 detection 與 court keypoint notebook 支援學生輸入 Roboflow API key 後，透過 Roboflow 官方 Python SDK 自動下載自己的標註資料：

- `day1/d1_03_bbox_homework.ipynb`：下載 `yolov8` detection export，目標位置是 `assets/datasets/roboflow_bbox_yolo/`。
- `day1/d1_02_keypoint_annotation_roboflow_lab.ipynb`：下載 `coco` keypoint export，目標位置是 `assets/datasets/roboflow_court_coco/`，再自動轉成 YOLO pose dataset `assets/datasets/roboflow_court_yolo_pose/`。
- Notebook 會先檢查資料是否已下載或已轉換；已存在時直接沿用。需要重抓或重轉時，設定 notebook 內的 `FORCE_DOWNLOAD = True` 或 `FORCE_CONVERSION = True`。
- Colab 若尚未安裝 SDK，重新執行 notebook 開頭的 bootstrap / requirements 安裝 cell 即可安裝 `roboflow`。

學生在 Roboflow 網頁完成標註後，還需要手動到專案的 `Versions` 頁面按 `Generate New Version` / `Publish`。只有已發布的 dataset version 才能被 notebook 的 API 下載流程抓到。建議學生依照下列順序操作：

1. 先解壓課程提供的學生上傳素材：
   `assets/student_uploads/roboflow_pose_student_upload_images.zip` 或 `assets/student_uploads/roboflow_bbox_student_upload_images.zip`。
2. 到 Roboflow 左側 side menu > `DATA` > `Upload Data`，把 zip 內的 5 張圖片上傳到自己的專案。
3. 接著到 side menu > `DATA` > `Annotate`，在 `Unassigned` 區塊點 `Annotate Images` 完成標註。
4. 到 `Versions` 頁建立新的 dataset version。
5. 記下 `workspace slug`、`project slug`、新的 `version number`。
6. 回到 notebook，把 `USE_ROBOFLOW_DOWNLOAD = True`，再填入 `ROBOFLOW_WORKSPACE`、`ROBOFLOW_PROJECT`、`ROBOFLOW_VERSION`。
7. `ROBOFLOW_API_KEY` 可留空，執行 cell 時會用 `getpass()` 安全輸入。

其中 pose 練習包內含以下 5 張圖片：

- `boston-celtics-new-york-knicks-game-1-q1-01_54-01_48_mp4-0000.jpg`
- `boston-celtics-new-york-knicks-game-1-q1-01_54-01_48_mp4-0001.jpg`
- `boston-celtics-new-york-knicks-game-1-q1-01_54-01_48_mp4-0002.jpg`
- `boston-celtics-new-york-knicks-game-1-q1-01_54-01_48_mp4-0003.jpg`
- `boston-celtics-new-york-knicks-game-1-q1-01_54-01_48_mp4-0004.jpg`

偵測結果再交由 ByteTrack 或 BoT-SORT 做跨 frame 關聯；短暫漏偵可用插值補齊。球體尺寸小、移動快且容易遮擋，因此課堂會先檢查漏偵與軌跡是否合理，不把 confidence 直接當成正確率。

## 建議上課順序

### Day 1：座標、點選工具、Homography、Roboflow keypoint / bbox 作業準備

1. `day1/d1_01_keypoint_pairing_homography.ipynb`
2. `day1/d1_02_keypoint_annotation_roboflow_lab.ipynb`
3. `day1/d1_03_bbox_homework.ipynb`

### Day 2：Detection 與 BBOX-to-BEV

1. `day2/d2_01_yolo26_detection.ipynb`
2. `day2/d2_02_yolo_players_to_bev.ipynb`
3. `day2/d2_03_bbox_to_bev_integration.ipynb`

### Day 3：ByteTrack 與位置數據化

1. `day3/d3_01_tracking_concept_iou_association.ipynb`
2. `day3/d3_02_yolo_bytetrack_tracking.ipynb`
3. `day3/d3_03_team_clustering.ipynb`
4. `day3/d3_04_tracking_to_bev_mini_project.ipynb`

### Day 4：投籃事件、人體姿態與出手角度

1. `day4/d4_01_ball_detector_bytetrack_preview.ipynb`
2. `day4/d4_02_yolo_pose_and_shot_lab.ipynb`
3. `day4/d4_03_ball_tracking_and_release_point_lab.ipynb`

### Day 5：Project Proposal 或完成專案 Demo

Day 5 不再新增固定 notebook。學生依 `day5/project_proposal_spec.tex` 準備提案或已完成專案展示；可假設 player tracking、隊伍、背號與 BEV 等先決模組存在，但必須清楚區分已完成、沿用與規劃中的部分。

老師每日前一小時的高層次授課主軸整理於 `daily_lecture_outline.tex`；內容以研究觀念、原理、數學公式與 pseudo code 為主，不含實作 code。

## 學生影片上傳

Day 4（以及選擇相關 Day 5 題目的組別）請先開啟
`day4/d4_01_ball_detector_bytetrack_preview.ipynb`。Notebook 會引導你們使用以下任一方式：

1. **Google Drive（推薦）**：在 Drive 內進入
   `我的雲端硬碟/basketball_hackathon/course/assets/raw/`，把 `.mp4`、`.mov`、`.m4v`
   或其他常見影片檔拖進去，再在 notebook 的 `VIDEO_FILENAME` 填入檔名。
2. **直接從 Colab 上傳**：把 notebook 的 `USE_BROWSER_UPLOAD` 改成 `True`，執行該格後
   從電腦選取影片。檔案會先寫入 Google Drive 的 `assets/raw/`，不是只留在暫存 runtime。

兩種方式都會把影片轉成瀏覽器較容易播放的：

```text
assets/converted/student_video.mp4
```

如果不指定新影片，notebook 會沿用課程內建的 `assets/converted/video_001.mp4`。Day 4-02
與 Day 4-03 都會讀取 Day 4-01 顯示的同一個 `using video` 路徑；圖片、CSV、JSON 與
overlay 影片則寫入 Google Drive 的 `assets/results/`。

## 路徑提醒

請不要使用中文資料夾名稱或中文影片檔名。中文課名只用於顯示；實際路徑使用：

```text
basketball_hackathon/course/
```
