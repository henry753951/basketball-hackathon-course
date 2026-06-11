# Python「黑客松」實戰分析籃球運動數據

English title: **Python Hackathon: Cracking Basketball Data in Action**

這份 repo 是五天課程用的教學版。設計原則：

- Day 1 - Day 3：不一開始就包成完整系統，而是把「點座標、Homography、Detection、ByteTrack、BEV」拆成小任務。
- Day 4 - Day 5：切換到近距離投籃影片，讓學生自己上傳影片，做人體姿態、球軌跡與投籃分析。
- Notebook 優先可讀、可改少量變數；`src/` 只放重複使用的小工具。

## 結構

```text
course/
├── init_colab.ipynb
├── d1/
├── d2/
├── d3/
├── d4/
├── d5/
├── assets/
│   ├── raw/          # 學生原始影片或壓縮檔
│   ├── converted/    # ffmpeg 轉成 mp4 後的影片
│   ├── samples/      # 課程範例圖片、json、短影片
│   └── results/      # 分析輸出
├── src/
├── requirements.txt
└── workshop_schedule_updated.tex
```

## 建議上課順序

### Day 1：座標、點選工具、Homography、Roboflow keypoint / bbox 作業準備

1. `d1_01_colab_and_coordinate_click_tool.ipynb`
2. `d1_02_homography_point_projection.ipynb`
3. `d1_03_keypoint_annotation_roboflow_lab.ipynb`
4. `d1_04_bbox_homework_setup.ipynb`

### Day 2：Detection 與 BBOX-to-BEV

1. `d2_01_manual_detection_box_to_bev.ipynb`
2. `d2_02_yolo26_detection_preview.ipynb`
3. `d2_03_roboflow_bbox_training_preview.ipynb`
4. `d2_04_bbox_to_bev_integration.ipynb`

### Day 3：ByteTrack 與位置數據化

1. `d3_01_tracking_concept_iou_association.ipynb`
2. `d3_02_bytetrack_demo_with_sample_boxes.ipynb`
3. `d3_03_tracking_to_bev_mini_project.ipynb`

### Day 4：近距離投籃影片、人體姿態與球軌跡

1. `d4_01_upload_and_convert_shooting_video.ipynb`
2. `d4_02_mediapipe_pose_angle_lab.ipynb`
3. `d4_03_ball_tracking_and_release_point_lab.ipynb`

### Day 5：整合、報表與展示

1. `d5_01_shooting_analysis_pipeline_runner.ipynb`
2. `d5_02_report_visualization_builder.ipynb`
3. `d5_03_showcase_export.ipynb`

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
