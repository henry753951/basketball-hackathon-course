# Course Assets

This directory contains course inputs, sample data and generated notebook results. English and Japanese notebooks use the same files and paths as the original versions.

## Directories

| Path | Purpose |
| --- | --- |
| `raw/` | Student-uploaded source videos and archives. |
| `raw/reference_videos/` | Reference game clips for Days 1–3: YOLO detection, court keypoints, ByteTrack and BEV. |
| `converted/` | Converted MP4 videos, used by Day 4 and relevant Day 5 projects. |
| `datasets/` | Roboflow detection/keypoint exports. See [the dataset guide](datasets/README.md). |
| `models/` | Supplied trained detector, court-keypoint and pose weights. |
| `samples/` | Built-in inputs for demonstrations, local validation and exercises before model outputs are available. |
| `results/` | Generated figures, CSV, JSON and archives. These outputs are excluded from version control. |
| `student_uploads/` | Image archives for students to upload to their own Roboflow projects. |

## Samples and Exercises

| File | Purpose | Exercises |
| --- | --- | --- |
| `sample_court_frame.png` | Camera-view court image for point correspondences, homography and player-footpoint projection. | Day 1-01; Day 2-02. |
| `sample_bev_court.json` | Colored BEV court specification rendered by `src.geometry_utils.render_bev_court`. | Day 1-01; Day 1-02; Day 2-02; Day 2-03; Day 3-04. |

Use the [course notebook index](../README.md#notebooks-in-teaching-order) to open these exercises in English, Japanese or Traditional Chinese.

## Day 4 Models

Day 4 uses three supplied Ultralytics weights without requiring students to train them:

- `models/detectors/ball_rimV8.pt`: detects `ball` and `rim`.
- `models/detectors/shot_detection.pt`: provides frame-level shot-candidate scores.
- `models/pose/yolov8n-pose.pt`: predicts body keypoints for the shooter.

Day 4-03 identifies candidate shot intervals, then starts an event-specific ball path near the shooter's wrist. This avoids selecting a different ball merely because it has the highest confidence. Short gaps can be predicted or interpolated. Confidence is not a guarantee of correctness; inspect the release images before using the release frame or angles.

The setup notebook checks all three model files and tests write access to `assets/results/`. In Colab, Day 4 requires a course root under `/content/drive/MyDrive/basketball_hackathon/course/` so outputs remain in Drive after the runtime ends.

## File Placement

- Put your source videos in `raw/`. In Drive, the full location is `My Drive/basketball_hackathon/course/assets/raw/`. The Day 4-01 browser-upload cell also saves there.
- Keep reference game clips in `raw/reference_videos/`.
- Place Roboflow exports in their corresponding `datasets/` subdirectories.
- Place trained weights in the appropriate `models/` subdirectories.
- Save converted video in `converted/` and generated results in `results/`.
- Do not overwrite files in `samples/` unless instructed by the teaching team.
