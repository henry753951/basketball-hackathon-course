# Course Project / Notebook Plan

## Day 1 - Coordinates, Homography, Annotation Setup

- `day1/d1_01_keypoint_pairing_homography.ipynb`: BEV keypoint and camera-frame pairing tool; students build at least four correspondences, compute homography, and project a footpoint.
- `day1/d1_03_keypoint_annotation_roboflow_lab.ipynb`: Roboflow COCO keypoint export, YOLO pose conversion, optional Ultralytics training, trained court keypoint inference.
- `day1/d1_04_bbox_homework_setup.ipynb`: BBOX class design, Roboflow YOLO export check, optional Ultralytics detector training, trained detector inference.

## Day 2 - Detection and BBOX-to-BEV

- `day2/d2_01_manual_detection_box_to_bev.ipynb`: trained detector box → bottom-center → interactive BEV projection.
- `day2/d2_02_yolo26_detection.ipynb`: trained YOLO26 detector inference on reference video.
- `day2/d2_03_roboflow_bbox_training.ipynb`: Roboflow YOLO export, optional detector training, trained detector video preview.
- `day2/d2_04_bbox_to_bev_integration.ipynb`: detector + court keypoint model + homography → BEV video.

## Day 3 - ByteTrack and Tactical Board

- `day3/d3_01_tracking_concept_iou_association.ipynb`: IoU matrix from real detector outputs on adjacent frames.
- `day3/d3_02_yolo_bytetrack_tracking.ipynb`: Ultralytics tracking mode with `bytetrack.yaml`.
- `day3/d3_03_tracking_to_bev_mini_project.ipynb`: detector + ByteTrack + court keypoint homography → BEV path video.

## Day 4 - Outdoor Shooting Video Analysis

- `day4/d4_01_upload_and_convert_shooting_video.ipynb`: Buzzheavier or Drive upload; ffmpeg conversion.
- `day4/d4_02_mediapipe_pose_angle_lab.ipynb`: MediaPipe skeleton / synthetic fallback; elbow, knee, shoulder angles.
- `day4/d4_03_ball_tracking_and_release_point_lab.ipynb`: orange ball tracking, ball path and release frame proxy.

## Day 5 - Final Report and Showcase

- `day5/d5_01_shooting_analysis_pipeline_runner.ipynb`: integrate Day 4 outputs into summary JSON.
- `day5/d5_02_report_visualization_builder.ipynb`: angle plot, ball path, showcase skeleton.
- `day5/d5_03_showcase_export.ipynb`: copy deliverables and make `student_showcase.zip`.
