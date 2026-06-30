# Course Project / Notebook Plan

## Day 1 - Coordinates, Homography, Annotation Setup

- `day1/d1_01_colab_and_coordinate_click_tool.ipynb`: HTML canvas click tool; students copy image coordinates.
- `day1/d1_02_homography_point_projection.ipynb`: camera points + BEV points → homography matrix → project one point.
- `day1/d1_03_keypoint_annotation_roboflow_lab.ipynb`: keypoint label order, error check, optional training skeleton.
- `day1/d1_04_bbox_homework_setup.ipynb`: BBOX class design and Roboflow homework setup.

## Day 2 - Detection and BBOX-to-BEV

- `day2/d2_01_manual_detection_box_to_bev.ipynb`: hard-coded player box → bottom-center → BEV.
- `day2/d2_02_yolo26_detection_preview.ipynb`: trained model if available; sample JSON fallback.
- `day2/d2_03_roboflow_bbox_training_preview.ipynb`: Roboflow / YOLO training preview, disabled by default.
- `day2/d2_04_bbox_to_bev_integration.ipynb`: multiple player boxes → projected BEV points.

## Day 3 - ByteTrack and Tactical Board

- `day3/d3_01_tracking_concept_iou_association.ipynb`: IoU matrix and association.
- `day3/d3_02_bytetrack_demo_with_sample_boxes.ipynb`: track IDs from a simple tracker / ByteTrack concept.
- `day3/d3_03_tracking_to_bev_mini_project.ipynb`: track paths projected to BEV.

## Day 4 - Outdoor Shooting Video Analysis

- `day4/d4_01_upload_and_convert_shooting_video.ipynb`: Buzzheavier or Drive upload; ffmpeg conversion.
- `day4/d4_02_mediapipe_pose_angle_lab.ipynb`: MediaPipe skeleton / synthetic fallback; elbow, knee, shoulder angles.
- `day4/d4_03_ball_tracking_and_release_point_lab.ipynb`: orange ball tracking, ball path and release frame proxy.

## Day 5 - Final Report and Showcase

- `day5/d5_01_shooting_analysis_pipeline_runner.ipynb`: integrate Day 4 outputs into summary JSON.
- `day5/d5_02_report_visualization_builder.ipynb`: angle plot, ball path, showcase skeleton.
- `day5/d5_03_showcase_export.ipynb`: copy deliverables and make `student_showcase.zip`.
