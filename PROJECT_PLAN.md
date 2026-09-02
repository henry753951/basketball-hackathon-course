# Course Project / Notebook Plan

## Day 1 - Coordinates, Homography, Annotation Setup

- `day1/d1_01_keypoint_pairing_homography.ipynb`: BEV keypoint and camera-frame pairing tool; students build at least four correspondences, compute homography, and project a footpoint.
- `day1/d1_02_keypoint_annotation_roboflow_lab.ipynb`: Roboflow COCO keypoint export, YOLO pose conversion, optional Ultralytics training, trained court keypoint inference.
- `day1/d1_03_bbox_homework.ipynb`: BBOX class design, Roboflow YOLO export check, optional Ultralytics detector training, trained detector inference.

## Day 2 - Detection and BBOX-to-BEV

- `day2/d2_01_yolo26_detection.ipynb`: trained YOLO26 detector inference on reference video.
- `day2/d2_02_yolo_players_to_bev.ipynb`: trained detector box → bottom-center → project all detected players to BEV in one pass.
- `day2/d2_03_bbox_to_bev_integration.ipynb`: detector + 即時場地 keypoint + homography integration → BEV video.

## Day 3 - ByteTrack and Tactical Board

- `day3/d3_01_tracking_concept_iou_association.ipynb`: IoU matrix from real detector outputs on adjacent frames.
- `day3/d3_02_yolo_bytetrack_tracking.ipynb`: Ultralytics tracking mode with `bytetrack.yaml`.
- `day3/d3_03_team_clustering.ipynb`: torso crops → HSV histogram features → two-cluster K-means, with box/crop/feature/result visualizations.
- `day3/d3_04_tracking_to_bev_mini_project.ipynb`: detector + ByteTrack + court keypoint homography → BEV paths, then track-level team clustering with original-frame, feature-space, team-colored BEV visualizations, and a side-by-side team tracking video.

## Day 4 - Provided Models, Shot Events, Pose, and Release Angles

- `day4/d4_01_ball_detector_bytetrack_preview.ipynb`: direct inference with the provided ball detector + supervision ByteTrack preview on converted video.
- `day4/d4_02_yolo_pose_and_shot_lab.ipynb`: provided YOLO pose + shot detector → active shooter, bilateral keypoints, joint angles, and per-frame shot score.
- `day4/d4_03_ball_tracking_and_release_point_lab.ipynb`: shot candidate segment + shooter wrist + all ball candidates → event-specific ball path, release frame, 2D launch angle, joint angles, and visual QA outputs.

## Day 5 - Project Proposal or Completed Demo

- No fixed implementation notebook is added on Day 5.
- `day5/project_proposal_spec.tex`: proposal/demo deliverables, presentation flow, evaluation rubric, and topic directions.
- Students may assume prerequisites such as player tracking, team classification, jersey recognition, and BEV projection, but must distinguish completed, reused, and planned modules.
- `daily_lecture_outline.tex`: high-level lecture topics for the first hour of every day; theory, mathematics, research concepts, and pseudo code only.
