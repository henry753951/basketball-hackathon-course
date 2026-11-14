# Python Basketball Analytics Workshop

A five-day course for students who know basic Python but are new to computer vision and sports analytics. The course covers image coordinates, homography, object detection, tracking, bird's-eye-view (BEV) projection and shooting-motion analysis.

- **Days 1–3:** image coordinates, homography, detection, ByteTrack and court-position data.
- **Day 4:** close-up shooting video, human pose, ball trajectories and release timing.
- **Day 5:** a project proposal or completed-project demonstration explaining the problem, methods, evaluation, limitations and next steps.

Notebooks contain the main teaching workflow. Shared utilities are in `src/`.

## Start Here

1. Open the setup notebook for your language: [English](init_colab_en.ipynb), [Japanese](init_colab_ja.ipynb) or [Traditional Chinese](init_colab.ipynb).
2. In Colab, select **Runtime → Change runtime type → T4 GPU**. Another NVIDIA GPU is also suitable. Do not select TPU for this course's Ultralytics/PyTorch workflow.
3. Run setup to synchronize the course to `MyDrive/basketball_hackathon/course/` in Google Drive.
4. Open the notebooks from the course folder in Drive, following the order below.

The English and Japanese versions include the explanations, code comments, display labels and messages. Shared interactive tools and course messages follow the language selected by the notebook. Program identifiers, model classes, data paths and exported CSV/JSON field names remain compatible across versions. Use a separate kernel when switching language versions.

## Notebooks in Teaching Order

Each row provides the same exercise in three languages. English and Japanese files include a link to open them in Colab. You can also open any version from your synchronized Google Drive course folder.

| Unit | Topic | English | Japanese | Traditional Chinese |
| --- | --- | --- | --- | --- |
| Setup | Synchronize the course and check models | [Notebook](init_colab_en.ipynb) | [Notebook](init_colab_ja.ipynb) | [Notebook](init_colab.ipynb) |
| Day 1-01 | Keypoint correspondences and homography | [Notebook](day1/d1_01_keypoint_pairing_homography_en.ipynb) | [Notebook](day1/d1_01_keypoint_pairing_homography_ja.ipynb) | [Notebook](day1/d1_01_keypoint_pairing_homography.ipynb) |
| Day 1-02 | Roboflow court keypoints and YOLO Pose fine-tuning | [Notebook](day1/d1_02_keypoint_annotation_roboflow_lab_en.ipynb) | [Notebook](day1/d1_02_keypoint_annotation_roboflow_lab_ja.ipynb) | [Notebook](day1/d1_02_keypoint_annotation_roboflow_lab.ipynb) |
| Day 1-03 | Bounding-box annotation homework | [Notebook](day1/d1_03_bbox_homework_en.ipynb) | [Notebook](day1/d1_03_bbox_homework_ja.ipynb) | [Notebook](day1/d1_03_bbox_homework.ipynb) |
| Day 2-01 | Interpret YOLO detection outputs | [Notebook](day2/d2_01_yolo26_detection_en.ipynb) | [Notebook](day2/d2_01_yolo26_detection_ja.ipynb) | [Notebook](day2/d2_01_yolo26_detection.ipynb) |
| Day 2-02 | Player footpoints and BEV projection | [Notebook](day2/d2_02_yolo_players_to_bev_en.ipynb) | [Notebook](day2/d2_02_yolo_players_to_bev_ja.ipynb) | [Notebook](day2/d2_02_yolo_players_to_bev.ipynb) |
| Day 2-03 | Integrate detection, court keypoints and BEV | [Notebook](day2/d2_03_bbox_to_bev_integration_en.ipynb) | [Notebook](day2/d2_03_bbox_to_bev_integration_ja.ipynb) | [Notebook](day2/d2_03_bbox_to_bev_integration.ipynb) |
| Day 3-01 | IoU association between adjacent frames | [Notebook](day3/d3_01_tracking_concept_iou_association_en.ipynb) | [Notebook](day3/d3_01_tracking_concept_iou_association_ja.ipynb) | [Notebook](day3/d3_01_tracking_concept_iou_association.ipynb) |
| Day 3-02 | YOLO and ByteTrack tracking | [Notebook](day3/d3_02_yolo_bytetrack_tracking_en.ipynb) | [Notebook](day3/d3_02_yolo_bytetrack_tracking_ja.ipynb) | [Notebook](day3/d3_02_yolo_bytetrack_tracking.ipynb) |
| Day 3-03 | Jersey-color team clustering | [Notebook](day3/d3_03_team_clustering_en.ipynb) | [Notebook](day3/d3_03_team_clustering_ja.ipynb) | [Notebook](day3/d3_03_team_clustering.ipynb) |
| Day 3-04 | Integrate tracking, BEV paths and team clustering | [Notebook](day3/d3_04_tracking_to_bev_mini_project_en.ipynb) | [Notebook](day3/d3_04_tracking_to_bev_mini_project_ja.ipynb) | [Notebook](day3/d3_04_tracking_to_bev_mini_project.ipynb) |
| Day 4-01 | Prepare shooting video and inspect ball tracking | [Notebook](day4/d4_01_ball_detector_bytetrack_preview_en.ipynb) | [Notebook](day4/d4_01_ball_detector_bytetrack_preview_ja.ipynb) | [Notebook](day4/d4_01_ball_detector_bytetrack_preview.ipynb) |
| Day 4-02 | Shooter pose and joint angles | [Notebook](day4/d4_02_yolo_pose_and_shot_lab_en.ipynb) | [Notebook](day4/d4_02_yolo_pose_and_shot_lab_ja.ipynb) | [Notebook](day4/d4_02_yolo_pose_and_shot_lab.ipynb) |
| Day 4-03 | Shot events, release frames and 2D launch angles | [Notebook](day4/d4_03_ball_tracking_and_release_point_lab_en.ipynb) | [Notebook](day4/d4_03_ball_tracking_and_release_point_lab_ja.ipynb) | [Notebook](day4/d4_03_ball_tracking_and_release_point_lab.ipynb) |

Day 5 introduces no additional prescribed notebook. Prepare a proposal or completed-project demonstration using the requirements below. You may assume prerequisite modules such as player tracking, team recognition, jersey numbers and BEV are available, but clearly distinguish completed work, reused tools and planned work.

## Handouts and Project Requirements

| Document | English | Japanese | Traditional Chinese |
| --- | --- | --- | --- |
| Full course handout | [PDF](output/pdf/python_basketball_hackathon_course_material_en.pdf) · [LaTeX](python_basketball_hackathon_course_material_en.tex) | [PDF](output/pdf/python_basketball_hackathon_course_material_ja.pdf) · [LaTeX](python_basketball_hackathon_course_material_ja.tex) | [PDF](output/pdf/python_basketball_hackathon_course_material.pdf) · [LaTeX](python_basketball_hackathon_course_material.tex) |
| Five-day schedule and assessment checklist | [PDF](output/pdf/workshop_schedule_updated_en.pdf) · [LaTeX](workshop_schedule_updated_en.tex) | [PDF](output/pdf/workshop_schedule_updated_ja.pdf) · [LaTeX](workshop_schedule_updated_ja.tex) | [PDF](output/pdf/workshop_schedule_updated.pdf) · [LaTeX](workshop_schedule_updated.tex) |
| Day 5 proposal and demonstration requirements | [PDF](output/pdf/day5_project_proposal_spec_en.pdf) · [LaTeX](day5/project_proposal_spec_en.tex) | [PDF](output/pdf/day5_project_proposal_spec_ja.pdf) · [LaTeX](day5/project_proposal_spec_ja.tex) | [PDF](output/pdf/day5_project_proposal_spec.pdf) · [LaTeX](day5/project_proposal_spec.tex) |

Translations have been reviewed against the Chinese originals, including task IDs, points, presentation timing, submission requirements, formulas and paths. Daily checklist scores total 100 points, with up to 10 bonus points. The separate 100-point rubric in the Day 5 requirements evaluates the project itself.

The original schedule refers to `daily_lecture_outline.tex`, which is not included in this repository. Use the full handout for theory, algorithms and formulas.

The translated PDFs were built with Tectonic 0.15.0 and visually checked. Their LaTeX sources use the macOS fonts Helvetica Neue, Menlo and Hiragino Sans W3/W6. On other operating systems, substitute available fonts before compiling.

## Local Environment

The project uses `uv` for Python dependencies and notebook validation. On Apple Silicon, the workflow supports PyTorch MPS (Metal GPU). It selects CUDA on supported NVIDIA systems and falls back to CPU when no supported GPU is available.

```bash
uv sync --dev
uv run python -m compileall src
uv run ruff check .
uv run pyright src
```

Check whether the current environment can access the Mac GPU:

```bash
uv run python -c 'import torch; print(torch.__version__); print("MPS:", torch.backends.mps.is_available())'
```

Run notebooks from the repository root using the synchronized `uv` environment:

```bash
PYTHONUTF8=1 uv run jupyter execute <notebook-path> --timeout=300 --kernel_name=python3
```

All 28 English and Japanese notebooks were executed locally with the supplied inputs using Python 3.14.7 and uv 0.12.7. The shared setup selects an available CJK font for Japanese plot labels. Google Drive mounting and authenticated Roboflow downloads were not exercised in this local validation.

Day 4 uses these supplied weights directly; neither model training nor a Roboflow API key is required:

```text
assets/models/detectors/ball_rimV8.pt
assets/models/detectors/shot_detection.pt
assets/models/pose/yolov8n-pose.pt
```

## Colab and Google Drive

The setup notebook:

1. Mounts your Google Drive.
2. Synchronizes the repository to `MyDrive/basketball_hackathon/course/`.
3. Installs the course packages.
4. Checks the three Day 4 model files and tests write access to `assets/results/`.

The default `update_preserve_assets` mode preserves student files under `assets/` that are absent from the new snapshot. For matching names, the new repository version takes precedence. Notebooks and `src/` are updated, so save a separate copy of any course code you have edited. `fresh_install` replaces the entire course folder.

Each daily notebook's first code cell mounts Drive when available, locates the course root, installs `requirements.txt` in Colab and adds the root to Python's import path.

Open Day 4 notebooks from `basketball_hackathon/course/day4/` in Drive. They check the models again and save results under `/content/drive/MyDrive/basketball_hackathon/course/assets/results/`. Images, CSV, JSON and preview videos remain in Drive after the runtime ends.

## Repository Structure

```text
course/
├── init_colab.ipynb          # Traditional Chinese setup
├── init_colab_en.ipynb       # English setup
├── init_colab_ja.ipynb       # Japanese setup
├── day1/                    # Original and translated notebooks
├── day2/
├── day3/
├── day4/
├── day5/                    # Proposal and demonstration requirements
├── assets/
│   ├── raw/                 # Student source videos and archives
│   ├── converted/           # Converted MP4 videos
│   ├── datasets/            # Roboflow exports
│   ├── models/              # Supplied trained weights
│   ├── samples/             # Sample images and JSON
│   └── results/             # Generated analysis outputs
├── src/                     # Shared utilities
├── requirements.txt
└── workshop_schedule_updated.tex
```

## Data and Sample Assets

| Path | Purpose |
| --- | --- |
| `assets/raw/` | Student-uploaded source videos or archives. |
| `assets/raw/reference_videos/` | Reference game clips for Days 1–3: detection, court keypoints, tracking and BEV. |
| `assets/converted/` | Converted MP4 videos for Day 4 and relevant Day 5 projects. |
| `assets/datasets/` | Student or instructor Roboflow detection/keypoint exports. |
| `assets/models/` | Supplied YOLO detectors, court-keypoint and pose weights. |
| `assets/samples/` | Built-in sample inputs for demonstrations, local checks and exercises before model outputs are available. |
| `assets/results/` | Generated figures, CSV, JSON and archives. Contents are excluded from version control except `.gitkeep`. |

The built-in samples include:

| File | Units | Purpose |
| --- | --- | --- |
| `sample_court_frame.png` | Days 1–3 | Camera-view court image for clicking points, homography and detection/tracking visualization. |
| `sample_bev_court.json` | Days 1–3 | Colored BEV court specification rendered by `src.geometry_utils.render_bev_court`. |

See [the assets guide](assets/README.md) and [dataset guide](assets/datasets/README.md) for details.

## Day 4 Shooting Analysis

Day 4-01 inspects ball detection and ByteTrack. Day 4-02 uses the shot detector to identify the shooter and exports bilateral pose angles. Day 4-03 starts an event-specific ball path near the shooter's wrist, then estimates the release frame, joint angles at release and a 2D launch angle.

Inspect the single-frame ball boxes, skeleton, event ball path and release images before interpreting the numbers. Confidence is not a guaranteed accuracy rate. Small size, rapid motion and occlusion make ball detection difficult; brief gaps may be predicted or interpolated. The launch angle is an image-plane estimate, not a calibrated 3D measurement.

## Downloading Roboflow Datasets

The Day 1 annotation notebooks can use the official Roboflow Python SDK to download your own dataset:

- **Day 1-03:** a `yolov8` detection export to `assets/datasets/roboflow_bbox_yolo/`.
- **Day 1-02:** a `coco` keypoint export to `assets/datasets/roboflow_court_coco/`, followed by conversion to `assets/datasets/roboflow_court_yolo_pose/`.
- Existing downloads and conversions are reused by default. Set `FORCE_DOWNLOAD = True` or `FORCE_CONVERSION = True` to repeat the relevant step.
- If the SDK is missing in Colab, rerun the initial setup cell to install `requirements.txt`.

After editing annotations on the Roboflow website, create a dataset version under `Versions` using `Generate New Version` / `Publish`. The API downloads published versions, not unversioned annotation edits.

1. Extract `assets/student_uploads/roboflow_pose_student_upload_images.zip` or `assets/student_uploads/roboflow_bbox_student_upload_images.zip`.
2. In the Roboflow sidebar, select `DATA` > `Upload Data` and upload the five images to your project.
3. Select `DATA` > `Annotate`, then `Annotate Images` under `Unassigned`, and complete the annotations.
4. Create and publish a new dataset version under `Versions`.
5. Note its `workspace slug`, `project slug` and `version number`.
6. Set `USE_ROBOFLOW_DOWNLOAD = True` in the notebook and enter `ROBOFLOW_WORKSPACE`, `ROBOFLOW_PROJECT` and `ROBOFLOW_VERSION`.
7. Leave `ROBOFLOW_API_KEY` empty to enter it securely with `getpass()` at runtime.

The pose upload archive contains:

```text
boston-celtics-new-york-knicks-game-1-q1-01_54-01_48_mp4-0000.jpg
boston-celtics-new-york-knicks-game-1-q1-01_54-01_48_mp4-0001.jpg
boston-celtics-new-york-knicks-game-1-q1-01_54-01_48_mp4-0002.jpg
boston-celtics-new-york-knicks-game-1-q1-01_54-01_48_mp4-0003.jpg
boston-celtics-new-york-knicks-game-1-q1-01_54-01_48_mp4-0004.jpg
```

## Uploading Student Video

Start with the Day 4-01 notebook for your language. Choose either method:

1. **Google Drive, recommended:** open `My Drive/basketball_hackathon/course/assets/raw/`, upload an `.mp4`, `.mov`, `.m4v` or another supported video, and enter its filename in `VIDEO_FILENAME`.
2. **Upload through Colab:** set `USE_BROWSER_UPLOAD = True`, run the cell and select a video from your computer. It is saved to Drive's `assets/raw/` before conversion.

Both methods produce:

```text
assets/converted/student_video.mp4
```

With the default settings, the notebook uses `assets/converted/video_001.mp4`. Day 4-02 and Day 4-03 must show the same `using video` path as Day 4-01. Images, CSV, JSON and overlay videos are saved in `assets/results/`.

## File and Folder Names

Use ASCII folder and video filenames. Display titles may be localized; the actual course path remains:

```text
basketball_hackathon/course/
```
