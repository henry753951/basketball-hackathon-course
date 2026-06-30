# Assets 說明

本資料夾存放課程使用的輸入資料、範例素材與 Notebook 輸出結果。

## 子資料夾

| 路徑 | 說明 |
| --- | --- |
| `raw/` | 學生自行上傳的原始影片或壓縮檔。 |
| `converted/` | Notebook 轉檔後的 MP4 影片。Day 4 與 Day 5 會優先讀取此資料夾。 |
| `samples/` | 課程內建範例素材。用於課堂示範、本機驗證，以及尚未取得學生影片或模型輸出時的替代輸入。 |
| `results/` | Notebook 執行後產生的圖片、CSV、JSON 與壓縮檔。此資料夾內容不納入版本控制。 |

## samples 檔案

| 檔案 | 說明 |
| --- | --- |
| `sample_court_frame.png` | 球場相機視角範例圖；用於座標點選、Homography、Detection 與 Tracking 視覺化。 |
| `sample_bev_court.png` | 鳥瞰圖球場底圖；用於 BEV 投影點與球員移動路徑。 |
| `sample_homography_points.json` | 相機座標與 BEV 座標的對應點，並包含單一球員 bbox 範例。 |
| `sample_detections_frame0.json` | 單張影像的 detection 範例輸出，包含 class、confidence 與 bbox。 |
| `sample_tracking_boxes.json` | 多影格 bbox 範例；用於 IoU association、track ID 與 BEV 路徑投影。 |
| `sample_ball_motion.mp4` | 籃球運動短片範例；當尚未上傳學生影片時，用於球軌跡追蹤。 |

## 使用規範

- 學生自行拍攝或下載的原始影片放入 `raw/`。
- 經 Notebook 轉檔後的影片放入 `converted/`。
- Notebook 產生的結果放入 `results/`。
- `samples/` 內檔案為課程範例資料，除非教師另有指示，請勿覆蓋。
