"""Rewrite Day 1--3 notebooks as beginner-oriented teaching articles.

This script intentionally edits notebook JSON through nbformat-like dictionaries so that
the instructional text remains reviewable here instead of being hidden in manual Jupyter edits.
Run it from the repository root with ``python scripts/revise_course_notebooks.py``.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def lines(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save(path: str, notebook: dict) -> None:
    # Notebook format 4.5 requires stable cell IDs. Deriving them from path, position, and
    # source keeps reruns deterministic and avoids nbformat's MissingIDFieldWarning.
    for index, cell in enumerate(notebook.get("cells", [])):
        seed = f"{path}:{index}:{''.join(cell.get('source', []))}".encode("utf-8")
        cell["id"] = "cell-" + hashlib.sha1(seed).hexdigest()[:12]
    (ROOT / path).write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


INTRODUCTIONS = {
    "day1/d1_01_keypoint_pairing_homography.ipynb": """
# Day 1-01｜從畫面座標走到球場座標：Keypoint 配對與 Homography

> 我們先回答一個最實際的問題：**轉播畫面裡的一個點，在真正的球場上位於哪裡？**  
> 這一單元不預設電腦視覺背景；只要會執行 Python cell、看懂 list 與函式呼叫即可。

## 我們會完成什麼

- 在相機畫面與鳥瞰球場圖（Bird's-Eye View, **BEV**）上指出同一個實體位置。
- 用至少四組對應點求出 Homography（平面透視轉換）矩陣。
- 用重投影誤差檢查標定是否可信，而不是只看程式有沒有跑完。
- 把球員偵測框底邊中心投影到 BEV，建立後續位置分析的入口。

## 先認識本單元名詞

- **Keypoint（關鍵點）**：容易在兩張圖中辨認的固定位置，例如罰球線端點。
- **Pair（配對）**：同一個實體位置在相機圖與 BEV 圖中的兩組座標。
- **Homography（單應性）**：描述同一平面在兩種視角間如何轉換的 $3\\times3$ 矩陣。
- **Reprojection error（重投影誤差）**：把點投影後，預測位置與人工指定位置相差多遠。
""",
    "day1/d1_02_keypoint_annotation_roboflow_lab.ipynb": """
# Day 1-02｜把球場位置教給模型：Roboflow Keypoint 標註與 YOLO Pose 微調

> 上一單元由我們手動配對少量球場點；這一單元進一步整理成資料集，讓模型學習在新畫面中自動找球場關鍵點。  
> 我們會先理解資料格式，再決定是否訓練；課堂上不必把「成功訓練大型模型」當成唯一成果。

## 我們會完成什麼

- 看懂圖片、標註檔與 `data.yaml` 各自扮演的角色。
- 理解 COCO keypoint 格式與 YOLO Pose 格式為什麼需要轉換。
- 分辨「從通用權重開始訓練」與「從課程權重繼續微調」的差別。
- 用單張圖與短影片檢查模型輸出，觀察漏點、錯點與信心分數。

## 先認識本單元名詞

- **Dataset（資料集）**：影像與正確答案（標註）的集合。
- **Annotation（標註）**：我們提供給模型學習的正確位置與類別。
- **Pose model**：輸出一組有固定順序關鍵點的模型；此處的「pose」指球場結構，不是人體姿勢。
- **Fine-tuning（微調）**：從已學過其他資料的權重繼續訓練，通常比完全從零開始省時。
""",
    "day1/d1_03_bbox_homework.ipynb": """
# Day 1-03｜課後銜接任務：建立球員偵測的 Bounding Box 資料

> 這份 notebook 的重點是**準備 Day 2 會用到的資料**，不是提前重講完整 YOLO 推論。  
> 我們會完成少量圖片的框選、資料版本發布與格式檢查；Day 2 再專注觀察模型如何輸出偵測框。

## 我們會完成什麼

- 為球員、球、籃框等物件定義一致的類別名稱。
- 在 Roboflow 上傳並標註課程提供的少量圖片。
- 發布 dataset version，下載成 YOLO detection 格式並檢查資料夾。
- 選做：使用老師提供的模型跑一張驗證圖，確認資料流程已接通。

## 先認識本單元名詞

- **Bounding box / BBOX（邊界框）**：用左上角與右下角包住物件的矩形。
- **Class（類別）**：框內物件的名稱，例如 `player` 或 `ball`。
- **Dataset version（資料集版本）**：固定當下標註與切分的可重現快照；只有發布後才能由 API 下載。
- **Homework 與 Day 2 的分工**：今天產生資料，明天解讀模型輸出，兩者不重複。
""",
    "day2/d2_01_yolo26_detection.ipynb": """
# Day 2-01｜讀懂模型看見了什麼：YOLO Detection 推論

> 我們承接 Day 1 的標註作業，但不再重複上傳與畫框流程。這一單元改站在「模型輸出」的角度，逐一讀懂每個偵測框。  
> 你不需要先知道神經網路的所有細節；先把輸入、輸出與錯誤案例看清楚，就能建立可靠的實作直覺。

## 我們會完成什麼

- 載入已訓練的 YOLO 權重，對影片中的一個 frame 執行推論。
- 讀懂 `class_name`、`confidence` 與 `bbox_xyxy`。
- 對短片段輸出偵測預覽，觀察框如何隨時間變化。
- 說明 confidence 門檻如何影響漏偵與誤偵。

## 先認識本單元名詞

- **Inference（推論）**：把新影像交給已訓練模型，取得預測結果。
- **Confidence（信心分數）**：模型對這筆預測有多確定；它不是正確率保證。
- **`xyxy`**：`[左, 上, 右, 下]` 四個像素座標。
- **Threshold（門檻）**：只保留信心分數高於指定值的預測。
""",
    "day2/d2_02_yolo_players_to_bev.ipynb": """
# Day 2-02｜從球員框到戰術板：Footpoint 與 BEV 投影

> YOLO 告訴我們球員在畫面中的矩形範圍，但戰術分析更關心「球員站在球場哪裡」。  
> 我們會把每個框的底邊中心當作腳接觸地面的近似點，再沿用 Day 1 的 Homography 投影到 BEV。

## 我們會完成什麼

- 從每個 player BBOX 計算 bottom-center footpoint。
- 清楚區分「偵測誤差」與「Homography 標定誤差」。
- 將所有球員一次投影到 BEV，並用相同標籤對照原圖與鳥瞰圖。
- 輸出可供後續追蹤使用的結構化 JSON。

## 先認識本單元名詞

- **Footpoint（落腳點）**：以偵測框底邊中心近似球員與地面的接觸位置。
- **Projection（投影）**：用 Homography 把相機像素座標轉成 BEV 座標。
- **誤差鏈**：框沒有貼準、腳點近似不準、球場標定不準，都會讓最後的 BEV 點偏移。
""",
    "day2/d2_03_bbox_to_bev_integration.ipynb": """
# Day 2-03｜讓整段影片自動運作：Detection、球場 Keypoint 與 BEV 整合

> 前一單元使用一張畫面與一組固定 Homography；現在我們把流程延伸到影片，讓每個 frame 都重新偵測球員與球場結構。  
> 這是第一次把多個模型與幾何步驟串成 pipeline（處理流程）。

## 我們會完成什麼

- 對每個 frame 執行球員偵測與球場 keypoint 偵測。
- 從當前球場 keypoints 估計 Homography，再投影球員 footpoints。
- 輸出「原始視角 + BEV」左右並排影片與逐 frame JSON。
- 看懂本流程仍缺少跨 frame 身分，因此自然銜接 Day 3 tracking。

## 先認識本單元名詞

- **Pipeline（處理流程）**：前一步輸出成為後一步輸入的一連串運算。
- **Per-frame（逐影格）**：每張畫面獨立處理；此時相鄰畫面中的同一人還沒有共同 ID。
- **Fallback（備援）**：當某一 frame 的 keypoints 不足時，暫時沿用上一個可靠結果以降低閃爍。
""",
}


GPU_NOTE = """
## 執行環境提醒

- 建議在 Colab 選擇 **GPU** 執行階段；Ultralytics / PyTorch 不會在本課程設定下直接使用 TPU。
- 沒有 GPU 仍可用 CPU 執行，但模型推論與影片輸出會比較久。先把 `MAX_FRAMES` 調小，就能快速確認流程。
- 每格執行前先讀「這一格要做什麼」，再看輸出是否符合預期；不要只以「沒有紅字」判斷成功。
"""


CODE_PURPOSES = {
    3: "定位課程資料夾、必要時取得 repo，並載入共用的課程環境。",
}


def improve_day1_day2() -> None:
    for path, intro in INTRODUCTIONS.items():
        nb = load(path)
        nb["cells"][0] = markdown(intro)
        if len(nb["cells"]) > 1 and nb["cells"][1]["cell_type"] == "markdown":
            nb["cells"][1] = markdown(GPU_NOTE)

        # Every code cell begins with a plain-language purpose. This helps beginners decide
        # what to inspect without turning every line into a distracting comment wall.
        previous_heading = "執行本段資料處理"
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "markdown":
                text = "".join(cell.get("source", []))
                headings = [line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("##")]
                if headings:
                    previous_heading = headings[-1]
                continue
            source = "".join(cell.get("source", []))
            if source.startswith("# 這一格要做什麼："):
                continue
            purpose = CODE_PURPOSES.get(index, previous_heading)
            cell["source"] = lines(
                f"# 這一格要做什麼：{purpose}\n"
                "# 建議先執行原始設定；確認輸出後，再一次只改一個參數觀察差異。\n"
                + source
            )

        # Make the homework-to-Day-2 boundary explicit in both notebooks.
        if path.endswith("d1_03_bbox_homework.ipynb"):
            nb["cells"][2] = markdown("""
## 工作坊流程（資料準備，不重講推論）

1. 解壓課程提供的 5 張圖片並上傳 Roboflow。
2. 依統一類別規格畫 BBOX；每個框要貼近物件，但不要切掉身體主要部分。
3. 在 `Versions` 建立並發布 dataset version。
4. 回到 notebook 下載並檢查 `data.yaml`、images、labels 是否成對。
5. 選做快速預覽：只確認模型與資料能連通；confidence 與錯誤案例留到 Day 2 詳談。

### 交付檢查

- 至少 5 張圖片完成標註。
- 類別名稱與課程規格一致。
- 能說明一個「框太鬆、框太緊或類別錯誤」會造成的問題。
""")
        elif path.endswith("d2_01_yolo26_detection.ipynb"):
            nb["cells"][2] = markdown("""
## 工作坊流程（承接 Day 1）

1. 直接使用課程提供的已訓練權重；不重做 Roboflow 上傳與標註。
2. 先看單張 frame，逐欄解讀每個偵測結果。
3. 比較影像上的框與 DataFrame 中的 `bbox_xyxy` 是否對得起來。
4. 再處理短片段，觀察漏偵、重複框與 confidence 波動。
5. 為 Day 3 留下問題：相鄰 frame 的同一位球員，要如何得到同一個 ID？
""")
        elif path.endswith("d2_02_yolo_players_to_bev.ipynb"):
            # Reuse the working Day 1 sample instead of leaving an invalid TODO string that
            # causes json.loads() to fail before students can inspect the projection flow.
            day1_pair_source = "".join(
                load("day1/d1_01_keypoint_pairing_homography.ipynb")["cells"][7]["source"]
            ).replace("PAIR_DATA_JSON", "HOMOGRAPHY_PAIR_JSON")
            nb["cells"][4] = code(day1_pair_source)
            nb["cells"][5] = markdown("""
### 先使用可執行的範例配對，再換成自己的結果

下一格已放入 Day 1 範例 frame 對應的合法 JSON，所以整份 notebook 可以先直接執行。完成第一次投影後，再把 `HOMOGRAPHY_PAIR_JSON` 的內容換成你在 Day 1 工具匯出的結果。

若改用不同相機、不同影片或相機有移動，必須重新標定；相機像素點只對該視角有效，不能因為是同一座球場就直接沿用。
""")
        save(path, nb)


def improve_iou_notebook() -> None:
    path = "day3/d3_01_tracking_concept_iou_association.ipynb"
    nb = load(path)
    bootstrap = nb["cells"][3]
    bootstrap["source"] = lines(
        "# 這一格要做什麼：定位 repo、準備課程環境並載入共用模組。\n"
        + "".join(bootstrap.get("source", [])).removeprefix(
            "# 這一格要做什麼：定位 repo、準備課程環境並載入共用模組。\n"
        )
    )
    nb["cells"] = [
        markdown("""
# Day 3-01｜兩張畫面中的框，哪一些屬於同一位球員？IoU 關聯

> Day 2 已經能在每個 frame 找到球員，但模型尚不知道 `Frame 15` 的某個框與 `Frame 16` 的哪個框是同一人。  
> 我們先用最簡單、可完全看懂的 IoU 配對建立 tracking 直覺，再於下一單元交給 ByteTrack 處理長影片。

## 我們會完成什麼

- 為前後兩張圖的框命名為 `A0, A1, ...` 與 `B0, B1, ...`。
- 手動拆解交集、聯集與 IoU，建立所有框的兩兩比較矩陣。
- 用 heatmap 看清楚每個 A 框和所有 B 框的關係。
- 用全域 greedy 規則建立一對一配對，最後把配對線畫回影像。

## 先認識本單元名詞

- **Association（關聯）**：判斷不同 frame 的兩筆 detection 是否屬於同一目標。
- **IoU（Intersection over Union）**：兩個矩形的交集面積除以聯集面積，範圍為 0 到 1。
- **Greedy（貪婪法）**：每次先選目前分數最高且不衝突的候選配對。
"""),
        markdown("""
## 本單元定位

- 我們只比較相鄰兩個 frame，目的是看懂關聯過程，不是重做完整 ByteTrack。
- 這一格運算量不大，CPU 就能完成；若 YOLO 推論較慢，可降低 `IMGSZ`。
- 請同時看「影像、矩陣、配對線」，不要只看最後的 assignment 表格。
"""),
        markdown("""
## 工作坊流程

1. 對相鄰兩個 frame 執行 detector，只保留球員框。
2. 在圖上標出 `A0...` 與 `B0...`，讓矩陣的列與欄能回到真實框。
3. 計算所有 `A_i` 與 `B_j` 的 IoU，畫成 heatmap。
4. 將所有候選依 IoU 由大到小排序，建立不重複的一對一配對。
5. 把配對線畫回兩張圖，討論快速移動、遮擋與框抖動為何會失敗。
"""),
        bootstrap,
        markdown("""
## Step 1｜選擇影片、模型與相鄰 frame

`FRAME_B = FRAME_A + 1` 表示我們只跨一個 frame。兩張圖時間越接近，同一位球員的框通常重疊越多，IoU 關聯也越容易成立。
"""),
        code("""
# 這一格要做什麼：設定資料來源與實驗參數。
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import pandas as pd

from src.yolo_utils import (
    PLAYER_CLASS_NAMES,
    detector_model_path,
    first_reference_video,
    read_video_frame,
    run_detector_on_image,
)

VIDEO_PATH = first_reference_video(COURSE_ROOT)
MODEL_PATH = detector_model_path(COURSE_ROOT)
FRAME_A = 15
FRAME_B = FRAME_A + 1  # 只比較下一個 frame，先把問題簡化。
CONF = 0.25            # 低於此 confidence 的 detection 不會保留。
IMGSZ = 960            # 推論影像尺寸；較小會較快，但小物件可能更難偵測。

print("video:", VIDEO_PATH)
print("model:", MODEL_PATH)
print("frames:", FRAME_A, "->", FRAME_B)
"""),
        markdown("""
## Step 2｜先把每個偵測框編號

`A2` 不是球員的永久身分，只表示「Frame A 的第 3 個框」；`B2` 也只是 Frame B 的第 3 個框。真正的跨影格身分要等關聯後才能建立。
"""),
        code("""
# 這一格要做什麼：讀取兩張圖、執行偵測，並只留下 player 類別。
frame_a = read_video_frame(VIDEO_PATH, FRAME_A)
frame_b = read_video_frame(VIDEO_PATH, FRAME_B)

dets_a, _ = run_detector_on_image(
    MODEL_PATH, frame_a, conf=CONF, imgsz=IMGSZ, frame_index=FRAME_A
)
dets_b, _ = run_detector_on_image(
    MODEL_PATH, frame_b, conf=CONF, imgsz=IMGSZ, frame_index=FRAME_B
)

players_a = [det for det in dets_a if det.class_name in PLAYER_CLASS_NAMES]
players_b = [det for det in dets_b if det.class_name in PLAYER_CLASS_NAMES]

print(f"Frame A 有 {len(players_a)} 個 player boxes")
print(f"Frame B 有 {len(players_b)} 個 player boxes")
"""),
        code("""
# 這一格要做什麼：將矩陣標籤 A0/B0 畫回真正的偵測框。
def draw_indexed_boxes(ax, image, detections, prefix, title):
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    for index, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox_xyxy
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor="#00E5FF", linewidth=2.5))
        ax.text(x1, max(0, y1 - 6), f"{prefix}{index}", color="black",
                fontsize=11, weight="bold",
                bbox={"facecolor": "#00E5FF", "alpha": 0.9, "pad": 2})

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
draw_indexed_boxes(axes[0], frame_a, players_a, "A", f"Frame A = {FRAME_A}")
draw_indexed_boxes(axes[1], frame_b, players_b, "B", f"Frame B = {FRAME_B}")
plt.tight_layout()
plt.show()
"""),
        markdown("""
### 如何讀上圖

稍後矩陣中的「第 `A1` 列、第 `B3` 欄」，就是在問：**左圖 A1 的框與右圖 B3 的框有多少比例重疊？** 先建立這個圖像對照，矩陣就不會只是一堆抽象數字。
"""),
        markdown(r"""
## Step 3｜把 IoU 拆成可檢查的幾何步驟

若框 $A=(a_{x1},a_{y1},a_{x2},a_{y2})$、框 $B=(b_{x1},b_{y1},b_{x2},b_{y2})$，先取兩框交集矩形：

$$
x_{\mathrm{left}}=\max(a_{x1},b_{x1}),\quad
y_{\mathrm{top}}=\max(a_{y1},b_{y1})
$$

$$
x_{\mathrm{right}}=\min(a_{x2},b_{x2}),\quad
y_{\mathrm{bottom}}=\min(a_{y2},b_{y2})
$$

接著計算

$$
\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}
=\frac{\text{intersection}}{\text{area}_A+\text{area}_B-\text{intersection}}.
$$

- `0`：完全沒有重疊。
- 接近 `1`：兩框幾乎位於相同位置。
- IoU 只看空間重疊，不知道球衣顏色、移動方向或球員外觀。
"""),
        code("""
# 這一格要做什麼：逐步計算單一 pair 的 IoU，再建立所有 A/B 組合的矩陣。
def iou(box_a, box_b):
    '''Return the intersection-over-union of two [x1, y1, x2, y2] boxes.'''
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # 交集矩形的左上角要取較大的座標，右下角要取較小的座標。
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection_width = max(0.0, ix2 - ix1)
    intersection_height = max(0.0, iy2 - iy1)
    intersection = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0

# matrix[i, j] 表示 A_i 與 B_j 的 IoU；每個 A 框都會和每個 B 框比較。
matrix = np.zeros((len(players_a), len(players_b)), dtype=float)
for i, det_a in enumerate(players_a):
    for j, det_b in enumerate(players_b):
        matrix[i, j] = iou(det_a.bbox_xyxy, det_b.bbox_xyxy)

matrix_df = pd.DataFrame(
    np.round(matrix, 3),
    index=[f"A{i}" for i in range(len(players_a))],
    columns=[f"B{j}" for j in range(len(players_b))],
)
matrix_df
"""),
        code("""
# 這一格要做什麼：用顏色深淺顯示所有偵測框之間的關係。
fig, ax = plt.subplots(figsize=(max(6, len(players_b) * 0.8), max(4, len(players_a) * 0.65)))
heatmap = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(players_b)), labels=[f"B{j}" for j in range(len(players_b))])
ax.set_yticks(range(len(players_a)), labels=[f"A{i}" for i in range(len(players_a))])
ax.set_xlabel("Frame B detections")
ax.set_ylabel("Frame A detections")
ax.set_title("Pairwise IoU：每個 A 框與每個 B 框的重疊關係")
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        value = matrix[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                color="white" if value > 0.55 else "black")
fig.colorbar(heatmap, ax=ax, label="IoU (0=no overlap, 1=same box)")
plt.tight_layout()
plt.show()
"""),
        markdown("""
## Step 4｜先列出每個 A 框的最佳候選

每一列最大值回答「這個 A 框最像哪一個 B 框？」但它還不是最終配對，因為不同 A 框可能同時把同一個 B 框當成最佳答案。一對一限制會在下一步處理。
"""),
        code("""
# 這一格要做什麼：只做候選診斷，尚未套用一對一限制。
best_match_rows = []
for i in range(matrix.shape[0]):
    if matrix.shape[1] == 0:
        break
    j = int(matrix[i].argmax())
    best_match_rows.append({
        "frame_a_box": f"A{i}",
        "best_frame_b_box": f"B{j}",
        "best_iou": float(matrix[i, j]),
        "passes_threshold": bool(matrix[i, j] >= 0.30),
    })

best_match_df = pd.DataFrame(best_match_rows)
best_match_df
"""),
        markdown("""
## Step 5｜用全域 greedy 規則建立一對一配對

1. 列出所有高於門檻的 `(A_i, B_j)` 候選。
2. 依 IoU 從大到小排序。
3. 每次取目前最高分；若 A 或 B 已被使用，就跳過。
4. 直到沒有候選可選。

這仍不是 ByteTrack，但比逐列直接決定更公平：不會因為 `A0` 比 `A1` 先出現在迴圈，就搶走其實更適合 `A1` 的 B 框。
"""),
        code("""
# 這一格要做什麼：從所有 pair 中，依分數建立不衝突的一對一配對。
IOU_THRESHOLD = 0.30
candidates = [
    (float(matrix[i, j]), i, j)
    for i in range(matrix.shape[0])
    for j in range(matrix.shape[1])
    if matrix[i, j] >= IOU_THRESHOLD
]
candidates.sort(reverse=True)  # Python tuple 會先依第一個欄位（IoU）排序。

assignments = []
used_a, used_b = set(), set()
for score, i, j in candidates:
    if i in used_a or j in used_b:
        continue
    used_a.add(i)
    used_b.add(j)
    assignments.append({
        "frame_a_box": f"A{i}",
        "frame_b_box": f"B{j}",
        "iou": round(score, 3),
        "a_index": i,
        "b_index": j,
    })

assignment_df = pd.DataFrame(assignments)
assignment_df[["frame_a_box", "frame_b_box", "iou"]] if not assignment_df.empty else assignment_df
"""),
        code("""
# 這一格要做什麼：把 assignment 表格中的關係畫回兩張圖。
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
draw_indexed_boxes(axes[0], frame_a, players_a, "A", f"Frame {FRAME_A}")
draw_indexed_boxes(axes[1], frame_b, players_b, "B", f"Frame {FRAME_B}")

# 透過公開 API 取得 colormap，避免型別檢查器把動態屬性 `plt.cm.tab10` 視為未知。
colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(assignments))))
for color, pair in zip(colors, assignments):
    box_a = players_a[pair["a_index"]].bbox_xyxy
    box_b = players_b[pair["b_index"]].bbox_xyxy
    center_a = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2)
    center_b = ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
    connection = ConnectionPatch(
        xyA=center_b, coordsA=axes[1].transData,
        xyB=center_a, coordsB=axes[0].transData,
        color=color, linewidth=2.5, alpha=0.9,
    )
    fig.add_artist(connection)
    axes[0].text(*center_a, f" IoU={pair['iou']:.2f}", color=color, weight="bold")

fig.suptitle("Greedy assignments：同色連線代表被視為同一位球員", fontsize=14)
plt.tight_layout()
plt.show()
"""),
        markdown("""
## 從簡化 IoU 走向 ByteTrack

純 IoU 在球員快速移動、彼此交錯、被遮擋或 detector 漏偵時容易失敗。ByteTrack 會維護既有 track 狀態，並分兩階段使用高、低 confidence detections，嘗試讓短暫不確定的目標仍能接回原 ID。下一單元會直接觀察 `track_id` 與軌跡。

### 請你回答

1. heatmap 中是否有一列同時出現兩個相近的高分？這代表什麼不確定性？
2. 把 `IOU_THRESHOLD` 改成 `0.10` 或 `0.60`，配對數量如何改變？
3. 哪一種籃球畫面情況會讓正確配對的 IoU 反而很低？
"""),
        markdown("""
## 本單元產出

- 有 A/B 編號的前後 frame 對照圖。
- 所有 detection pairs 的 IoU heatmap。
- 一對一 assignment 表格與跨圖配對線。

這一單元刻意不輸出影片；我們先把兩個 frame 的關係看清楚，再進入長時間追蹤。
"""),
    ]
    save(path, nb)


def improve_bytetrack_notebook() -> None:
    path = "day3/d3_02_yolo_bytetrack_tracking.ipynb"
    nb = load(path)
    bootstrap = nb["cells"][3]
    bootstrap["source"] = lines(
        "# 這一格要做什麼：定位 repo、準備課程環境並載入共用模組。\n"
        + "".join(bootstrap.get("source", [])).removeprefix(
            "# 這一格要做什麼：定位 repo、準備課程環境並載入共用模組。\n"
        )
    )
    nb["cells"] = [
        markdown("""
# Day 3-02｜讓同一位球員跨畫面保留身分：YOLO + ByteTrack

> 前一單元只比較兩張圖；現在我們把 detection 連成長一點的時間序列。  
> 畫面上的 `#7` 不是背號，也不代表真實姓名，而是 tracker 在這段影片暫時分配的 `track_id`。

## 我們會完成什麼

- 理解 detector 與 tracker 的分工：前者找框，後者連接時間關係。
- 輸出帶有 BBOX、`track_id` 與最近移動軌跡的影片。
- 將每個 frame 的追蹤結果整理為 DataFrame。
- 用持續 frame 數與 confidence 檢查 ID 是否穩定。

## 先認識本單元名詞

- **Multi-object tracking（多目標追蹤）**：同時維持多個目標在時間上的身分。
- **Track ID**：tracker 在單段影片內分配的暫時識別碼；影片重跑後可能不同。
- **Trajectory / trace（軌跡）**：同一 track 最近幾個位置連成的線。
- **ID switch（身分交換）**：兩位球員交錯後，tracker 把彼此的 ID 接反。
"""),
        markdown("""
## 本單元定位

- 我們直接使用現成 ByteTrack，不要求學生重寫完整演算法。
- 請把注意力放在輸入 detection、輸出 track 與失敗情況三者的關係。
- 影片較慢時先把 `MAX_FRAMES` 改成 `30`；確認流程後再增加。
"""),
        markdown("""
## 工作坊流程

1. 設定影片、模型與推論參數。
2. 執行 `YOLO detection -> ByteTrack association -> 視覺化`。
3. 在影片中觀察同一個 ID 的框與短軌跡是否連續。
4. 讀取 JSON/DataFrame，理解每列代表「某個 track 在某個 frame 的狀態」。
5. 統計各 ID 的生命週期，找出太短或可能交換的 track。
"""),
        bootstrap,
        markdown("""
## Step 1｜設定輸入、輸出與運算範圍

`START_FRAME` 決定從影片哪裡開始，`MAX_FRAMES` 決定處理多久。先選短片段能讓我們更快反覆調整參數。
"""),
        code("""
# 這一格要做什麼：集中設定資料路徑與可調整參數。
import pandas as pd

from src.video_utils import display_video_in_notebook
from src.yolo_utils import (
    detector_model_path,
    first_reference_video,
    write_bytetrack_preview_video,
)

VIDEO_PATH = first_reference_video(COURSE_ROOT)
MODEL_PATH = detector_model_path(COURSE_ROOT)
OUTPUT_PATH = COURSE_ROOT / "assets" / "results" / "d3_02_bytetrack_preview.mp4"
START_FRAME = 0
MAX_FRAMES = 120
CONF = 0.25
IMGSZ = 960

print("video:", VIDEO_PATH)
print("model:", MODEL_PATH)
print("output:", OUTPUT_PATH)
print(f"frame range: {START_FRAME} .. {START_FRAME + MAX_FRAMES - 1}")
"""),
        markdown("""
## Step 2｜執行 ByteTrack 並畫出最近軌跡

共用函式在每個 frame 依序做：

```text
影像 -> YOLO 偵測框 -> ByteTrack 更新 ID -> 畫框、標籤與最近軌跡 -> 寫入影片/JSON
```

軌跡線只保留最近一小段位置，目的是讓我們看出移動方向；它不是完整比賽路徑。標籤 `#12 player 0.87` 依序代表 track ID、類別與 detection confidence。
"""),
        code("""
# 這一格要做什麼：產生追蹤預覽影片與逐 frame 結構化紀錄。
preview_video, records = write_bytetrack_preview_video(
    video_path=VIDEO_PATH,
    model_path=MODEL_PATH,
    output_path=OUTPUT_PATH,
    max_frames=MAX_FRAMES,
    conf=CONF,
    imgsz=IMGSZ,
    start_frame=START_FRAME,
)
preview_json = preview_video.with_suffix(".json")

print("preview video:", preview_video)
print("preview json:", preview_json)
print("number of track records:", len(records))
display_video_in_notebook(preview_video, loop=True)
"""),
        markdown("""
### 影片觀察清單

- 同一位球員移動時，框上的 `#ID` 是否大多保持不變？
- 球員互相遮擋後，原 ID 是重新出現、換成新 ID，還是和別人交換？
- 軌跡若突然大幅跳躍，可能是偵測框跳動，也可能是錯誤關聯。
"""),
        markdown("""
## Step 3｜把影片中的標籤讀成資料表

DataFrame 的每一列代表「某一個 `track_id` 在某一個 frame 的一次觀測」，不是一位球員的完整摘要。相同 ID 會跨多列重複出現，這正是軌跡資料的基本形式。
"""),
        code("""
# 這一格要做什麼：固定欄位順序，方便逐欄解讀追蹤紀錄。
track_columns = [
    "frame",       # 來源影片的 frame 編號
    "track_id",    # tracker 暫時分配的跨 frame 身分
    "class_id",    # 模型內部的類別編號
    "class_name",  # 人類可讀類別，例如 player
    "confidence",  # detector 對這個框的信心
    "bbox_xyxy",   # [left, top, right, bottom]
]
tracks = pd.DataFrame(records, columns=track_columns)
tracks.head(10)
"""),
        markdown("""
## Step 4｜以 track 為單位做健康檢查

- `frames_seen` 很小：可能只是誤偵，也可能是剛進入或離開畫面。
- `first_frame` 到 `last_frame` 很長，但 `frames_seen` 很少：中間可能多次漏偵。
- `mean_confidence` 低：這個 ID 的輸入框較不可靠，後續位置分析要更小心。

這些指標只能幫我們定位可疑 track，不能單獨證明追蹤正確。
"""),
        code("""
# 這一格要做什麼：把逐 frame 紀錄聚合成每個 track_id 一列的摘要。
summary_columns = [
    "track_id", "frames_seen", "first_frame", "last_frame",
    "time_span_frames", "coverage_ratio", "mean_confidence",
]
if tracks.empty:
    track_summary = pd.DataFrame(columns=summary_columns)
else:
    track_summary = (
        tracks.groupby("track_id", dropna=False)
        .agg(
            frames_seen=("frame", "count"),
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    track_summary["time_span_frames"] = (
        track_summary["last_frame"] - track_summary["first_frame"] + 1
    )
    track_summary["coverage_ratio"] = (
        track_summary["frames_seen"] / track_summary["time_span_frames"]
    )
    track_summary = track_summary.sort_values(
        ["frames_seen", "first_frame"], ascending=[False, True]
    )

track_summary[summary_columns].head(15)
"""),
        markdown("""
## 本單元產出與銜接

- `assets/results/d3_02_bytetrack_preview.mp4`：含 ID、BBOX 與短軌跡的影片。
- `assets/results/d3_02_bytetrack_preview.json`：逐 frame tracking records。
- `track_summary`：每個 ID 的持續時間、覆蓋率與平均 confidence。

下一單元會把每個 BBOX 的 footpoint 投影到 BEV；之後再用球衣色彩把球員分成兩隊。
"""),
    ]
    save(path, nb)


def improve_bev_notebook() -> None:
    path = "day3/d3_03_tracking_to_bev_mini_project.ipynb"
    nb = load(path)
    nb["cells"][0] = markdown("""
# Day 3-03｜把 Track ID 放到球場上：ByteTrack-to-BEV 小專案

> 現在我們把前三天的元件串起來：YOLO 找球員、ByteTrack 維持 ID、球場 keypoints 估計 Homography、footpoint 轉成 BEV 座標。  
> 最終每一列資料都回答：**哪一個 track，在哪一個 frame，位於球場上的哪裡？**

## 我們會完成什麼

- 輸出原始視角與 BEV 並排的追蹤影片。
- 讀懂 `track_id -> bbox -> footpoint -> bev_xy` 的資料流。
- 將逐 frame 資料匯出 CSV，建立後續速度、路徑與站位分析的共同格式。
- 分辨 Homography 暫時失效與 tracking ID 失效是兩種不同問題。
""")
    nb["cells"][1] = markdown("""
## 前情提要（只保留這次需要的部分）

- Day 1：Homography 負責「相機座標 -> BEV 座標」。
- Day 2：player BBOX 的底邊中心作為 footpoint。
- Day 3-02：ByteTrack 為相鄰 frames 中的同一目標維持 `track_id`。

本 notebook 不再重算手動 IoU，也不重講 YOLO 欄位；我們專注在三者如何接成一條資料流。
""")
    nb["cells"][2] = markdown("""
## 工作坊流程

1. 設定影片、player detector、court keypoint model 與 BEV 規格。
2. 執行 tracking-to-BEV pipeline 並觀看左右對照影片。
3. 沿著一列資料檢查 `bbox_xyxy -> foot_x/foot_y -> bev_x/bev_y`。
4. 匯出 CSV，對每個 `track_id` 計算出現範圍與平均位置。
5. 記錄限制：ID switch、漏偵、球場 keypoint 不足與投影抖動。
""")
    replacements = {
        4: """
## Step 1｜設定整合流程的四種輸入

- 比賽影片：提供每個 frame。
- Player detector：輸出球員 BBOX。
- Court keypoint model：輸出球場固定點，用來估計 Homography。
- BEV 規格：定義鳥瞰球場的大小與線條位置。
""",
        6: """
## Step 2｜執行 `detection -> tracking -> footpoint -> Homography`

`write_bytetrack_bev_video()` 內部依序執行：

```text
frame
  -> player BBOX
  -> ByteTrack track_id
  -> BBOX bottom-center footpoint
  -> court keypoints -> Homography
  -> BEV coordinate
```

若當前 frame 的球場 keypoints 不足，程式會短暫沿用上一個可靠 Homography。這能減少 BEV 閃爍，但無法修復長時間看不到球場線的片段。
""",
        8: """
## Step 3｜沿著一列資料檢查座標如何改變

- `bbox_xyxy`：模型在相機畫面找到的矩形。
- `foot_x, foot_y`：矩形底邊中心，仍是相機像素座標。
- `bev_x, bev_y`：經 Homography 後的鳥瞰球場座標。
- `keypoint_count`：當時有多少球場點可支援 Homography；數量少時要提高警覺。
""",
        10: """
## Step 4｜匯出可供研究分析的長表格

這種「每個 frame、每個 track 一列」的長表格可延伸為：

- 單一球員 BEV 路徑與移動距離。
- 速度、加速度與急停事件估計。
- 兩隊陣形、球員間距與攻守站位。
- 與投籃、傳球或回合事件的時間對齊。

注意：`track_id` 不是球員姓名或背號。若要跨片段辨認真實球員，還需要 Re-ID、背號辨識或人工校正。
""",
        12: """
## 我們完成了什麼，也還缺什麼

Day 1–3 的位置資料主幹已完成：標註格式、YOLO detection、court-keypoint Homography、ByteTrack 與 BEV 路徑輸出。下一份 notebook 會加入隊伍分群，使用球員上半身的球衣色彩把 crops 分成兩組。

仍需記得：

- Homography 錯誤會讓位置投影偏移，但不一定改變 track ID。
- Tracking 錯誤會讓路徑接到錯的人，即使 BEV 投影本身正確。
- Team clustering 只能產生 Team A / B；它不知道真實隊名，也不等同背號或球員辨識。
""",
        13: """
## 本單元產出

- `assets/results/d3_03_bytetrack_bev.mp4`：原始視角與 BEV 並排影片。
- `assets/results/d3_03_bytetrack_bev.json`：逐 frame 原始紀錄。
- `assets/results/d3_03_bytetrack_bev.csv`：可供 pandas 與研究分析使用的長表格。
""",
    }
    for index, text in replacements.items():
        nb["cells"][index] = markdown(text)

    purposes = {
        3: "準備課程環境並載入共用函式。",
        5: "指定影片、兩個模型、BEV 規格與輸出路徑。",
        7: "執行完整 tracking-to-BEV pipeline 並播放結果。",
        9: "將 JSON records 整理成欄位固定的 DataFrame。",
        11: "匯出 CSV，並以 track_id 聚合出位置摘要。",
    }
    for index, purpose in purposes.items():
        source = "".join(nb["cells"][index].get("source", []))
        if not source.startswith("# 這一格要做什麼："):
            nb["cells"][index]["source"] = lines(f"# 這一格要做什麼：{purpose}\n" + source)
    save(path, nb)


def create_team_clustering_notebook() -> None:
    bootstrap = load("day3/d3_03_tracking_to_bev_mini_project.ipynb")["cells"][3]
    nb = {
        "cells": [
            markdown("""
# Day 3-04｜球衣顏色能告訴我們哪一隊嗎？Team Clustering

> Tracking 告訴我們「是不是同一個暫時 ID」，但還不知道球員屬於哪一隊。  
> 我們參考研究專案中的 team classifier，將流程簡化成新手可讀版本：**裁切上半身 -> HSV 色彩直方圖 -> K-means 分成兩群 -> 畫回原圖。**

## 我們會完成什麼

- 從 player BBOX 只裁切較可能包含球衣的上半身區域。
- 把每張 crop 轉成 HSV 二維色彩直方圖，形成可比較的特徵向量。
- 自己實作兩群 K-means，理解「分群」不是已知隊名的分類。
- 同時看原始框、球衣 crops、特徵散點圖與 Team A/B 結果。

## 先認識本單元名詞

- **Crop（裁切圖）**：從原圖取出的局部影像。
- **Feature / embedding（特徵向量）**：把影像轉成一串可計算距離的數字。
- **HSV**：Hue（色相）、Saturation（飽和度）、Value（明度）；比直接 RGB 更方便描述球衣顏色。
- **Clustering（分群）**：沒有提供正確隊名，演算法只依相似度把資料分組。
- **K-means**：反覆進行「分配到最近中心」與「更新群中心」的分群方法。
"""),
            markdown("""
## 與參考研究程式的關係

參考專案同樣先收集 player crops，再使用色彩直方圖或影像模型 embedding，最後以 `KMeans(n_clusters=2)` 分成兩隊。本課程保留這條核心資料流，但做三項教學化簡化：

1. 單張代表 frame，讓四種視覺化可以直接互相對照。
2. 使用 HSV 直方圖，避免先引入大型影像 embedding 模型。
3. 用 NumPy 寫出兩群 K-means，讓每一步都看得見。

正式專案可從多個 frames 收集 crops，對同一 `track_id` 多次投票，結果會比單張圖穩定。
"""),
            markdown("""
## 工作坊流程

1. 選擇一個球員數量足夠、兩隊球衣可見的 frame。
2. 執行 player detector，從每個 BBOX 取上半身 crop。
3. 對照 BBOX 編號與 crop 編號，確認裁切沒有落到地板或觀眾席。
4. 將 crops 轉成 HSV histogram features。
5. 用 K-means 分成兩群，再用 PCA 僅作 2D 視覺化。
6. 把 Team A / B 顏色畫回原圖並討論失敗案例。
"""),
            bootstrap,
            markdown("""
## Step 1｜選擇代表 frame 並取得 player boxes

隊伍分群需要一張能看見兩隊球衣的畫面。若偵測到的球員少於兩人，請改 `FRAME_INDEX`；若裁切太小，可提高 `IMGSZ`。
"""),
            code("""
# 這一格要做什麼：設定 frame、模型與輸出路徑，並取得 player detections。
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from src.cv_utils import save_json
from src.yolo_utils import (
    PLAYER_CLASS_NAMES,
    detector_model_path,
    first_reference_video,
    read_video_frame,
    run_detector_on_image,
)

VIDEO_PATH = first_reference_video(COURSE_ROOT)
MODEL_PATH = detector_model_path(COURSE_ROOT)
FRAME_INDEX = 30
CONF = 0.25
IMGSZ = 960
OUTPUT_IMAGE = COURSE_ROOT / "assets" / "results" / "d3_04_team_clustering.png"
OUTPUT_JSON = OUTPUT_IMAGE.with_suffix(".json")

frame = read_video_frame(VIDEO_PATH, FRAME_INDEX)
detections, _ = run_detector_on_image(
    MODEL_PATH, frame, conf=CONF, imgsz=IMGSZ, frame_index=FRAME_INDEX
)
players = [det for det in detections if det.class_name in PLAYER_CLASS_NAMES]
if len(players) < 2:
    raise RuntimeError("此 frame 找到的球員少於 2 人，請調整 FRAME_INDEX 後重跑。")

print("video:", VIDEO_PATH)
print("frame:", FRAME_INDEX)
print("player boxes:", len(players))
"""),
            markdown("""
## Step 2｜只裁切較可能是球衣的區域

直接使用整個 BBOX 會混入球場地板、鞋子與背景。以下裁切保留框寬中間約 60%、框高上方約 10%–65%，用來近似軀幹。這不是人體姿態模型，只是一個容易理解的啟發式規則（heuristic）。
"""),
            code("""
# 這一格要做什麼：把每個 player box 轉成上半身 crop，並保留原 box 索引。
def torso_crop(image, bbox_xyxy):
    '''Crop the central upper-body region from one [x1, y1, x2, y2] box.'''
    image_h, image_w = image.shape[:2]
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    box_w, box_h = x2 - x1, y2 - y1

    # 水平排除左右背景；垂直排除頭頂與腿部，聚焦球衣常出現的位置。
    crop_x1 = int(np.clip(x1 + 0.20 * box_w, 0, image_w - 1))
    crop_x2 = int(np.clip(x2 - 0.20 * box_w, crop_x1 + 1, image_w))
    crop_y1 = int(np.clip(y1 + 0.10 * box_h, 0, image_h - 1))
    crop_y2 = int(np.clip(y1 + 0.65 * box_h, crop_y1 + 1, image_h))
    return image[crop_y1:crop_y2, crop_x1:crop_x2]

crops, valid_players, source_indices = [], [], []
for source_index, player in enumerate(players):
    crop = torso_crop(frame, player.bbox_xyxy)
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        continue  # 太小的 crop 沒有足夠顏色資訊，先排除。
    crops.append(crop)
    valid_players.append(player)
    source_indices.append(source_index)

if len(crops) < 2:
    raise RuntimeError("可用的球衣 crops 少於 2 張，請選擇球員較清楚的 frame。")
"""),
            code("""
# 這一格要做什麼：把原圖 BBOX 編號與球衣 crop 編號並排檢查。
fig = plt.figure(figsize=(16, 8))
grid = fig.add_gridspec(2, max(3, len(crops)), height_ratios=[2.4, 1])
ax_main = fig.add_subplot(grid[0, :])
ax_main.imshow(frame)
ax_main.set_title("原始 frame：P0、P1... 對應下方的 torso crops")
ax_main.axis("off")
for local_index, player in enumerate(valid_players):
    x1, y1, x2, y2 = player.bbox_xyxy
    ax_main.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                fill=False, edgecolor="#FFD166", linewidth=2.5))
    ax_main.text(x1, y1, f"P{local_index}", color="black", weight="bold",
                 bbox={"facecolor": "#FFD166", "alpha": 0.9})

for index, crop in enumerate(crops):
    ax = fig.add_subplot(grid[1, index])
    ax.imshow(crop)
    ax.set_title(f"P{index} torso")
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),
            markdown(r"""
## Step 3｜把不同大小的 crop 轉成固定長度色彩特徵

我們在 HSV 空間統計「色相 H 與飽和度 S 的組合出現幾次」，得到二維直方圖，再攤平成向量並做 $L_2$ 正規化：

$$
\hat{\mathbf{h}}=\frac{\mathbf{h}}{\lVert\mathbf{h}\rVert_2+\epsilon}.
$$

這樣每張 crop 不論原始寬高，都會變成相同長度的數字；正規化則降低 crop 大小對總計數的影響。
"""),
            code("""
# 這一格要做什麼：將每張 RGB crop 編碼成 HSV(H,S) 正規化直方圖。
def color_hist_embedding(crop_rgb, hue_bins=12, saturation_bins=8):
    '''Return a normalized 2D H/S histogram as one flat feature vector.'''
    hsv = cv2.cvtColor(np.ascontiguousarray(crop_rgb), cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], None,
        [hue_bins, saturation_bins],
        [0, 180, 0, 256],
    ).flatten().astype(np.float32)
    norm = float(np.linalg.norm(histogram))
    return histogram / max(norm, 1e-12)

features = np.vstack([color_hist_embedding(crop) for crop in crops])
print("number of crops:", len(crops))
print("feature matrix shape:", features.shape)
print("每一列是一位球員的 crop；每一欄是一個 H/S 顏色區間。")
"""),
            markdown(r"""
## Step 4｜用兩群 K-means 反覆做「分配」與「更新」

對第 $i$ 張 crop 的特徵 $\mathbf{x}_i$，分配到最近中心：

$$
c_i=\arg\min_{k\in\{0,1\}}\lVert\mathbf{x}_i-\boldsymbol{\mu}_k\rVert_2^2.
$$

再把同群特徵平均，更新中心 $\boldsymbol{\mu}_k$。重複直到分群不再改變或達到迭代上限。群編號 `0/1` 本身沒有隊名意義，所以後面只稱 Team A / Team B。
"""),
            code("""
# 這一格要做什麼：用 NumPy 實作固定兩群的 K-means，讓每一步都可閱讀。
def kmeans_two_clusters(feature_matrix, max_iterations=50):
    if len(feature_matrix) < 2:
        raise ValueError("K-means 至少需要兩筆資料。")

    # 第一個中心取 P0；第二個中心取離 P0 最遠者，避免一開始太接近。
    first_center_index = 0
    distances_from_first = np.linalg.norm(
        feature_matrix - feature_matrix[first_center_index], axis=1
    )
    second_center_index = int(np.argmax(distances_from_first))
    centers = feature_matrix[[first_center_index, second_center_index]].copy()
    labels = np.full(len(feature_matrix), -1, dtype=int)

    iterations_run = 0
    for iteration in range(max_iterations):
        iterations_run = iteration + 1
        # distances[i, k]：第 i 位球員到第 k 個群中心的距離。
        distances = np.linalg.norm(
            feature_matrix[:, None, :] - centers[None, :, :], axis=2
        )
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # 每個中心改成該群所有特徵的平均；空群則保留原中心。
        for cluster_id in (0, 1):
            members = feature_matrix[labels == cluster_id]
            if len(members) > 0:
                centers[cluster_id] = members.mean(axis=0)
    return labels, centers, iterations_run

team_labels, cluster_centers, iterations = kmeans_two_clusters(features)
print("iterations:", iterations)
print("Team A members:", np.where(team_labels == 0)[0].tolist())
print("Team B members:", np.where(team_labels == 1)[0].tolist())
"""),
            markdown("""
## Step 5｜把高維特徵壓成 2D，只為了看見群與群的距離

HSV histogram 有很多維，無法直接畫在平面上。我們用 PCA 的核心作法（中心化後做 SVD）投影到兩個方向。**K-means 使用的是原始完整特徵；2D 座標只用於視覺化，不是分群輸入。**
"""),
            code("""
# 這一格要做什麼：同時呈現 crop、2D 特徵位置與畫回原圖的隊伍結果。
centered_features = features - features.mean(axis=0, keepdims=True)
_, _, vh = np.linalg.svd(centered_features, full_matrices=False)
component_count = min(2, vh.shape[0])
feature_2d = centered_features @ vh[:component_count].T
if component_count == 1:
    feature_2d = np.column_stack([feature_2d[:, 0], np.zeros(len(feature_2d))])

team_colors = {0: "#00B7FF", 1: "#FF4D6D"}
team_names = {0: "Team A", 1: "Team B"}
fig, axes = plt.subplots(1, 2, figsize=(17, 7))

# 左圖：將分群結果畫回真正的 BBOX，確認每個框與 team label 的關係。
axes[0].imshow(frame)
axes[0].set_title("Team clustering 畫回原始 frame")
axes[0].axis("off")
result_rows = []
for index, (player, cluster_id) in enumerate(zip(valid_players, team_labels)):
    x1, y1, x2, y2 = player.bbox_xyxy
    color = team_colors[int(cluster_id)]
    axes[0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                fill=False, edgecolor=color, linewidth=3))
    axes[0].text(x1, y1, f"P{index} {team_names[int(cluster_id)]}",
                 color="white", weight="bold",
                 bbox={"facecolor": color, "alpha": 0.9})
    result_rows.append({
        "frame": FRAME_INDEX,
        "player_index": index,
        "source_detection_index": source_indices[index],
        "team_cluster": int(cluster_id),
        "team_name": team_names[int(cluster_id)],
        "bbox_xyxy": [float(value) for value in player.bbox_xyxy],
    })

# 右圖：每個點是一張 torso crop；距離近表示 HSV 色彩分布較相似。
for cluster_id in (0, 1):
    mask = team_labels == cluster_id
    axes[1].scatter(feature_2d[mask, 0], feature_2d[mask, 1], s=130,
                    color=team_colors[cluster_id], label=team_names[cluster_id], alpha=0.85)
for index, (x, y) in enumerate(feature_2d):
    axes[1].annotate(f"P{index}", (x, y), xytext=(6, 6), textcoords="offset points")
axes[1].axhline(0, color="gray", linewidth=0.6)
axes[1].axvline(0, color="gray", linewidth=0.6)
axes[1].set_title("HSV histogram features 的 2D PCA 視圖")
axes[1].set_xlabel("PCA component 1")
axes[1].set_ylabel("PCA component 2")
axes[1].legend()
axes[1].grid(alpha=0.2)

OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
fig.savefig(OUTPUT_IMAGE, dpi=160, bbox_inches="tight")
plt.show()
save_json(result_rows, OUTPUT_JSON)
print("saved image:", OUTPUT_IMAGE)
print("saved data:", OUTPUT_JSON)
"""),
            markdown("""
## 如何判讀結果與限制

請不要只問「有沒有分成兩色」，而要依序檢查：

1. **框是否正確？** 若 BBOX 包到裁判或觀眾，後面再好的分群也無法補救。
2. **crop 是否真的包含球衣？** 太多地板、皮膚或陰影會污染顏色特徵。
3. **同隊點是否在 2D 視圖靠近？** 若交錯，可能是光線、主客場球衣相似或 PCA 壓縮造成。
4. **Team A/B 是否在原圖合理？** 群編號不代表真實隊名，也不會辨認背號。

### 正式專案的延伸

- 從多個 frames 收集 crops，再以同一 `track_id` 多數決決定隊伍。
- 排除裁判，或加入「裁判」第三群。
- 使用預訓練影像 embedding 取代單純色彩直方圖。
- 結合 OCR / jersey-number model，從 team + number 進一步辨認球員。

## 本單元產出

- `assets/results/d3_04_team_clustering.png`：原圖分群與特徵散點對照。
- `assets/results/d3_04_team_clustering.json`：每個 player BBOX 的 Team A/B 結果。
"""),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    save("day3/d3_04_team_clustering.ipynb", nb)


def main() -> None:
    improve_day1_day2()
    improve_iou_notebook()
    improve_bytetrack_notebook()
    improve_bev_notebook()
    create_team_clustering_notebook()


if __name__ == "__main__":
    main()
