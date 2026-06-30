from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Sequence

from IPython.display import HTML, display


def show_click_tool(
    image_path: str | Path,
    canvas_width: int = 1000,
    title: str = "座標點選工具",
) -> None:
    """顯示可點選影像座標的 Notebook 互動工具。"""
    image_path = Path(image_path)
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    canvas_id = "canvas_" + image_path.stem.replace("-", "_")
    output_id = "output_" + image_path.stem.replace("-", "_")

    html = f"""
    <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
      <h3 style="margin-bottom: 8px;">{title}</h3>
      <p style="margin: 0 0 8px 0;">
        點選影像中的參考點；下方會即時列出原始影像座標。
      </p>
      <canvas id="{canvas_id}" style="border:1px solid #999; max-width:100%;"></canvas>
      <p style="margin: 8px 0 4px 0;">已記錄座標：</p>
      <textarea id="{output_id}" rows="5" style="width:100%; font-family:monospace;"></textarea>
      <br />
      <button onclick="window.clear_{canvas_id}()" style="margin-top: 8px;">清除座標</button>
    </div>
    <script>
    (function() {{
      const img = new Image();
      const canvas = document.getElementById("{canvas_id}");
      const ctx = canvas.getContext("2d");
      const output = document.getElementById("{output_id}");
      const points = [];

      img.onload = function() {{
        const scale = {canvas_width} / img.width;
        canvas.width = {canvas_width};
        canvas.height = Math.round(img.height * scale);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      }};

      img.src = "data:image/png;base64,{b64}";

      function redraw() {{
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "rgb(40, 120, 255)";
        ctx.strokeStyle = "white";
        ctx.lineWidth = 2;
        ctx.font = "16px sans-serif";
        for (let i = 0; i < points.length; i++) {{
          const p = points[i];
          const sx = p[0] * canvas.width / img.width;
          const sy = p[1] * canvas.height / img.height;
          ctx.beginPath();
          ctx.arc(sx, sy, 6, 0, 2 * Math.PI);
          ctx.fill();
          ctx.strokeText(String(i), sx + 8, sy - 8);
          ctx.fillText(String(i), sx + 8, sy - 8);
        }}
        output.value = JSON.stringify(points);
      }}

      canvas.addEventListener("click", function(event) {{
        const rect = canvas.getBoundingClientRect();
        const sx = event.clientX - rect.left;
        const sy = event.clientY - rect.top;
        const x = Math.round(sx * img.width / rect.width);
        const y = Math.round(sy * img.height / rect.height);
        points.push([x, y]);
        redraw();
      }});

      window.clear_{canvas_id} = function() {{
        points.length = 0;
        redraw();
      }};
    }})();
    </script>
    """
    display(HTML(html))


def show_court_keypoint_pair_tool(
    bev_image_path: str | Path,
    camera_image_path: str | Path,
    keypoints: Sequence[dict[str, Any]],
    canvas_width: int = 1000,
    title: str = "Court Keypoint Pairing Tool",
) -> None:
    """顯示上方 BEV keypoint、下方 camera frame 的配對標定工具。"""
    bev_image_path = Path(bev_image_path)
    camera_image_path = Path(camera_image_path)
    bev_b64 = base64.b64encode(bev_image_path.read_bytes()).decode("utf-8")
    camera_b64 = base64.b64encode(camera_image_path.read_bytes()).decode("utf-8")
    base_id = f"{bev_image_path.stem}_{camera_image_path.stem}".replace("-", "_")
    bev_canvas_id = f"pair_bev_{base_id}"
    camera_canvas_id = f"pair_camera_{base_id}"
    output_id = f"pair_output_{base_id}"
    status_id = f"pair_status_{base_id}"
    keypoints_json = json.dumps(
        [
            {
                "name": str(item["name"]),
                "xy": [float(item["xy"][0]), float(item["xy"][1])],
            }
            for item in keypoints
        ],
        ensure_ascii=False,
    )

    html = f"""
    <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
      <h3 style="margin: 0 0 8px 0;">{title}</h3>
      <p style="margin: 0 0 10px 0;">
        先在上方 BEV 選擇一個 keypoint，再在下方相機畫面點選對應位置。至少建立四組 pair 後即可估計 Homography。
      </p>
      <div id="{status_id}" style="margin: 0 0 8px 0; font-weight: 600;"></div>
      <canvas id="{bev_canvas_id}" style="border:1px solid #999; max-width:100%; display:block; margin-bottom:12px;"></canvas>
      <canvas id="{camera_canvas_id}" style="border:1px solid #999; max-width:100%; display:block;"></canvas>
      <p style="margin: 8px 0 4px 0;">pairs JSON：</p>
      <textarea id="{output_id}" rows="10" style="width:100%; font-family:monospace;"></textarea>
      <br />
      <button onclick="window.clear_{base_id}()" style="margin-top: 8px;">清除 pairs</button>
    </div>
    <script>
    (function() {{
      const bevImg = new Image();
      const camImg = new Image();
      const bevCanvas = document.getElementById("{bev_canvas_id}");
      const camCanvas = document.getElementById("{camera_canvas_id}");
      const bctx = bevCanvas.getContext("2d");
      const cctx = camCanvas.getContext("2d");
      const output = document.getElementById("{output_id}");
      const status = document.getElementById("{status_id}");
      const keypoints = {keypoints_json};
      const pairs = [];
      let selected = null;
      let imagesReady = 0;

      function eventToImage(canvas, img, event) {{
        const rect = canvas.getBoundingClientRect();
        const x = Math.round((event.clientX - rect.left) * img.width / rect.width);
        const y = Math.round((event.clientY - rect.top) * img.height / rect.height);
        return [x, y];
      }}

      function imageToCanvas(canvas, img, point) {{
        return [
          point[0] * canvas.width / img.width,
          point[1] * canvas.height / img.height
        ];
      }}

      function nearestKeypoint(point) {{
        let best = null;
        let bestDist = Infinity;
        for (const kp of keypoints) {{
          const dx = kp.xy[0] - point[0];
          const dy = kp.xy[1] - point[1];
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < bestDist) {{
            best = kp;
            bestDist = dist;
          }}
        }}
        return bestDist <= 35 ? best : null;
      }}

      function drawLabel(ctx, text, x, y, color) {{
        ctx.font = "13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        ctx.lineWidth = 4;
        ctx.strokeStyle = "rgba(255,255,255,0.95)";
        ctx.strokeText(text, x, y);
        ctx.fillStyle = color;
        ctx.fillText(text, x, y);
      }}

      function redrawBev() {{
        bctx.drawImage(bevImg, 0, 0, bevCanvas.width, bevCanvas.height);
        for (const kp of keypoints) {{
          const p = imageToCanvas(bevCanvas, bevImg, kp.xy);
          const alreadyPaired = pairs.some(pair => pair.keypoint_name === kp.name);
          const isSelected = selected && selected.name === kp.name;
          const color = isSelected ? "rgb(255,80,80)" : (alreadyPaired ? "rgb(46,204,113)" : "rgb(70,120,220)");
          bctx.beginPath();
          bctx.arc(p[0], p[1], isSelected ? 7 : 5, 0, Math.PI * 2);
          bctx.fillStyle = color;
          bctx.fill();
          drawLabel(bctx, kp.name, p[0] + 8, p[1] - 8, color);
        }}
      }}

      function redrawCamera() {{
        cctx.drawImage(camImg, 0, 0, camCanvas.width, camCanvas.height);
        for (const pair of pairs) {{
          const p = imageToCanvas(camCanvas, camImg, pair.camera_xy);
          cctx.beginPath();
          cctx.arc(p[0], p[1], 6, 0, Math.PI * 2);
          cctx.fillStyle = "rgb(255,80,80)";
          cctx.fill();
          drawLabel(cctx, pair.keypoint_name, p[0] + 8, p[1] - 8, "rgb(255,80,80)");
        }}
      }}

      function updateOutput() {{
        const data = {{
          pairs: pairs,
          camera_points: pairs.map(pair => pair.camera_xy),
          bev_points: pairs.map(pair => pair.bev_xy)
        }};
        output.value = JSON.stringify(data, null, 2);
        const pending = selected ? `selected: ${{selected.name}}` : "select a BEV keypoint";
        status.textContent = `${{pairs.length}} pair(s). ${{pending}}`;
      }}

      function redraw() {{
        redrawBev();
        redrawCamera();
        updateOutput();
      }}

      function setupCanvas(canvas, img) {{
        const scale = Math.min(1, {canvas_width} / img.width);
        canvas.width = Math.round(img.width * scale);
        canvas.height = Math.round(img.height * scale);
      }}

      function maybeInit() {{
        imagesReady += 1;
        if (imagesReady !== 2) return;
        setupCanvas(bevCanvas, bevImg);
        setupCanvas(camCanvas, camImg);
        redraw();
      }}

      bevCanvas.addEventListener("click", function(event) {{
        const point = eventToImage(bevCanvas, bevImg, event);
        selected = nearestKeypoint(point);
        redraw();
      }});

      camCanvas.addEventListener("click", function(event) {{
        if (!selected) {{
          status.textContent = "select a BEV keypoint first";
          return;
        }}
        const cameraPoint = eventToImage(camCanvas, camImg, event);
        const existingIndex = pairs.findIndex(pair => pair.keypoint_name === selected.name);
        const pair = {{
          keypoint_name: selected.name,
          bev_xy: selected.xy.map(v => Number(v.toFixed(2))),
          camera_xy: cameraPoint
        }};
        if (existingIndex >= 0) {{
          pairs[existingIndex] = pair;
        }} else {{
          pairs.push(pair);
        }}
        selected = null;
        redraw();
      }});

      window.clear_{base_id} = function() {{
        pairs.length = 0;
        selected = null;
        redraw();
      }};

      bevImg.onload = maybeInit;
      camImg.onload = maybeInit;
      bevImg.src = "data:image/png;base64,{bev_b64}";
      camImg.src = "data:image/png;base64,{camera_b64}";
    }})();
    </script>
    """
    display(HTML(html))


def show_bbox_to_bev_tool(
    image_path: str | Path,
    initial_bbox: Sequence[float],
    homography_matrix: Sequence[Sequence[float]],
    bev_width: int,
    bev_height: int,
    canvas_width: int = 1000,
    title: str = "BBOX-to-BEV 互動工具",
) -> None:
    """顯示可調整 bbox 並即時投影 bottom-center footpoint 的 Notebook 工具。"""
    image_path = Path(image_path)
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    base_id = image_path.stem.replace("-", "_")
    canvas_id = f"bbox_canvas_{base_id}"
    bev_id = f"bev_canvas_{base_id}"
    output_id = f"bbox_output_{base_id}"
    sliders_id = f"bbox_sliders_{base_id}"
    bbox_json = json.dumps([float(v) for v in initial_bbox])
    H_json = json.dumps([[float(v) for v in row] for row in homography_matrix])

    html = f"""
    <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
      <h3 style="margin-bottom: 8px;">{title}</h3>
      <div style="display:grid; grid-template-columns:minmax(0, 1fr) 360px; gap:16px; align-items:start;">
        <div>
          <canvas id="{canvas_id}" style="border:1px solid #999; max-width:100%;"></canvas>
          <div id="{sliders_id}" style="margin-top:10px; display:grid; gap:6px;"></div>
        </div>
        <div>
          <canvas id="{bev_id}" width="{bev_width}" height="{bev_height}" style="border:1px solid #999; width:100%;"></canvas>
          <textarea id="{output_id}" rows="7" style="margin-top:8px; width:100%; font-family:monospace;"></textarea>
        </div>
      </div>
    </div>
    <script>
    (function() {{
      const img = new Image();
      const canvas = document.getElementById("{canvas_id}");
      const ctx = canvas.getContext("2d");
      const bev = document.getElementById("{bev_id}");
      const bctx = bev.getContext("2d");
      const output = document.getElementById("{output_id}");
      const sliderHost = document.getElementById("{sliders_id}");
      const bbox = {bbox_json};
      const H = {H_json};
      const labels = ["x1", "y1", "x2", "y2"];
      const inputs = [];

      function project(x, y) {{
        const denom = H[2][0] * x + H[2][1] * y + H[2][2];
        return [
          (H[0][0] * x + H[0][1] * y + H[0][2]) / denom,
          (H[1][0] * x + H[1][1] * y + H[1][2]) / denom
        ];
      }}

      function addSlider(index, maxValue) {{
        const row = document.createElement("label");
        row.style.display = "grid";
        row.style.gridTemplateColumns = "32px 1fr 64px";
        row.style.gap = "8px";
        row.style.alignItems = "center";
        const name = document.createElement("span");
        name.textContent = labels[index];
        const input = document.createElement("input");
        input.type = "range";
        input.min = "0";
        input.max = String(maxValue);
        input.value = String(Math.round(bbox[index]));
        const value = document.createElement("input");
        value.type = "number";
        value.value = input.value;
        value.style.width = "64px";
        input.addEventListener("input", function() {{
          value.value = input.value;
          bbox[index] = Number(input.value);
          redraw();
        }});
        value.addEventListener("input", function() {{
          input.value = value.value;
          bbox[index] = Number(value.value);
          redraw();
        }});
        row.appendChild(name);
        row.appendChild(input);
        row.appendChild(value);
        sliderHost.appendChild(row);
        inputs.push(input);
      }}

      function drawBev(point) {{
        bctx.fillStyle = "rgb(248, 248, 248)";
        bctx.fillRect(0, 0, bev.width, bev.height);
        bctx.strokeStyle = "rgb(40, 40, 40)";
        bctx.lineWidth = 2;
        bctx.strokeRect(20, 20, bev.width - 40, bev.height - 40);
        bctx.beginPath();
        bctx.moveTo(bev.width / 2, 20);
        bctx.lineTo(bev.width / 2, bev.height - 20);
        bctx.stroke();
        bctx.fillStyle = "rgb(255, 80, 80)";
        bctx.beginPath();
        bctx.arc(point[0], point[1], 7, 0, Math.PI * 2);
        bctx.fill();
      }}

      function redraw() {{
        const scale = canvas.width / img.width;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        const x1 = bbox[0] * scale;
        const y1 = bbox[1] * scale;
        const x2 = bbox[2] * scale;
        const y2 = bbox[3] * scale;
        const foot = [(bbox[0] + bbox[2]) / 2, bbox[3]];
        const footScaled = [foot[0] * scale, foot[1] * scale];
        const bevPoint = project(foot[0], foot[1]);

        ctx.strokeStyle = "rgb(255, 80, 80)";
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = "rgb(40, 120, 255)";
        ctx.beginPath();
        ctx.arc(footScaled[0], footScaled[1], 6, 0, Math.PI * 2);
        ctx.fill();
        drawBev(bevPoint);
        output.value = JSON.stringify({{
          bbox_xyxy: bbox.map(v => Math.round(v)),
          footpoint_xy: foot.map(v => Number(v.toFixed(2))),
          bev_xy: bevPoint.map(v => Number(v.toFixed(2)))
        }}, null, 2);
      }}

      img.onload = function() {{
        const scale = {canvas_width} / img.width;
        canvas.width = {canvas_width};
        canvas.height = Math.round(img.height * scale);
        addSlider(0, img.width);
        addSlider(1, img.height);
        addSlider(2, img.width);
        addSlider(3, img.height);
        redraw();
      }};
      img.src = "data:image/png;base64,{b64}";
    }})();
    </script>
    """
    display(HTML(html))


def show_homography_projection_tool(
    image_path: str | Path,
    bev_image_path: str | Path,
    homography_matrix: Sequence[Sequence[float]],
    canvas_width: int = 1000,
    title: str = "Homography Projection Tool",
) -> None:
    """Click any point in the camera image and project it to the BEV court."""
    image_path = Path(image_path)
    bev_image_path = Path(bev_image_path)
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    bev_b64 = base64.b64encode(bev_image_path.read_bytes()).decode("utf-8")
    base_id = f"{image_path.stem}_{bev_image_path.stem}".replace("-", "_")
    camera_canvas_id = f"proj_camera_{base_id}"
    bev_canvas_id = f"proj_bev_{base_id}"
    output_id = f"proj_output_{base_id}"
    status_id = f"proj_status_{base_id}"
    H_json = json.dumps([[float(v) for v in row] for row in homography_matrix])

    html = f"""
    <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
      <h3 style="margin: 0 0 8px 0;">{title}</h3>
      <p style="margin: 0 0 10px 0;">
        點選左側相機畫面任一位置；工具會立即用目前的 Homography 矩陣投影到右側 BEV。
      </p>
      <div id="{status_id}" style="margin: 0 0 8px 0; font-weight: 600;"></div>
      <div style="display:grid; grid-template-columns:minmax(0, 1fr) minmax(0, 1fr); gap:16px; align-items:start;">
        <canvas id="{camera_canvas_id}" style="border:1px solid #999; max-width:100%; display:block;"></canvas>
        <canvas id="{bev_canvas_id}" style="border:1px solid #999; max-width:100%; display:block;"></canvas>
      </div>
      <p style="margin: 8px 0 4px 0;">projection JSON：</p>
      <textarea id="{output_id}" rows="10" style="width:100%; font-family:monospace;"></textarea>
      <br />
      <button onclick="window.clear_{base_id}()" style="margin-top: 8px;">清除點位</button>
    </div>
    <script>
    (function() {{
      const cameraImg = new Image();
      const bevImg = new Image();
      const cameraCanvas = document.getElementById("{camera_canvas_id}");
      const bevCanvas = document.getElementById("{bev_canvas_id}");
      const cctx = cameraCanvas.getContext("2d");
      const bctx = bevCanvas.getContext("2d");
      const output = document.getElementById("{output_id}");
      const status = document.getElementById("{status_id}");
      const H = {H_json};
      const clicks = [];
      let imagesReady = 0;

      function eventToImage(canvas, img, event) {{
        const rect = canvas.getBoundingClientRect();
        const x = Math.round((event.clientX - rect.left) * img.width / rect.width);
        const y = Math.round((event.clientY - rect.top) * img.height / rect.height);
        return [x, y];
      }}

      function imageToCanvas(canvas, img, point) {{
        return [
          point[0] * canvas.width / img.width,
          point[1] * canvas.height / img.height
        ];
      }}

      function project(point) {{
        const x = point[0];
        const y = point[1];
        const denom = H[2][0] * x + H[2][1] * y + H[2][2];
        return [
          (H[0][0] * x + H[0][1] * y + H[0][2]) / denom,
          (H[1][0] * x + H[1][1] * y + H[1][2]) / denom
        ];
      }}

      function drawMarker(ctx, x, y, fillColor, label) {{
        ctx.beginPath();
        ctx.arc(x, y, 7, 0, Math.PI * 2);
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.font = "13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        ctx.lineWidth = 4;
        ctx.strokeStyle = "rgba(255,255,255,0.95)";
        ctx.strokeText(label, x + 8, y - 8);
        ctx.fillStyle = fillColor;
        ctx.fillText(label, x + 8, y - 8);
      }}

      function redraw() {{
        cctx.drawImage(cameraImg, 0, 0, cameraCanvas.width, cameraCanvas.height);
        bctx.drawImage(bevImg, 0, 0, bevCanvas.width, bevCanvas.height);
        for (let i = 0; i < clicks.length; i++) {{
          const item = clicks[i];
          const cp = imageToCanvas(cameraCanvas, cameraImg, item.camera_xy);
          const bp = imageToCanvas(bevCanvas, bevImg, item.bev_xy);
          drawMarker(cctx, cp[0], cp[1], "rgb(255,80,80)", String(i));
          drawMarker(bctx, bp[0], bp[1], "rgb(46,204,113)", String(i));
        }}
        output.value = JSON.stringify({{
          projections: clicks,
          camera_points: clicks.map(item => item.camera_xy),
          bev_points: clicks.map(item => item.bev_xy)
        }}, null, 2);
        status.textContent = `${{clicks.length}} projected point(s)` + (clicks.length > 0 ? "" : ". click the camera image");
      }}

      function setupCanvas(canvas, img) {{
        const scale = Math.min(1, {canvas_width} / img.width);
        canvas.width = Math.round(img.width * scale);
        canvas.height = Math.round(img.height * scale);
      }}

      function maybeInit() {{
        imagesReady += 1;
        if (imagesReady !== 2) return;
        setupCanvas(cameraCanvas, cameraImg);
        setupCanvas(bevCanvas, bevImg);
        redraw();
      }}

      cameraCanvas.addEventListener("click", function(event) {{
        const cameraPoint = eventToImage(cameraCanvas, cameraImg, event);
        const bevPoint = project(cameraPoint);
        clicks.push({{
          camera_xy: cameraPoint.map(v => Number(v.toFixed(2))),
          bev_xy: bevPoint.map(v => Number(v.toFixed(2)))
        }});
        redraw();
      }});

      window.clear_{base_id} = function() {{
        clicks.length = 0;
        redraw();
      }};

      cameraImg.onload = maybeInit;
      bevImg.onload = maybeInit;
      cameraImg.src = "data:image/png;base64,{image_b64}";
      bevImg.src = "data:image/png;base64,{bev_b64}";
    }})();
    </script>
    """
    display(HTML(html))
