from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Sequence

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
        點選影像中的參考點；下方會即時列出座標。請依照課堂指定順序點選。
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
        const x = Math.round(sx * img.width / canvas.width);
        const y = Math.round(sy * img.height / canvas.height);
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
