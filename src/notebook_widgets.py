from __future__ import annotations

import base64
from pathlib import Path

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
