from __future__ import annotations

import base64
import os
import shutil
import subprocess
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Callable, cast

import cv2
from IPython.display import HTML, display
import requests

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".3gp", ".avi", ".mkv"}
ARCHIVE_EXTS = {".zip"}


def find_course_root() -> Path:
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path("/content/drive/MyDrive/basketball_hackathon/course"),
        Path("/content/basketball_hackathon_course"),
    ]
    for c in candidates:
        if (c / "src").exists() and (c / "assets").exists():
            return c.resolve()
    raise FileNotFoundError(
        "找不到 course root。請確認你已經執行 init_colab，或目前工作目錄在 course 裡。"
    )


def list_videos(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    paths = []
    for ext in VIDEO_EXTS:
        paths.extend(folder.glob(f"*{ext}"))
        paths.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(paths))


def download_file(
    url: str, output_path: str | Path, chunk_size: int = 1024 * 1024
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with output_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return output_path


def unpack_archives(raw_dir: str | Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    extracted = []
    for p in raw_dir.iterdir() if raw_dir.exists() else []:
        if p.suffix.lower() == ".zip":
            target = raw_dir / p.stem
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(target)
            extracted.append(target)
    return extracted


def convert_video(
    input_path: str | Path, output_path: str | Path, fps: int = 30, max_side: int = 1280
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg_binary()
    if ffmpeg is None:
        return convert_video_with_opencv(
            input_path=input_path,
            output_path=output_path,
            fps=fps,
            max_side=max_side,
        )
    vf = f"scale='if(gt(iw,ih),{max_side},-2)':'if(gt(ih,iw),{max_side},-2)',fps={fps},format=yuv420p"
    cmd = [
        str(ffmpeg),
        "-y",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(output_path),
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_path


def convert_video_with_opencv(
    input_path: str | Path, output_path: str | Path, fps: int = 30, max_side: int = 1280
) -> Path:
    """Fallback video conversion when ffmpeg is unavailable on the machine."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(input_path)

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"無法讀取影片內容：{input_path}")

    src_h, src_w = first_frame.shape[:2]
    scale = min(1.0, max_side / max(src_w, src_h))
    out_w = max(2, int(round(src_w * scale)))
    out_h = max(2, int(round(src_h * scale)))
    if out_w % 2 == 1:
        out_w += 1
    if out_h % 2 == 1:
        out_h += 1

    writer, writer_codec = open_mp4_video_writer(output_path, fps=float(fps), frame_size=(out_w, out_h))

    print(
        "ffmpeg not found; using OpenCV fallback conversion",
        f"({input_path.name} -> {output_path.name}, codec={writer_codec})",
    )

    frame = first_frame
    while True:
        resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(resized)
        ok, frame = cap.read()
        if not ok:
            break

    writer.release()
    cap.release()
    return output_path


def ensure_notebook_playable_mp4(
    video_path: str | Path,
    *,
    overwrite: bool = True,
    crf: int = 23,
    preset: str = "veryfast",
) -> Path:
    """Re-encode a video to a browser-friendly H.264 MP4 for notebook playback."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    ffmpeg = find_ffmpeg_binary()
    if ffmpeg is None:
        return video_path

    output_path = video_path
    if overwrite:
        output_path = video_path.with_name(video_path.stem + ".notebook.mp4")

    cmd = [
        str(ffmpeg),
        "-y",
        "-i",
        str(video_path),
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-preset",
        preset,
        "-crf",
        str(crf),
        str(output_path),
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if overwrite:
        output_path.replace(video_path)
        return video_path
    return output_path


def display_video_in_notebook(
    video_path: str | Path,
    *,
    width: int = 960,
    controls: bool = True,
    muted: bool = True,
    loop: bool = False,
) -> None:
    """Embed a local MP4 directly into notebook output."""
    video_path = Path(video_path)
    mime_type = "video/mp4"
    b64 = base64.b64encode(video_path.read_bytes()).decode("utf-8")
    attrs = []
    if controls:
        attrs.append("controls")
    if muted:
        attrs.append("muted")
    if loop:
        attrs.append("loop")
    attr_text = " ".join(attrs)
    html = f"""
    <video {attr_text} width="{width}" style="max-width:100%; height:auto;">
      <source src="data:{mime_type};base64,{b64}" type="{mime_type}">
      無法預覽影片，請下載後播放。
    </video>
    """
    display(HTML(html))


def video_fourcc(codec: str = "avc1") -> int:
    if len(codec) != 4:
        raise ValueError(f"codec must be exactly 4 characters, got {codec!r}")

    fourcc = cast(
        Callable[[str, str, str, str], int] | None,
        getattr(cv2, "VideoWriter_fourcc", None),
    )
    if callable(fourcc):
        return fourcc(codec[0], codec[1], codec[2], codec[3]) # pyright: ignore[reportReturnType]

    legacy = cast(
        Callable[[str, str, str, str], int] | None,
        getattr(getattr(cv2, "VideoWriter", None), "fourcc", None),
    )
    if callable(legacy):
        return legacy(codec[0], codec[1], codec[2], codec[3])
    raise RuntimeError("OpenCV does not expose a VideoWriter fourcc helper.")


def open_mp4_video_writer(
    output_path: str | Path,
    *,
    fps: float,
    frame_size: tuple[int, int],
    preferred_codecs: tuple[str, ...] = ("avc1", "mp4v"),
) -> tuple[cv2.VideoWriter, str]:
    output_path = Path(output_path)
    for codec in preferred_codecs:
        writer = cv2.VideoWriter(str(output_path), video_fourcc(codec), fps, frame_size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(
        f"無法建立輸出影片：{output_path}；已嘗試 codecs={preferred_codecs!r}"
    )


@lru_cache(maxsize=1)
def find_ffmpeg_binary() -> Path | None:
    candidates: list[str | Path | None] = [
        os.environ.get("FFMPEG_BINARY"),
        shutil.which("ffmpeg"),
        Path.home() / "scoop" / "shims" / "ffmpeg.exe",
        Path.home() / "scoop" / "apps" / "ffmpeg-shared" / "current" / "bin" / "ffmpeg.exe",
    ]
    candidates.extend(sorted((Path.home() / "scoop" / "apps" / "ffmpeg-shared").glob("*/bin/ffmpeg.exe"), reverse=True))
    candidates.append(
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Softdeluxe"
        / "Free Download Manager"
        / "ffmpeg.exe"
    )
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists() and ffmpeg_supports_h264(path):
            return path
    return None


def ffmpeg_supports_h264(ffmpeg_path: str | Path) -> bool:
    try:
        result = subprocess.run(
            [str(ffmpeg_path), "-hide_banner", "-loglevel", "error", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return "libx264" in result.stdout


def convert_all_raw_videos(
    course_root: str | Path, fps: int = 30, max_side: int = 1280
) -> list[Path]:
    course_root = Path(course_root)
    raw_dir = course_root / "assets" / "raw"
    reference_dir = (raw_dir / "reference_videos").resolve()
    converted_dir = course_root / "assets" / "converted"
    unpack_archives(raw_dir)
    videos = []
    for ext in VIDEO_EXTS:
        videos.extend(raw_dir.rglob(f"*{ext}"))
        videos.extend(raw_dir.rglob(f"*{ext.upper()}"))
    videos = [
        path
        for path in videos
        if reference_dir not in path.resolve().parents
    ]
    outputs = []
    for i, src in enumerate(sorted(set(videos)), start=1):
        out = converted_dir / f"video_{i:03d}.mp4"
        outputs.append(convert_video(src, out, fps=fps, max_side=max_side))
    return outputs


def extract_frame(video_path: str | Path, frame_index: int = 0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"無法讀取 frame {frame_index}: {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def pick_first_converted_video(course_root: str | Path) -> Path:
    course_root = Path(course_root)
    videos = list_videos(course_root / "assets" / "converted")
    if videos:
        return videos[0]
    raise FileNotFoundError(
        "找不到 converted 影片。請先在 Day 4-01 上傳影片並完成轉檔。"
    )
