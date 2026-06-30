from __future__ import annotations

import subprocess
import sys
from pathlib import Path


DEFAULT_COURSE_ROOT = Path("/content/drive/MyDrive/basketball_hackathon/course")


def find_course_root(preferred: str | Path | None = None) -> Path:
    """從 Colab Drive 路徑或本機 repo 尋找課程根目錄。"""
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(Path(preferred))
    candidates.extend(
        [
            DEFAULT_COURSE_ROOT,
            Path.cwd(),
            *Path.cwd().parents,
        ]
    )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "src").exists() and (candidate / "assets").exists():
            return candidate

    raise FileNotFoundError(
        "找不到課程根目錄。請先執行 init_colab.ipynb，或確認目前目錄位於課程 repo 內。"
    )


def add_course_to_path(course_root: str | Path) -> Path:
    course_root = Path(course_root).expanduser().resolve()
    if str(course_root) not in sys.path:
        sys.path.insert(0, str(course_root))
    return course_root


def running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore[import-not-found]  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def install_requirements_if_colab(course_root: str | Path) -> None:
    """Colab runtime 重新啟動後，自動安裝 Notebook 需要的套件。"""
    if not running_in_colab():
        return

    requirements = Path(course_root) / "requirements.txt"
    if not requirements.exists():
        print("找不到 requirements.txt，略過套件安裝。")
        return

    cmd = [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)]
    print("正在確認 Colab 執行環境套件...")
    subprocess.run(cmd, check=True)


def print_environment_summary(course_root: str | Path) -> None:
    course_root = Path(course_root)
    print("課程根目錄:", course_root)
    print("素材資料夾:", course_root / "assets")
    print("工具模組:", course_root / "src")
