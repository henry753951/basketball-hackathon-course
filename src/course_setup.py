from __future__ import annotations

import os
import tempfile
import subprocess
import sys
from pathlib import Path
import shutil


DEFAULT_COURSE_ROOT = Path("/content/drive/MyDrive/basketball_hackathon/course")
DEFAULT_DRIVE_MOUNT_POINT = "/content/drive"
DEFAULT_REPO_SYNC_EXCLUDES = (
    ".git",
    ".github",
    ".venv",
    "__pycache__",
    ".ipynb_checkpoints",
)
DEFAULT_PRESERVE_PATHS = ("assets/results",)


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


def mount_drive_if_colab(mount_point: str = DEFAULT_DRIVE_MOUNT_POINT) -> bool:
    """在 Colab 中掛載 Google Drive；本機執行時直接略過。"""
    if not running_in_colab():
        return False

    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount(mount_point)
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


def clone_repo_snapshot(repo_url: str, branch: str, clone_dir: str | Path) -> Path:
    clone_dir = Path(clone_dir).expanduser().resolve()
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "-b",
        branch,
        repo_url,
        str(clone_dir),
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return clone_dir


def rsync_tree(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    exclude: tuple[str, ...] = DEFAULT_REPO_SYNC_EXCLUDES,
) -> Path:
    source_dir = Path(source_dir).expanduser().resolve()
    target_dir = Path(target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("rsync"):
        cmd = ["rsync", "-av"]
        for pattern in exclude:
            cmd.append(f"--exclude={pattern}")
        cmd.extend([f"{source_dir}/", f"{target_dir}/"])

        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return target_dir

    for item in source_dir.iterdir():
        if item.name in exclude:
            continue

        dst = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*exclude))
        else:
            shutil.copy2(item, dst)

    return target_dir


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def replace_tree_from_snapshot(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    preserve_paths: tuple[str, ...] = DEFAULT_PRESERVE_PATHS,
) -> Path:
    """用指定 snapshot 覆蓋課程資料夾，並回填保留路徑。"""
    source_dir = Path(source_dir).expanduser().resolve()
    target_dir = Path(target_dir).expanduser().resolve()
    work_parent = target_dir.parent

    sync_root = Path(
        tempfile.mkdtemp(prefix=".course-sync-", dir=os.fspath(work_parent))
    ).resolve()
    preserve_root = sync_root / "preserved"
    success = False

    try:
        if target_dir.exists():
            for rel_path in preserve_paths:
                src = target_dir / rel_path
                if not src.exists():
                    continue

                dst = preserve_root / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(os.fspath(src), os.fspath(dst))

            shutil.rmtree(target_dir)

        rsync_tree(source_dir, target_dir)

        for rel_path in preserve_paths:
            src = preserve_root / rel_path
            if not src.exists():
                continue
            _copy_path(src, target_dir / rel_path)

        success = True
        return target_dir
    except Exception:
        print("同步失敗，保留暫存資料夾供檢查:", sync_root)
        raise
    finally:
        if success:
            shutil.rmtree(sync_root, ignore_errors=True)


def refresh_course_checkout(
    repo_url: str,
    branch: str,
    target_dir: str | Path,
    *,
    preserve_paths: tuple[str, ...] = DEFAULT_PRESERVE_PATHS,
) -> Path:
    """重新抓最新課程 repo，並保留學生產出資料夾。"""
    target_dir = Path(target_dir).expanduser().resolve()
    work_parent = target_dir.parent
    sync_root = Path(
        tempfile.mkdtemp(prefix=".course-repo-", dir=os.fspath(work_parent))
    ).resolve()
    clone_dir = sync_root / "repo"

    try:
        clone_repo_snapshot(repo_url, branch, clone_dir)
        return replace_tree_from_snapshot(
            clone_dir,
            target_dir,
            preserve_paths=preserve_paths,
        )
    except Exception:
        print("repo 更新失敗，保留暫存資料夾供檢查:", sync_root)
        raise
    finally:
        shutil.rmtree(sync_root, ignore_errors=True)


def bootstrap_course_notebook(
    preferred: str | Path | None = None,
    *,
    mount_drive: bool = True,
    install_requirements: bool = True,
    show_summary: bool = True,
) -> Path:
    """為課程 notebook 準備一致的 Colab / 本機執行環境。"""
    if mount_drive:
        mount_drive_if_colab()

    course_root = find_course_root(preferred)
    add_course_to_path(course_root)

    if install_requirements:
        install_requirements_if_colab(course_root)
    if show_summary:
        print_environment_summary(course_root)

    return course_root
