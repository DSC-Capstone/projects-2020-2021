from pathlib import Path

def ensure_path_exists(path, is_dir=False):
    """
    Accepts a path string or pathutils Path object, then creates all directories
    contained in that path if they don't already exist.
    """

    p = Path(path)

    if is_dir:
        Path(path).mkdir(parents=True, exist_ok=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)