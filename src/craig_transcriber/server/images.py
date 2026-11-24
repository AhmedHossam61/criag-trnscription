"""
Smaller copies of big images for the site: portraits shown in small circles,
the emblem, comic covers. Each copy is made once, as WebP, and kept in
data/.cache/images under a name that changes when the original does.
Without Pillow, or for an image that is already small, the original is served.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

WIDTHS = (96, 192, 384, 800)   # the sizes the pages ask for; anything else snaps to the next one up
_lock = threading.Lock()


def snap(width: int | None) -> int | None:
    if not width:
        return None
    return next((w for w in WIDTHS if w >= width), WIDTHS[-1])


def sized(path: Path, width: int | None, cache_dir: Path) -> Path:
    """`path` at most `width` pixels wide, or `path` itself when that's no smaller."""
    width = snap(width)
    if width is None:
        return path
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return path
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{width}".encode()).hexdigest()[:20]
    out = cache_dir / f"{key}.webp"
    if out.exists():
        return out
    with _lock:   # two requests for the same new image make it once
        if out.exists():
            return out
        try:
            with Image.open(path) as im:
                if im.width <= width and st.st_size < 200_000:
                    return path
                im = ImageOps.exif_transpose(im)
                if im.width > width:
                    im = im.resize((width, max(1, round(im.height * width / im.width))), Image.LANCZOS)
                im = im.convert("RGBA" if "A" in im.getbands() or "transparency" in im.info else "RGB")
                cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = out.with_name(f"{out.stem}.{os.getpid()}.tmp")
                im.save(tmp, "WEBP", quality=82, method=4)
                tmp.replace(out)
        except OSError:      # not an image Pillow can read: serve it as it is
            return path
    return out
