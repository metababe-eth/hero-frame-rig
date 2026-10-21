"""Keep the original upload and make a separate, orientation-correct working PNG."""
import hashlib
from pathlib import Path
import shutil

from PIL import Image, ImageOps


def prepare_upload(path, directory, max_side=1600):
    source = Path(path)
    directory = Path(directory)
    with Image.open(source) as raw:
        stored_size = list(raw.size)
        orientation = raw.getexif().get(274, 1)
        source_format = raw.format
        working = ImageOps.exif_transpose(raw).convert('RGB')
        display_size = list(working.size)
        scale = min(1.0, max_side / max(working.size))
        if scale < 1:
            working = working.resize((max(1, round(working.width * scale)),
                                      max(1, round(working.height * scale))), Image.Resampling.LANCZOS)
        still = directory / 'still.png'
        working.save(still)
    suffix = source.suffix.lower()
    if not suffix or len(suffix) > 10 or not suffix[1:].isalnum():
        suffix = '.bin'
    original = directory / ('original' + suffix)
    shutil.copyfile(source, original)
    return str(still), str(original), {
        'filename': source.name,
        'archive_path': 'source/' + original.name,
        'sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
        'bytes': original.stat().st_size,
        'format': source_format,
        'stored_size': stored_size,
        'display_size': display_size,
        'exif_orientation_applied': orientation,
        'working_size': list(working.size),
        'processing': 'EXIF orientation applied, RGB conversion, Lanczos resize only if longest side exceeds limit.',
        'working_max_side': max_side,
    }
