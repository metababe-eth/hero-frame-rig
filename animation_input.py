"""Prepare a separate, fully retained frame for the bounded portrait motion study."""
import hashlib
from pathlib import Path
import shutil

from PIL import Image, ImageOps

WIDTH, HEIGHT = 704, 1280

def prepare_motion_input(source, directory):
    source, directory = Path(source), Path(directory)
    with Image.open(source) as raw:
        if raw.width * raw.height > 32_000_000 or getattr(raw, 'n_frames', 1) != 1:
            raise ValueError('Choose one still image with at most 32 million pixels.')
        original_size, original_mode = list(raw.size), raw.mode
        oriented = ImageOps.exif_transpose(raw)
        # Flatten transparency explicitly, rather than silently discarding alpha.
        rgba = oriented.convert('RGBA')
        matte = Image.new('RGBA', rgba.size, (16, 16, 16, 255))
        image = Image.alpha_composite(matte, rgba).convert('RGB')
    scale = min(WIDTH / image.width, HEIGHT / image.height)
    width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    x, y = (WIDTH-width)//2, (HEIGHT-height)//2
    canvas = Image.new('RGB', (WIDTH, HEIGHT), (16, 16, 16))
    canvas.paste(resized, (x, y))
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'animation_input.png'
    canvas.save(target)
    suffix = source.suffix.lower()
    if not suffix or len(suffix) > 10 or not suffix[1:].isalnum():
        suffix = '.bin'
    retained = directory / ('reference_original' + suffix)
    shutil.copyfile(source, retained)
    return str(target), {
        'source_file': retained.name, 'source_sha256': hashlib.sha256(retained.read_bytes()).hexdigest(),
        'source_size': original_size, 'source_mode': original_mode,
        'display_size': list(image.size), 'canvas_size': [WIDTH, HEIGHT],
        'fitted_size': [width, height], 'offset_xy': [x, y],
        'actual_scale_xy': [width/image.width, height/image.height],
        'padding_rgb': [16, 16, 16], 'crop': False,
        'processing': 'EXIF orientation; explicit dark alpha matte; fit whole image with Lanczos; pad without cropping.',
        'input_file': target.name, 'input_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    }
