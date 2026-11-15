"""Image and JSON input; bounded retries for transient image-read failures."""
import json
import time
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    for attempt in range(3):
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.1)


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)
