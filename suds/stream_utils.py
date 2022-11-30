import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any

from rich.console import Console

try:
    import gcsfs
except:
    pass

try:
    import s3fs
except:
    pass

from PIL import Image
from smart_open import open
import pyarrow.parquet as pq

NUM_RETRIES = 3

CONSOLE = Console(width=120)


def table_from_stream(path: str) -> Any:
    for i in range(NUM_RETRIES):
        try:
            return pq.read_table(path, filesystem=get_filesystem(path))
        except Exception as e:
            CONSOLE.log('Download failed for {} (attempt {})'.format(path, i + 1))
            if i == NUM_RETRIES - 1:
                raise e
            traceback.print_exc()
            time.sleep(10)


def buffer_from_stream(path: str) -> BytesIO:
    for i in range(NUM_RETRIES):
        try:
            buffer = BytesIO()
            with open(path, 'rb') as f:
                buffer.write(f.read())
            buffer.seek(0)
            return buffer
        except Exception as e:
            CONSOLE.log('Download failed for {} (attempt {})'.format(path, i + 1))
            if i == NUM_RETRIES - 1:
                raise e
            traceback.print_exc()
            time.sleep(10)


def image_from_stream(path: str) -> Image:
    return Image.open(buffer_from_stream(path))


def image_to_stream(img: Image, path: str) -> None:
    for i in range(NUM_RETRIES):
        try:
            extension = Path(path).suffix
            if extension == '.png':
                format = 'PNG'
            elif extension == '.jpg':
                format = 'JPEG'
            else:
                raise Exception(path)

            buffer = BytesIO()
            img.save(buffer, format=format)
            with open(path, 'wb') as f:
                f.write(buffer.getbuffer())
            return
        except Exception as e:
            CONSOLE.log('Download failed for {} (attempt {})'.format(path, i + 1))
            if i == NUM_RETRIES - 1:
                raise e
            traceback.print_exc()
            time.sleep(10)


def get_filesystem(path: str) -> Any:
    if path.startswith('s3://'):
        return s3fs.S3FileSystem()
    elif path.startswith('gs://'):
        return gcsfs.GCSFileSystem()
    else:
        return None
