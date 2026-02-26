from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import torch
from torch.hub import HASH_REGEX

from rfstudio.ui import console


def _download_impl(u, f, sha256):
    while True:
        buffer = u.read(8192)
        if len(buffer) == 0:
            break
        f.write(buffer)
        if sha256 is not None:
            sha256.update(buffer)

def download_url_to_file(
    url: str,
    *,
    filename: Path,
    hash_prefix: Optional[str] = None,
    verbose: bool = True,
) -> None:
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / 'temp'
        with src.open("w+b") as fsrc:
            sha256 = None
            if hash_prefix is not None:
                sha256 = hashlib.sha256()
            if verbose:
                console.print(f'Download [u]{url}[/u] to [bright_yellow][u]{filename}[/u][/bright_yellow]')
                if file_size is not None:
                    with console.progress('Downloading', wrap_file=True, transient=True) as ptrack:
                        u = ptrack(u, total=file_size)
                        _download_impl(u, fsrc, sha256)
                else:
                    with console.status('Downloading'):
                        _download_impl(u, fsrc, sha256)
            else:
                _download_impl(u, fsrc, sha256)

        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError(f'invalid hash value (expected "{hash_prefix}", got "{digest}")')
        shutil.move(src, filename)

def download_model_weights(
    url: str,
    *,
    cache_dir: Optional[Path] = None,
    check_hash: bool = False,
    filename: Optional[str] = None,
    verbose: bool = True,
) -> Path:
    cache_dir = (Path(torch.hub.get_dir()) / 'checkpoints') if cache_dir is None else cache_dir
    cache_dir.mkdir(exist_ok=True, parents=True)

    parts = urlparse(url)
    if filename is None:
        filename = os.path.basename(parts.path)
    cached_file = cache_dir / filename
    if cached_file.exists():
        return cached_file
    hash_prefix = None
    if check_hash:
        r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
        hash_prefix = r.group(1) if r else None
    download_url_to_file(url, filename=cached_file, hash_prefix=hash_prefix, verbose=verbose)
    return cached_file
