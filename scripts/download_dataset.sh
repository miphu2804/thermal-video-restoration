#!/usr/bin/env bash
# Download the FLIR ADAS v2 thermal 14-bit TIFFs (+ COCO labels) into data/.
#
# Source: Kaggle mirror of FLIR ADAS v2 (the official FLIR page is behind a form).
# Only the thermal part (~4.7 GB of the 11.9 GB archive) is fetched, using HTTP
# range requests against the remote zip, so the full archive is never stored.
# The wanted files sit in a few contiguous blocks of the zip, so they are read in
# ~64 MB ranges instead of one request per file: few requests, low CPU.
# Re-running resumes: files already present with the right size are skipped.
#
# Requires: uv, and Kaggle credentials in ~/.kaggle/kaggle.json.
# Usage: scripts/download_dataset.sh [workers]   (default 3, leaves cores free)
# The FLIR ADAS license forbids redistribution: keep data/ out of git.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DEST="$ROOT/data"
export WORKERS="${1:-3}"
export SLUG="samdazel/teledyne-flir-adas-thermal-dataset-v2"

[ -f "$HOME/.kaggle/kaggle.json" ] || { echo "Missing ~/.kaggle/kaggle.json" >&2; exit 1; }
mkdir -p "$DEST"

# `nice` keeps the download from competing with interactive work.
nice -n 10 uvx --quiet --with remotezip --with tqdm --with requests python - <<'PY'
import json
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import requests
from remotezip import RemoteZip
from tqdm import tqdm

dest, slug, workers = os.environ["DEST"], os.environ["SLUG"], int(os.environ["WORKERS"])
BATCH_BYTES = 64 * 2**20
MAX_GAP = 2**20
LOCAL_HEADER = 30  # fixed part of a zip local file header

cred = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))

# Kaggle answers with a redirect to a signed storage URL that supports ranges.
resp = requests.get(
    f"https://www.kaggle.com/api/v1/datasets/download/{slug}",
    auth=(cred["username"], cred["key"]),
    allow_redirects=False,
    timeout=30,
)
url = resp.headers.get("Location")
if resp.status_code != 302 or not url:
    raise SystemExit(f"Cannot resolve download URL (HTTP {resp.status_code})")


def wanted(name: str) -> bool:
    if name.endswith("/"):
        return False
    if "/" not in name:  # top-level README / checksum / mapping files
        return True
    if "thermal" not in name:
        return False
    return ("/analyticsData/" in name and name.endswith(".tiff")) or name.endswith(
        (".json", ".txt", ".tsv")
    )


with RemoteZip(url) as z:
    infos = sorted(
        (i for i in z.infolist() if wanted(i.filename)), key=lambda i: i.header_offset
    )

todo = [
    i
    for i in infos
    if not (
        os.path.exists(os.path.join(dest, i.filename))
        and os.path.getsize(os.path.join(dest, i.filename)) == i.file_size
    )
]
print(f"{len(infos)} files wanted, {len(infos) - len(todo)} already present")


def end_of(info) -> int:
    # Local header extra field may differ from the central one; pad generously.
    return info.header_offset + LOCAL_HEADER + len(info.filename) + 4096 + info.compress_size


# Group nearby files into one ranged read each.
batches, cur = [], []
for info in todo:
    if cur and (
        info.header_offset - end_of(cur[-1]) > MAX_GAP
        or end_of(info) - cur[0].header_offset > BATCH_BYTES
    ):
        batches.append(cur)
        cur = []
    cur.append(info)
if cur:
    batches.append(cur)


def extract(info, buf: bytes, base: int) -> None:
    pos = info.header_offset - base
    name_len, extra_len = struct.unpack("<HH", buf[pos + 26 : pos + 30])
    start = pos + LOCAL_HEADER + name_len + extra_len
    data = buf[start : start + info.compress_size]
    if info.compress_type == 8:
        data = zlib.decompress(data, -15)
    elif info.compress_type != 0:
        raise ValueError(f"unsupported compression {info.compress_type}")
    if len(data) != info.file_size:
        raise ValueError(f"size mismatch for {info.filename}")
    out = os.path.join(dest, info.filename)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".part", "wb") as f:
        f.write(data)
    os.replace(out + ".part", out)


failed = []


def fetch(batch) -> None:
    base = batch[0].header_offset
    last = end_of(batch[-1]) - 1
    for _ in range(4):
        try:
            r = requests.get(url, headers={"Range": f"bytes={base}-{last}"}, timeout=120)
            r.raise_for_status()
            for info in batch:
                extract(info, r.content, base)
            break
        except Exception:
            continue
    else:
        failed.extend(i.filename for i in batch)
    bar.update(sum(i.compress_size for i in batch))


with tqdm(
    total=sum(i.compress_size for i in todo), unit="B", unit_scale=True, desc="FLIR ADAS"
) as bar, ThreadPoolExecutor(workers) as ex:
    list(ex.map(fetch, batches))

if failed:
    raise SystemExit(f"{len(failed)} files failed, re-run to resume: {failed[:3]}")
print("Done:", dest)
PY
