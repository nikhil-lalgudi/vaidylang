"""Fetch only pinned public DeepSeek safetensors headers for placement accounting."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint import Checkpoint


REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
SOURCE = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/{REVISION}"
HEADER_LIMIT = 16 * 1024 * 1024


def read_range(url, start, stop):
    """Read exactly the advertised range, refusing a full-shard HTTP response."""
    request = Request(url, headers={"Range": f"bytes={start}-{stop}"})
    with urlopen(request, timeout=35) as response:
        expected = stop + 1 - start
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not re.fullmatch(rf"bytes {start}-{stop}/\d+", content_range):
            raise ValueError(f"Expected HTTP 206 for bytes {start}-{stop} of {url}")
        data = response.read(expected + 1)
        if len(data) != expected:
            raise ValueError(f"Expected {expected} response bytes for {url}, received {len(data)}")
        return data


def fetch_header(shard, output):
    url = f"{SOURCE}/{shard}?download=true"
    size = int.from_bytes(read_range(url, 0, 7), "little")
    if not 0 < size <= HEADER_LIMIT:
        raise ValueError(f"{shard} header size {size} exceeds the {HEADER_LIMIT}-byte metadata limit")
    header = read_range(url, 8, 7 + size)
    if not isinstance(json.loads(header), dict):
        raise ValueError(f"{shard} header must be a JSON object")
    (output / f"{shard}.header.json").write_bytes(header)
    return len(header)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local pinned config and index")
    parser.add_argument("--output", type=Path, required=True, help="Directory for JSON headers")
    args = parser.parse_args(argv)
    checkpoint = Checkpoint.open(args.checkpoint, metadata_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as workers:
        bytes_read = sum(workers.map(lambda shard: fetch_header(shard, args.output), checkpoint.shards))
    print(f"Fetched {bytes_read:,} metadata bytes from {len(checkpoint.shards)} shards at {REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
