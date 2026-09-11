#!/usr/bin/env python
"""
Convert the MXNet RecordIO CASIA-WebFace dataset (train.rec/train.idx) into
an ImageFolder-style layout that `src.data.FaceFolder` can read directly:

    <output>/<identity_id>/<index>.jpg

Does NOT require mxnet. train.rec's per-image header already carries the
identity label (see notes/decisions.md for how this was verified), so this
script reads the RecordIO container format directly with `struct`.

Record layout (little-endian), read via the offsets in train.idx:
    uint32 magic       (0xced7230a)
    uint32 lrecord      -- top 3 bits: continuation flag (cflag), always 0
                            here (no fragmented records in this file)
                         -- low 29 bits: payload length
    payload[length]:
        24-byte IRHeader: uint32 flag, float32 label, uint64 id, uint64 id2
        remaining bytes: raw JPEG, IF flag == 0 (flag == 2 records are
        per-identity index-range bookkeeping, not images -- skipped)

    # smoke test on a handful of identities
    python scripts/convert_rec_to_folders.py --max-identities 20

    # full default extraction (1000 identities, <=50 images each)
    python scripts/convert_rec_to_folders.py
"""

from __future__ import annotations

import argparse
import io
import struct
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT.parent / "archive" / "casia-webface"
DEFAULT_OUTPUT = ROOT.parent / "archive" / "casia-webface-folders"

MAGIC = 0xCED7230A
CHUNK_HEADER = struct.Struct("<II")
IR_HEADER = struct.Struct("<IfQQ")


def read_idx(idx_path: Path) -> list[tuple[int, int]]:
    entries = []
    with open(idx_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rid, offset = line.split()
            entries.append((int(rid), int(offset)))
    entries.sort()
    return entries


def read_record(f, offset: int, want_payload: bool):
    """Returns (flag, label, payload_or_None) or None if the record is unreadable."""
    f.seek(offset)
    head = f.read(CHUNK_HEADER.size)
    if len(head) < CHUNK_HEADER.size:
        return None
    magic, lrecord = CHUNK_HEADER.unpack(head)
    if magic != MAGIC:
        return None
    length = lrecord & ((1 << 29) - 1)
    if length < IR_HEADER.size:
        return None
    hdr_bytes = f.read(IR_HEADER.size)
    if len(hdr_bytes) < IR_HEADER.size:
        return None
    flag, label, _rid, _rid2 = IR_HEADER.unpack(hdr_bytes)
    if not want_payload:
        return flag, label, None
    payload = f.read(length - IR_HEADER.size)
    if len(payload) < length - IR_HEADER.size:
        return None
    return flag, label, payload


def count_images_per_label(rec_path: Path, idx_entries: list[tuple[int, int]]) -> Counter:
    counts: Counter = Counter()
    with open(rec_path, "rb") as f:
        for _rid, offset in tqdm(idx_entries, desc="pass 1/2: counting images per identity"):
            rec = read_record(f, offset, want_payload=False)
            if rec is None:
                continue
            flag, label, _ = rec
            if flag == 0:
                counts[int(label)] += 1
    return counts


def choose_occurrences(counts: Counter, kept_labels: set[int], max_per_identity: int,
                        rng: np.random.Generator) -> dict[int, set[int]]:
    """For each kept label, randomly choose which of its occurrences (0-indexed in
    record order) to keep, so the sample isn't biased toward whatever came first
    in the scrape/record order."""
    selected = {}
    for label in kept_labels:
        total = counts[label]
        n = min(max_per_identity, total)
        chosen = rng.choice(total, size=n, replace=False)
        selected[label] = set(int(x) for x in chosen)
    return selected


def extract_images(rec_path: Path, idx_entries: list[tuple[int, int]], output: Path,
                    selected_occurrences: dict[int, set[int]]) -> tuple[Counter, int]:
    written: Counter = Counter()
    seen: Counter = Counter()
    unreadable = 0
    with open(rec_path, "rb") as f:
        for _rid, offset in tqdm(idx_entries, desc="pass 2/2: extracting images"):
            flag_label = read_record(f, offset, want_payload=False)
            if flag_label is None:
                continue
            flag, label, _ = flag_label
            if flag != 0:
                continue
            label = int(label)
            if label not in selected_occurrences:
                continue
            occurrence = seen[label]
            seen[label] += 1
            if occurrence not in selected_occurrences[label]:
                continue
            rec = read_record(f, offset, want_payload=True)
            if rec is None:
                unreadable += 1
                continue
            _, _, payload = rec
            try:
                img = Image.open(io.BytesIO(payload))
                img.load()
            except Exception:
                unreadable += 1
                continue
            identity_dir = output / f"{label:05d}"
            identity_dir.mkdir(parents=True, exist_ok=True)
            out_path = identity_dir / f"{written[label]:03d}.jpg"
            with open(out_path, "wb") as out:
                out.write(payload)
            written[label] += 1
    return written, unreadable


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def print_distribution(counts: list[int]) -> None:
    if not counts:
        print("  (nothing written)")
        return
    counts = sorted(counts)
    n = len(counts)
    print(f"  min={counts[0]}  max={counts[-1]}  "
          f"mean={sum(counts) / n:.1f}  median={counts[n // 2]}")
    buckets = Counter()
    for c in counts:
        lo = (c // 10) * 10
        buckets[lo] += 1
    for lo in sorted(buckets):
        bar = "#" * buckets[lo]
        print(f"  [{lo:3d}-{lo + 9:3d}] {buckets[lo]:4d} {bar}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help=f"dir containing train.rec/train.idx (default: {DEFAULT_INPUT})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"output dir, one subfolder per identity (default: {DEFAULT_OUTPUT})")
    p.add_argument("--min-images", type=int, default=20,
                    help="drop identities with fewer than this many images in the source (default: 20)")
    p.add_argument("--max-identities", type=int, default=1000,
                    help="cap on number of identities kept, 0 = no cap (default: 1000)")
    p.add_argument("--max-images-per-identity", type=int, default=50,
                    help="cap on images written per identity, 0 = no cap (default: 50)")
    p.add_argument("--seed", type=int, default=0,
                    help="seed for identity and per-identity image sampling (default: 0)")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    rec_path = args.input / "train.rec"
    idx_path = args.input / "train.idx"
    if not rec_path.exists() or not idx_path.exists():
        raise FileNotFoundError(f"expected train.rec and train.idx under {args.input}")

    idx_entries = read_idx(idx_path)
    print(f"train.idx: {len(idx_entries)} records")

    counts = count_images_per_label(rec_path, idx_entries)
    qualifying = sorted(label for label, c in counts.items() if c >= args.min_images)
    print(f"{len(counts)} identities total, {len(qualifying)} have >= {args.min_images} images")

    if args.max_identities and len(qualifying) > args.max_identities:
        qualifying = sorted(int(x) for x in
                             rng.choice(qualifying, size=args.max_identities, replace=False))
    kept_labels = set(qualifying)
    print(f"keeping {len(kept_labels)} identities (seed={args.seed})")

    max_per_identity = args.max_images_per_identity if args.max_images_per_identity else max(counts.values())
    selected_occurrences = choose_occurrences(counts, kept_labels, max_per_identity, rng)

    args.output.mkdir(parents=True, exist_ok=True)
    written, unreadable = extract_images(rec_path, idx_entries, args.output, selected_occurrences)

    total_images = sum(written.values())
    size = dir_size_bytes(args.output)

    print()
    print("=== summary ===")
    print(f"seed:                {args.seed}")
    print(f"identities kept:     {len(written)}")
    print(f"images written:      {total_images}")
    print(f"unreadable records:  {unreadable}")
    print(f"output size:         {human_bytes(size)}")
    print(f"output dir:          {args.output}")
    print("images-per-identity distribution:")
    print_distribution(list(written.values()))


if __name__ == "__main__":
    main()
