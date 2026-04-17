"""Batch-run DIRE inference on a folder of images and dump scores to CSV.

Usage:
    python scripts/measure_distribution.py \\
        --input-dir samples/ai \\
        --label fake \\
        --api-url https://[POD_ID]-8000.proxy.runpod.net \\
        --out dist_fake.csv

Iterate over a directory of labelled samples (real press photos in one run,
AI-generated samples in another), POST each to the RunPod inference endpoint,
and write per-image DIRE scores to CSV.

Use the output to:
- Decide the threshold_risk cutoff (e.g. 90th percentile of "real" scores)
- Measure the separation between real vs. diffusion-generated distributions
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import httpx


def analyze_one(client: httpx.Client, api_url: str, image_path: Path) -> dict:
    with image_path.open("rb") as fh:
        files = {"image": (image_path.name, fh, "image/jpeg")}
        up = client.post(f"{api_url}/upload", files=files, timeout=60.0)
    up.raise_for_status()
    file_id = up.json()["file_id"]

    r = client.post(f"{api_url}/infer", json={"file_id": file_id}, timeout=180.0)
    r.raise_for_status()
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--label", required=True, choices=["real", "fake"])
    ap.add_argument("--api-url", required=True, help="RunPod inference base URL")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    images = sorted(
        p for p in args.input_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        print(f"no images found under {args.input_dir}", file=sys.stderr)
        return 1

    fieldnames = ["filename", "label", "dire"]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", newline="", encoding="utf-8") as fh, httpx.Client() as client:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i, p in enumerate(images, 1):
            try:
                res = analyze_one(client, args.api_url, p)
            except Exception as e:
                print(f"[{i}/{len(images)}] {p.name}: FAILED ({e})", file=sys.stderr)
                continue
            row = {
                "filename": p.name,
                "label": args.label,
                "dire": res["dire"]["score"],
            }
            writer.writerow(row)
            print(f"[{i}/{len(images)}] {p.name}: dire={row['dire']:.3f}")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
