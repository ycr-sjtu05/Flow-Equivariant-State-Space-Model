from __future__ import annotations

import argparse
from pathlib import Path

from sfm.datasets.vil_index import build_vil_index
from sfm.utils.paths import CACHE_ROOT, get_vil_h5_files, ensure_project_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(CACHE_ROOT / "vil_index.json"),
        help="Where to save the built index JSON",
    )
    args = parser.parse_args()

    ensure_project_dirs()

    h5_files = get_vil_h5_files()
    print(f"Found {len(h5_files)} .h5 files")
    for p in h5_files:
        print(f"  {p}")

    index = build_vil_index(h5_files=h5_files, save_path=args.save_path)

    print("\n=== Index Summary ===")
    print(f"Saved to: {Path(args.save_path).resolve()}")
    print(f"num_files: {index['num_files']}")
    print(f"total_samples: {index['total_samples']}")

    print("\n=== Per-file Layout ===")
    for fm in index["files"]:
        print(
            f"- {Path(fm['path']).name} | "
            f"key={fm['video_key']} | "
            f"shape={tuple(fm['shape'])} | "
            f"sample_axis={fm['sample_axis']} | "
            f"time_axis={fm['time_axis']} | "
            f"H_axis={fm['height_axis']} | "
            f"W_axis={fm['width_axis']} | "
            f"num_samples={fm['num_samples']} | "
            f"num_frames={fm['num_frames']}"
        )


if __name__ == "__main__":
    main()
