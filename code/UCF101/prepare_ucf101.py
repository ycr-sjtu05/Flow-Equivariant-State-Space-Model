from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

import requests
from tqdm import tqdm


VIDEO_URL = (
    "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
)
SPLIT_URL = (
    "https://www.crcv.ucf.edu/data/UCF101/"
    "UCF101TrainTestSplits-RecognitionTask.zip"
)


def download_file(
    url: str,
    destination: Path,
    *,
    verify_tls: bool,
):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(
        destination.suffix + ".part"
    )

    headers = {}
    existing = partial.stat().st_size if partial.exists() else 0
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    with requests.get(
        url,
        stream=True,
        timeout=60,
        verify=verify_tls,
        headers=headers,
    ) as response:
        if response.status_code == 416:
            partial.replace(destination)
            return
        response.raise_for_status()

        supports_resume = response.status_code == 206
        mode = "ab" if supports_resume and existing > 0 else "wb"
        if mode == "wb":
            existing = 0

        total = response.headers.get("content-length")
        total = (
            int(total) + existing
            if total is not None
            else None
        )

        with partial.open(mode) as handle, tqdm(
            total=total,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))

    partial.replace(destination)


def extract_splits(archive: Path, output_root: Path):
    target = output_root / "ucfTrainTestlist"
    if target.exists():
        print(f"Split directory already exists: {target}")
        return target

    with zipfile.ZipFile(archive, "r") as handle:
        handle.extractall(output_root)
    print(f"Extracted split files to {target}")
    return target


def extract_videos(archive: Path, output_root: Path):
    target = output_root / "UCF-101"
    if target.exists():
        print(f"Video directory already exists: {target}")
        return target

    try:
        import rarfile
    except ImportError as error:
        raise RuntimeError(
            "Install rarfile first: pip install rarfile. "
            "A system extraction backend such as unrar or 7z "
            "is also required."
        ) from error

    try:
        with rarfile.RarFile(archive) as handle:
            handle.extractall(output_root)
    except Exception as error:
        raise RuntimeError(
            "RAR extraction failed. Install `unrar` or `7z`, then run "
            f"`unrar x {archive} {output_root}/`, or extract the archive "
            "manually so that UCF-101/ is inside the data root."
        ) from error

    print(f"Extracted videos to {target}")
    return target


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract official UCF101 files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./data/ucf101"),
    )
    parser.add_argument(
        "--videos",
        action="store_true",
        help="Download/extract the large UCF101 video archive.",
    )
    parser.add_argument(
        "--splits",
        action="store_true",
        help="Download/extract official recognition splits.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification only if the official server fails.",
    )
    args = parser.parse_args()

    if not args.videos and not args.splits:
        args.videos = True
        args.splits = True

    args.root.mkdir(parents=True, exist_ok=True)
    verify_tls = not args.insecure

    try:
        if args.splits:
            split_archive = (
                args.root
                / "UCF101TrainTestSplits-RecognitionTask.zip"
            )
            if not split_archive.exists():
                download_file(
                    SPLIT_URL,
                    split_archive,
                    verify_tls=verify_tls,
                )
            extract_splits(split_archive, args.root)

        if args.videos:
            video_archive = args.root / "UCF101.rar"
            if not video_archive.exists():
                download_file(
                    VIDEO_URL,
                    video_archive,
                    verify_tls=verify_tls,
                )
            extract_videos(video_archive, args.root)

    except requests.RequestException as error:
        print(
            "\nAutomatic download failed. Download the files manually "
            "from the official UCF101 website and arrange them as:\n"
            "  <root>/UCF-101/<ClassName>/*.avi\n"
            "  <root>/ucfTrainTestlist/classInd.txt\n"
            "  <root>/ucfTrainTestlist/trainlist01.txt\n"
            "  <root>/ucfTrainTestlist/testlist01.txt\n",
            file=sys.stderr,
        )
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
