import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


SERVERS = {
    "EU": "http://canis.vc.in.tum.de:8100/",
    "EU2": "http://kaldir.vc.in.tum.de/faceforensics/",
    "CA": "http://falas.cmpt.sfu.ca:8100/",
}


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "datasets" / "faceforensics"


def _read_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_file(url: str, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        print(f"[SKIP] {out_file}")
        return

    start = time.time()
    fd, tmp_name = tempfile.mkstemp(dir=out_file.parent)
    os.close(fd)

    def reporthook(count, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = min(count * block_size, total_size)
        percent = int(downloaded * 100 / total_size)
        mb = downloaded / (1024 * 1024)
        speed = downloaded / max(time.time() - start, 1) / 1024
        sys.stdout.write(f"\r  {percent:3d}% {mb:8.1f} MB {speed:8.0f} KB/s")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp_name, reporthook=reporthook)
        os.replace(tmp_name, out_file)
        print(f"\n[OK] {out_file.name}")
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _original_filenames(base_url: str, limit: int) -> list[str]:
    pairs = _read_json(f"{base_url}/misc/filelist.json")
    filenames = []
    for pair in pairs:
        filenames.extend(pair)
    return [f"{name}.mp4" for name in filenames[:limit]]


def _deepfake_filenames(base_url: str, limit: int) -> list[str]:
    pairs = _read_json(f"{base_url}/misc/filelist.json")
    filenames = []
    for pair in pairs:
        filenames.append("_".join(pair))
        filenames.append("_".join(pair[::-1]))
    return [f"{name}.mp4" for name in filenames[:limit]]


def _candidate_servers(server: str) -> list[str]:
    if server == "AUTO":
        return list(SERVERS)
    return [server, *[name for name in SERVERS if name != server]]


def _select_working_server(server: str) -> tuple[str, str, list[str]]:
    errors = []
    for name in _candidate_servers(server):
        server_url = SERVERS[name]
        base_url = f"{server_url}v3"
        try:
            _read_json(f"{base_url}/misc/filelist.json")
            return name, server_url, errors
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            errors.append(f"{name} ({server_url}): {exc}")

    raise ConnectionError(
        "Could not reach any FaceForensics++ mirror.\n"
        + "\n".join(f" - {error}" for error in errors)
        + "\n\nThis is usually a remote-server, firewall, proxy, or network issue. "
        "Try again later, try another network/VPN, or manually download using the "
        "official FaceForensics++ script from GitHub."
    )


def download_subset(output_dir: Path, compression: str, num_videos: int, server: str) -> None:
    server_name, server_url, previous_errors = _select_working_server(server)
    base_url = f"{server_url}v3"
    tos_url = f"{server_url}webpage/FaceForensics_TOS.pdf"

    if previous_errors:
        print("[WARN] Some mirrors were unavailable:")
        for error in previous_errors:
            print(f" - {error}")
        print()

    print(f"[INFO] Using FaceForensics++ mirror: {server_name} ({server_url})")

    print("FaceForensics++ terms of use:")
    print(tos_url)
    print()
    answer = input("Type YES if you have read and agree to the dataset terms: ").strip()
    if answer != "YES":
        raise SystemExit("[ABORTED] Terms were not accepted.")

    original_dir = output_dir / "original"
    manipulated_dir = output_dir / "manipulated"

    print(f"\n[INFO] Downloading {num_videos} original videos to {original_dir}")
    for filename in _original_filenames(base_url, num_videos):
        url = f"{base_url}/original_sequences/youtube/{compression}/videos/{filename}"
        _download_file(url, original_dir / filename)

    print(f"\n[INFO] Downloading {num_videos} Deepfakes videos to {manipulated_dir}")
    for filename in _deepfake_filenames(base_url, num_videos):
        url = f"{base_url}/manipulated_sequences/Deepfakes/{compression}/videos/{filename}"
        _download_file(url, manipulated_dir / filename)

    print("\n[DONE] FaceForensics++ subset downloaded.")
    print("Next step: run train_deepfake.bat")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a small FaceForensics++ subset for this project")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--compression", choices=["raw", "c23", "c40"], default="c23")
    parser.add_argument("--num-videos", type=int, default=10)
    parser.add_argument("--server", choices=["AUTO", *SERVERS], default="AUTO")
    args = parser.parse_args()

    download_subset(
        output_dir=Path(args.output_dir),
        compression=args.compression,
        num_videos=args.num_videos,
        server=args.server,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
