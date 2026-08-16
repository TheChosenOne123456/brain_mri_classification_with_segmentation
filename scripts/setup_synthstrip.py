"""下载官方 SynthStrip 命令脚本和成人含 CSF 模型权重。"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import ProxyHandler, build_opener


SCRIPT_URL = (
    "https://raw.githubusercontent.com/freesurfer/freesurfer/"
    "dev/mri_synthstrip/mri_synthstrip"
)
MODEL_URL = (
    "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/"
    "requirements/synthstrip.1.pt"
)
MODEL_SHA256 = "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    opener,
    url,
    destination: Path,
    expected_sha256=None,
    force=False,
    timeout=60,
):
    if destination.is_file() and not force:
        digest = sha256_file(destination)
        if expected_sha256 is None or digest == expected_sha256:
            print(f"Already present: {destination}")
            return digest
        raise RuntimeError(
            f"Existing file has unexpected SHA-256: {destination} | {digest}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".download",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            print(f"Downloading: {url}")
            with opener.open(url, timeout=timeout) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temporary.write(chunk)

        digest = sha256_file(temporary_path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError(
                f"Downloaded SHA-256 mismatch for {url}: "
                f"expected {expected_sha256}, got {digest}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        print(f"Saved: {destination} | sha256={digest}")
        return digest
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main(args):
    output_root = Path(args.output_root).expanduser().resolve()
    tool_root = output_root / "synthstrip"
    script_path = tool_root / "mri_synthstrip"
    model_path = tool_root / "synthstrip.1.pt"
    opener = build_opener(ProxyHandler({}) if args.no_proxy else ProxyHandler())

    script_sha256 = download(
        opener,
        SCRIPT_URL,
        script_path,
        force=args.force,
        timeout=args.timeout,
    )
    model_sha256 = download(
        opener,
        MODEL_URL,
        model_path,
        expected_sha256=MODEL_SHA256,
        force=args.force,
        timeout=args.timeout,
    )
    script_path.chmod(script_path.stat().st_mode | 0o111)

    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": {
            "url": SCRIPT_URL,
            "path": str(script_path),
            "sha256": script_sha256,
        },
        "model": {
            "url": MODEL_URL,
            "path": str(model_path),
            "sha256": model_sha256,
        },
    }
    manifest_path = tool_root / "download_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest: {manifest_path}")
    print("Dependency: python -m pip install surfa==0.6.3")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the official SynthStrip script and model."
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Data experiment root, for example output/data-synthstrip",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload existing files.",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Ignore HTTP(S) proxy environment variables for the downloads.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request network timeout in seconds (default: 60).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
