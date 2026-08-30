"""Resume the selected remote checkpoint without embedding SSH credentials."""

import ast
import hashlib
import os
import pathlib
import sys
import time

import paramiko


ROOT = pathlib.Path(r"F:\study\DiffSHEG")
CONFIG_PATH = ROOT / "deploy_diffsheg.py"
REMOTE_PATH = "/root/DiffSHEG/checkpoints/beat/beat_FM_aa_x0_aligned_v1/model/pck_best.tar"
LOCAL_PATH = ROOT / "checkpoints/beat/beat_FM_aa_x0_aligned_v1/model/pck_best.tar"
EXPECTED_SHA256 = "0502df5330062b57ecf2b191841f0b478f6642b093f9e83bd5ecb230c88c061f"
REQUIRED_CONFIG = {"HOST", "PORT", "USER", "PASS"}
CURRENT_SSH_PORT = 48360


def load_connection_config(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in REQUIRED_CONFIG:
            values[target.id] = ast.literal_eval(node.value)
    missing = REQUIRED_CONFIG.difference(values)
    if missing:
        raise RuntimeError(f"Missing connection fields: {sorted(missing)}")
    return values


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    config = load_connection_config(CONFIG_PATH)
    partial_path = LOCAL_PATH.with_suffix(LOCAL_PATH.suffix + ".part")
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=config["HOST"],
        port=CURRENT_SSH_PORT,
        username=config["USER"],
        password=config["PASS"],
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    try:
        sftp = client.open_sftp()
        try:
            remote_size = sftp.stat(REMOTE_PATH).st_size
            offset = partial_path.stat().st_size if partial_path.exists() else 0
            if offset > remote_size:
                raise RuntimeError(f"Partial file is larger than remote: {offset} > {remote_size}")
            print(f"DOWNLOAD_RESUME {offset}/{remote_size}", flush=True)
            with sftp.open(REMOTE_PATH, "rb") as remote, partial_path.open("ab") as local:
                remote.seek(offset)
                remote.prefetch(file_size=remote_size, max_concurrent_requests=64)
                downloaded = offset
                report_at = time.monotonic()
                report_bytes = offset
                while downloaded < remote_size:
                    chunk = remote.read(min(4 * 1024 * 1024, remote_size - downloaded))
                    if not chunk:
                        raise IOError("Remote stream ended before the advertised file size")
                    local.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - report_at >= 10:
                        speed = (downloaded - report_bytes) / max(now - report_at, 1e-6) / 1024 / 1024
                        print(
                            f"DOWNLOAD_PROGRESS {downloaded}/{remote_size} "
                            f"{downloaded / remote_size:.1%} {speed:.1f} MiB/s",
                            flush=True,
                        )
                        report_at = now
                        report_bytes = downloaded
                local.flush()
                os.fsync(local.fileno())
        finally:
            sftp.close()
    finally:
        client.close()

    actual_hash = sha256(partial_path)
    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(f"SHA256 mismatch: {actual_hash}")
    os.replace(partial_path, LOCAL_PATH)
    print(f"DOWNLOAD_COMPLETE {LOCAL_PATH}", flush=True)
    print(f"SHA256 {actual_hash}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"DOWNLOAD_FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
