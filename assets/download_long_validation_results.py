"""Download bounded long-validation inference artifacts from the current server."""

from __future__ import annotations

import ast
import os
import pathlib
import stat

import paramiko


ROOT = pathlib.Path(r"F:\study\DiffSHEG")
CONFIG_PATH = ROOT / "deploy_diffsheg.py"
REMOTE_ROOT = "/root/DiffSHEG"
CURRENT_SSH_PORT = 48360
RELATIVE_PATHS = [
    "results/beat_34/test_on_val/beat_FM_aa_x0_aligned_v1/fixStart24/"
    "BestPCK_e479_sequential_overlap24_long3_diverse20",
    "results/beat_34/test_on_val_GT/beat_FM_aa_x0_aligned_v1/fixStart24/"
    "BestPCK_e479_sequential_overlap24_long3_diverse20",
]
SELECTION_PATH = "bvh_output/long_validation_visuals/selection.json"


def load_connection_config(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {"HOST", "USER", "PASS"}:
            values[target.id] = ast.literal_eval(node.value)
    missing = {"HOST", "USER", "PASS"}.difference(values)
    if missing:
        raise RuntimeError(f"Missing connection fields: {sorted(missing)}")
    return values


def download_file(sftp, remote_path, local_path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    partial = local_path.with_suffix(local_path.suffix + ".part")
    sftp.get(remote_path, str(partial))
    os.replace(partial, local_path)
    return local_path.stat().st_size


def download_tree(sftp, remote_dir, local_dir):
    files = 0
    total_bytes = 0
    for item in sftp.listdir_attr(remote_dir):
        remote_path = f"{remote_dir}/{item.filename}"
        local_path = local_dir / item.filename
        if stat.S_ISDIR(item.st_mode):
            child_files, child_bytes = download_tree(sftp, remote_path, local_path)
            files += child_files
            total_bytes += child_bytes
        else:
            total_bytes += download_file(sftp, remote_path, local_path)
            files += 1
    return files, total_bytes


def main():
    config = load_connection_config(CONFIG_PATH)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=config["HOST"],
        port=CURRENT_SSH_PORT,
        username=config["USER"],
        password=config["PASS"],
        timeout=20,
    )
    files = 0
    total_bytes = 0
    try:
        sftp = client.open_sftp()
        try:
            for relative in RELATIVE_PATHS:
                downloaded_files, downloaded_bytes = download_tree(
                    sftp,
                    f"{REMOTE_ROOT}/{relative}",
                    ROOT / pathlib.PurePosixPath(relative),
                )
                files += downloaded_files
                total_bytes += downloaded_bytes
            total_bytes += download_file(
                sftp,
                f"{REMOTE_ROOT}/{SELECTION_PATH}",
                ROOT / pathlib.PurePosixPath(SELECTION_PATH),
            )
            files += 1
        finally:
            sftp.close()
    finally:
        client.close()
    print(f"DOWNLOAD_COMPLETE files={files} bytes={total_bytes}")


if __name__ == "__main__":
    main()
