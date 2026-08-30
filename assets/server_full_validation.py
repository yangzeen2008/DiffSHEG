"""Orchestrate reproducible blend-5/blend-7 validation on the current server."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import stat
import tarfile
import time

import paramiko

from download_long_validation_results import (
    CONFIG_PATH,
    CURRENT_SSH_PORT,
    ROOT,
    load_connection_config,
)


REMOTE_ROOT = "/root/DiffSHEG"
EXPERIMENT = "beat_FM_aa_x0_aligned_v1"
CHECKPOINT = "pck_best.tar"
CHECKPOINT_EPOCH = 479
PREFLIGHT_RANGES = "0:20,1000:20,2000:20,3000:20,4000:20"
CODE_FILES = (
    "models/flow_matching.py",
    "trainers/ddpm_beat_trainer.py",
    "runner.py",
    "options/base_options.py",
    "datasets/beat.py",
    "utils/window_stitching.py",
    "utils/test_selection.py",
    "utils/cache_versions.py",
    "utils/motion_metrics.py",
)


def connect(port: int):
    config = load_connection_config(CONFIG_PATH)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=config["HOST"],
        port=port,
        username=config["USER"],
        password=config["PASS"],
        timeout=20,
    )
    return client


def run(client, command: str, *, timeout=None) -> str:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(
            f"Remote command failed with status {status}:\n{output}{error}"
        )
    return output + error


def environment(command: str) -> str:
    payload = (
        "source /root/miniconda3/etc/profile.d/conda.sh && "
        "conda activate diffsheg && "
        f"cd {shlex.quote(REMOTE_ROOT)} && {command}"
    )
    return "bash -lc " + shlex.quote(payload)


def result_relative(tag: str, *, output_gt: bool) -> str:
    split = "test_on_val_GT" if output_gt else "test_on_val"
    return (
        f"results/beat_34/{split}/{EXPERIMENT}/fixStart24/"
        f"BestPCK_e{CHECKPOINT_EPOCH}_sequential_overlap24_{tag}"
    )


def log_relative(tag: str) -> str:
    return f"logs/infer_{tag}.log"


def local_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(client) -> None:
    local_hashes = {
        relative: local_sha256(ROOT / relative)
        for relative in CODE_FILES
        if (ROOT / relative).exists()
    }
    quoted_files = " ".join(shlex.quote(relative) for relative in CODE_FILES)
    checkpoint = (
        f"checkpoints/beat/{EXPERIMENT}/model/{CHECKPOINT}"
    )
    remote = run(
        client,
        environment(
            "echo SERVER_STATUS; "
            "(pgrep -af '[p]ython.*runner.py' || true); "
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader; "
            "df -h /root/autodl-tmp /root 2>/dev/null | tail -n +2; "
            f"test -s {shlex.quote(checkpoint)}; "
            f"stat -c 'CHECKPOINT_BYTES=%s CHECKPOINT_MTIME=%y' {shlex.quote(checkpoint)}; "
            f"sha256sum {quoted_files}"
        ),
        timeout=120,
    )
    remote_hashes = {}
    for line in remote.splitlines():
        fields = line.split()
        if len(fields) == 2 and len(fields[0]) == 64:
            remote_hashes[fields[1].lstrip("./")] = fields[0]
    comparisons = {
        relative: {
            "local": local_hashes.get(relative),
            "remote": remote_hashes.get(relative),
            "match": local_hashes.get(relative) == remote_hashes.get(relative),
        }
        for relative in CODE_FILES
    }
    print(remote.strip())
    print("CODE_HASH_COMPARISON=" + json.dumps(comparisons, sort_keys=True))


def build_args(*, blend: int, tag: str, ranges: str | None, output_gt: bool):
    args = [
        "python", "runner.py",
        "--dataset_name", "beat",
        "--name", EXPERIMENT,
        "--mode", "test",
        "--test_on_val",
        "--flow_matching",
        "--fm_expression_condition", "x0",
        "--fm_solver", "rk4",
        "--fm_sample_steps", "50",
        "--fm_transition_blend", str(blend),
        "--n_poses", "34",
        "--batch_size", "60",
        "--addHubert", "True",
        "--encode_hubert", "True",
        "--unidiffuser", "True",
        "--axis_angle", "True",
        "--gpu_id", "0",
        "--beat_cache_name", "beat_4english_15_141",
        # This checkpoint was trained/evaluated with the legacy cache. The flag
        # only acknowledges that cache for diagnostic inference; runner.py
        # still forbids it for training.
        "--allow_legacy_motion_cache",
        "--ckpt", CHECKPOINT,
        "--sequential_test_windows",
        "--overlap_len", "24",
        "--test_result_tag", tag,
        "--no_fgd",
        "--workers", "8",
    ]
    if ranges:
        args.extend(("--test_window_ranges", ranges))
    if output_gt:
        args.append("--output_gt")
    return args


def start(
    client,
    *,
    blend: int,
    tag: str,
    ranges: str | None,
    output_gt: bool,
) -> None:
    running = run(client, "pgrep -af '[p]ython.*runner.py' || true").strip()
    if running:
        raise RuntimeError(f"A runner process is already active:\n{running}")
    relative_result = result_relative(tag, output_gt=output_gt)
    relative_log = log_relative(tag)
    existence = run(
        client,
        f"cd {shlex.quote(REMOTE_ROOT)} && "
        f"if test -e {shlex.quote(relative_result)}; then echo RESULT_EXISTS; fi; "
        f"if test -e {shlex.quote(relative_log)}; then echo LOG_EXISTS; fi",
    ).strip()
    if existence:
        raise RuntimeError(
            f"Refusing to overwrite existing preflight/full-validation output:\n{existence}"
        )

    command = " ".join(
        shlex.quote(value)
        for value in build_args(
            blend=blend, tag=tag, ranges=ranges, output_gt=output_gt
        )
    )
    launch = (
        "mkdir -p logs && "
        f"nohup {command} > {shlex.quote(relative_log)} 2>&1 < /dev/null & "
        "echo $!"
    )
    channel = client.get_transport().open_session()
    channel.exec_command(environment(launch))
    time.sleep(2.0)
    pid = run(
        client,
        "pgrep -f '[p]ython.*runner.py' | tail -n 1 || true",
    ).strip()
    channel.close()
    if not pid:
        raise RuntimeError(f"Validation did not start; inspect {relative_log}")
    print(
        f"STARTED pid={pid} blend={blend} output_gt={output_gt} "
        f"ranges={ranges or 'FULL'} tag={tag} log={relative_log}"
    )


def poll(client, *, tag: str, output_gt: bool) -> None:
    relative_result = result_relative(tag, output_gt=output_gt)
    relative_log = log_relative(tag)
    command = (
        "pid=$(pgrep -f '[p]ython.*runner.py' | head -n 1 || true); "
        "if test -n \"$pid\"; then echo RUNNING_PID=$pid; "
        "ps -p $pid -o etimes=,stat=,pcpu=,pmem=; "
        "else echo RUNNING_PID=none; fi; "
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader; "
        f"for d in gesture expression audio; do if test -d {shlex.quote(relative_result)}/$d; "
        f"then echo FILES_$d=$(find {shlex.quote(relative_result)}/$d -maxdepth 1 -type f | wc -l); fi; done; "
        f"if test -f {shlex.quote(relative_log)}; then "
        f"stat -c 'LOG_BIRTHTIME=%W LOG_MTIME=%Y LOG_BYTES=%s' {shlex.quote(relative_log)}; "
        f"tail -c 5000 {shlex.quote(relative_log)} | tr '\\r' '\\n' | tail -n 18; fi"
    )
    print(run(client, f"cd {shlex.quote(REMOTE_ROOT)} && {command}").strip())


def download_tree(sftp, remote_dir: str, local_dir: pathlib.Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    files = 0
    total = 0
    for item in sftp.listdir_attr(remote_dir):
        remote_path = f"{remote_dir}/{item.filename}"
        local_path = local_dir / item.filename
        if stat.S_ISDIR(item.st_mode):
            child_files, child_total = download_tree(sftp, remote_path, local_path)
            files += child_files
            total += child_total
        else:
            partial = local_path.with_suffix(local_path.suffix + ".part")
            sftp.get(remote_path, str(partial))
            partial.replace(local_path)
            files += 1
            total += local_path.stat().st_size
    return files, total


def download(client, *, tag: str, output_gt: bool) -> None:
    relative_result = result_relative(tag, output_gt=output_gt)
    local_root = ROOT / pathlib.PurePosixPath(relative_result)
    sftp = client.open_sftp()
    try:
        files, total = download_tree(
            sftp, f"{REMOTE_ROOT}/{relative_result}", local_root
        )
    finally:
        sftp.close()
    print(f"DOWNLOAD_COMPLETE tag={tag} files={files} bytes={total}")


def download_archive(client, *, tag: str, output_gt: bool) -> None:
    """Download many small result files as one temporary verified archive."""
    relative_result = result_relative(tag, output_gt=output_gt)
    kind = "gt" if output_gt else "prediction"
    remote_archive = f"/root/autodl-tmp/{tag}_{kind}.tar.gz"
    local_archive = ROOT / "bvh_output" / "full_validation" / "downloads" / (
        f"{tag}_{kind}.tar.gz"
    )
    local_archive.parent.mkdir(parents=True, exist_ok=True)
    run(
        client,
        f"tar -C {shlex.quote(REMOTE_ROOT)} -czf {shlex.quote(remote_archive)} "
        f"{shlex.quote(relative_result)}",
        timeout=300,
    )
    sftp = client.open_sftp()
    partial = local_archive.with_suffix(local_archive.suffix + ".part")
    try:
        sftp.get(remote_archive, str(partial))
        partial.replace(local_archive)
    finally:
        sftp.close()
        run(client, f"rm -f {shlex.quote(remote_archive)}")

    root_resolved = ROOT.resolve()
    with tarfile.open(local_archive, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            destination = (ROOT / member.name).resolve()
            if root_resolved not in (destination, *destination.parents):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        archive.extractall(ROOT)
    bytes_downloaded = local_archive.stat().st_size
    local_archive.unlink()
    print(
        f"ARCHIVE_DOWNLOAD_COMPLETE tag={tag} members={len(members)} "
        f"compressed_bytes={bytes_downloaded}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("audit", "start", "poll", "download", "download_archive")
    )
    parser.add_argument("--port", type=int, default=CURRENT_SSH_PORT)
    parser.add_argument("--blend", type=int, choices=(5, 7), default=7)
    parser.add_argument("--tag")
    parser.add_argument("--scope", choices=("preflight", "full"), default="preflight")
    parser.add_argument("--output-gt", action="store_true")
    args = parser.parse_args()
    ranges = PREFLIGHT_RANGES if args.scope == "preflight" else None
    tag = args.tag or (
        f"{args.scope}{100 if ranges else ''}_blend{args.blend}_v1"
    )

    client = connect(args.port)
    try:
        if args.action == "audit":
            audit(client)
        elif args.action == "start":
            start(
                client,
                blend=args.blend,
                tag=tag,
                ranges=ranges,
                output_gt=args.output_gt,
            )
        elif args.action == "poll":
            poll(client, tag=tag, output_gt=args.output_gt)
        elif args.action == "download":
            download(client, tag=tag, output_gt=args.output_gt)
        elif args.action == "download_archive":
            download_archive(client, tag=tag, output_gt=args.output_gt)
    finally:
        client.close()


if __name__ == "__main__":
    main()
