"""Preflight, start, and inspect the aligned DiffSHEG production training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import time

import paramiko

from download_long_validation_results import CONFIG_PATH, ROOT, load_connection_config


REMOTE_ROOT = "/root/DiffSHEG"
EXPERIMENT = "beat_FM_aa_x0_aligned_sync_v1"
CACHE_NAME = "beat_4english_15_141_sync_v1"
LOG_RELATIVE = f"logs/train_{EXPERIMENT}.log"
QUEUE_LOG_RELATIVE = f"logs/queue_{EXPERIMENT}.log"
MODEL_RELATIVE = f"checkpoints/beat/{EXPERIMENT}/model"
EXPECTED = {
    "train": {
        "samples": 38468,
        "motion_cache_id": "4680ae765e62451b9651d657b47d1acc",
    },
    "val": {
        "samples": 4609,
        "motion_cache_id": "4b95d723e2764e5bba7555b867f98ee7",
    },
}
CODE_FILES = (
    "runner.py",
    "options/base_options.py",
    "datasets/beat.py",
    "models/flow_matching.py",
    "trainers/ddpm_beat_trainer.py",
    "utils/cache_versions.py",
    "utils/hubert.py",
)
TRAIN_ARGS = (
    "python", "-u", "runner.py",
    "--dataset_name", "beat",
    "--name", EXPERIMENT,
    "--mode", "train",
    "--flow_matching",
    "--fm_expression_condition", "x0",
    "--fm_solver", "rk4",
    "--fm_sample_steps", "50",
    "--fm_transition_blend", "4",
    "--n_poses", "34",
    "--num_epochs", "500",
    "--batch_size", "256",
    "--lr", "0.0002",
    "--addHubert", "True",
    "--encode_hubert", "True",
    "--unidiffuser", "True",
    "--axis_angle", "True",
    "--add_vel_loss", "True",
    "--vel_loss_weight", "100",
    "--acc_loss_weight", "50",
    "--jerk_loss_weight", "10",
    "--x0_rec_weight", "100",
    "--grad_accum_steps", "1",
    "--diversity_loss_weight", "0",
    "--eval_every_e", "10",
    "--max_eval_samples", "512",
    "--latest_every_e", "10",
    "--save_every_e", "50",
    "--no_fgd",
    "--gpu_id", "0",
    "--beat_cache_name", CACHE_NAME,
    "--seed", "1234",
    "--workers", "32",
    "--persistent_workers", "True",
    "--prefetch_factor", "2",
    "--non_blocking_transfer", "True",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "queue", "start", "status"))
    parser.add_argument("--port", type=int, default=48360)
    parser.add_argument("--wait-seconds", type=int, default=20)
    return parser.parse_args()


def connect(port: int) -> paramiko.SSHClient:
    config = load_connection_config(CONFIG_PATH)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=config["HOST"],
        port=port,
        username=config["USER"],
        password=config["PASS"],
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def run(client: paramiko.SSHClient, command: str, *, timeout: int = 120) -> str:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(f"Remote command failed ({status}):\n{output}{error}")
    return output + error


def environment(command: str) -> str:
    payload = (
        "source /root/miniconda3/etc/profile.d/conda.sh && "
        "conda activate diffsheg && "
        f"cd {shlex.quote(REMOTE_ROOT)} && {command}"
    )
    return "bash -lc " + shlex.quote(payload)


def read_json(sftp: paramiko.SFTPClient, relative: str) -> dict:
    path = f"{REMOTE_ROOT}/{relative}"
    with sftp.open(path, "r") as handle:
        return json.loads(handle.read().decode("utf-8"))


def canonical_manifest_id(manifest: dict) -> str:
    payload = dict(manifest)
    payload.pop("manifest_id", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def local_hash(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_hashes(client: paramiko.SSHClient) -> dict[str, str]:
    quoted = " ".join(shlex.quote(relative) for relative in CODE_FILES)
    output = run(client, environment(f"sha256sum {quoted}"))
    return {
        pathlib.PurePosixPath(path).as_posix(): digest
        for digest, path in (line.split(maxsplit=1) for line in output.splitlines())
    }


def validate_caches(client: paramiko.SSHClient) -> dict:
    report: dict[str, dict] = {}
    sftp = client.open_sftp()
    try:
        for split, expected in EXPECTED.items():
            base = f"data/BEAT/beat_cache/{CACHE_NAME}/{split}"
            temporal = read_json(sftp, f"{base}/temporal_alignment_manifest.json")
            motion = read_json(
                sftp,
                f"{base}/bvh_rot_cache_len34_stride10/motion_manifest.json",
            )
            hubert = read_json(
                sftp,
                f"{base}/aud_feat_cache/hubert_large_ls960_ft/manifest.json",
            )
            failures = []
            if temporal.get("manifest_id") != canonical_manifest_id(temporal):
                failures.append("temporal manifest hash is invalid")
            if temporal.get("target_pose_fps") != 15:
                failures.append("pose FPS is not 15")
            if temporal.get("target_facial_fps") != 15:
                failures.append("facial FPS is not 15")
            if temporal.get("audio_sample_rate") != 16000:
                failures.append("audio rate is not 16000")
            if motion.get("cache_version") != 3:
                failures.append("motion cache version is not 3")
            if motion.get("sample_count") != expected["samples"]:
                failures.append("motion sample count mismatch")
            if motion.get("cache_id") != expected["motion_cache_id"]:
                failures.append("motion cache ID mismatch")
            if motion.get("temporal_manifest_id") != temporal.get("manifest_id"):
                failures.append("motion/temporal binding mismatch")
            if hubert.get("sample_count") != expected["samples"]:
                failures.append("HuBERT sample count mismatch")
            if hubert.get("source_motion_cache_version") != 3:
                failures.append("HuBERT source motion version mismatch")
            if hubert.get("source_motion_cache_id") != motion.get("cache_id"):
                failures.append("HuBERT/motion binding mismatch")
            report[split] = {
                "samples": motion.get("sample_count"),
                "temporal_manifest_id": temporal.get("manifest_id"),
                "motion_cache_id": motion.get("cache_id"),
                "hubert_cache_version": hubert.get("cache_version"),
                "failures": failures,
            }
    finally:
        sftp.close()

    count_command = " ; ".join(
        "printf '" + split + "='; find "
        + shlex.quote(
            f"data/BEAT/beat_cache/{CACHE_NAME}/{split}/aud_feat_cache/hubert_large_ls960_ft"
        )
        + " -maxdepth 1 -type f -name '*.npy' | wc -l"
        for split in EXPECTED
    )
    counts = {}
    for line in run(client, environment(count_command)).splitlines():
        split, value = line.split("=", 1)
        counts[split] = int(value)
    for split, expected in EXPECTED.items():
        report[split]["hubert_file_count"] = counts.get(split)
        if counts.get(split) != expected["samples"]:
            report[split]["failures"].append("HuBERT file count mismatch")
    return report


def status_snapshot(client: paramiko.SSHClient) -> str:
    command = (
        "echo PROCESSES; "
        "(pgrep -af '[p]ython.*runner.py' || true); "
        "echo GPU; "
        "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu "
        "--format=csv,noheader; "
        "echo DISK; "
        "df -h /root/autodl-tmp /root 2>/dev/null | tail -n +2; "
        "echo LOG; "
        f"if test -f {shlex.quote(LOG_RELATIVE)}; then "
        f"stat -c '%n %s bytes %y' {shlex.quote(LOG_RELATIVE)}; "
        f"tail -n 30 {shlex.quote(LOG_RELATIVE)}; else echo ABSENT; fi"
    )
    return run(client, environment(command))


def preflight(client: paramiko.SSHClient) -> dict:
    snapshot = status_snapshot(client)
    if EXPERIMENT in "\n".join(
        line for line in snapshot.splitlines() if "runner.py" in line
    ):
        raise RuntimeError(f"Experiment {EXPERIMENT} is already running")
    existence = run(
        client,
        environment(
            f"if test -e {shlex.quote(MODEL_RELATIVE)}; then echo MODEL_EXISTS; fi; "
            f"if test -s {shlex.quote(LOG_RELATIVE)}; then echo LOG_EXISTS; fi"
        ),
    ).splitlines()
    if existence:
        raise RuntimeError(
            "Refusing to overwrite an existing formal experiment: " + ", ".join(existence)
        )
    caches = validate_caches(client)
    failures = [
        f"{split}: {failure}"
        for split, item in caches.items()
        for failure in item["failures"]
    ]
    if failures:
        raise RuntimeError("Cache preflight failed: " + "; ".join(failures))
    local = {relative: local_hash(relative) for relative in CODE_FILES}
    remote = remote_hashes(client)
    mismatches = [
        relative
        for relative, digest in local.items()
        if remote.get(relative) != digest
    ]
    if mismatches:
        raise RuntimeError("Training code differs from verified local code: " + ", ".join(mismatches))
    return {"snapshot": snapshot, "caches": caches, "code_hashes_match": True}


def start(client: paramiko.SSHClient, wait_seconds: int) -> dict:
    audit = preflight(client)
    command = " ".join(shlex.quote(value) for value in TRAIN_ARGS)
    launch = (
        "mkdir -p logs; "
        "export OMP_NUM_THREADS=8; "
        f"nohup {command} > {shlex.quote(LOG_RELATIVE)} 2>&1 < /dev/null & "
        "pid=$!; echo PID=$pid"
    )
    launch_output = run(client, environment(launch))
    time.sleep(wait_seconds)
    snapshot = status_snapshot(client)
    if EXPERIMENT not in "\n".join(
        line for line in snapshot.splitlines() if "runner.py" in line
    ):
        raise RuntimeError("Training process did not remain alive after launch:\n" + snapshot)
    return {"preflight": audit, "launch": launch_output.strip(), "snapshot": snapshot}


def queue_after_full_validation(client: paramiko.SSHClient) -> dict:
    """Queue training behind the already-running bounded full-validation job."""
    audit = preflight(client)
    training_command = " ".join(shlex.quote(value) for value in TRAIN_ARGS)
    body = (
        ". /root/miniconda3/etc/profile.d/conda.sh && "
        "conda activate diffsheg && "
        f"cd {shlex.quote(REMOTE_ROOT)} && "
        "echo QUEUED_AT=$(date -Is) && "
        "while pgrep -f '[r]unner.py.*fullval_blend7_v1' >/dev/null; do "
        "echo WAITING_FOR_FULL_VALIDATION=$(date -Is); sleep 30; done && "
        "echo VALIDATION_RELEASED_AT=$(date -Is) && "
        f"if test -e {shlex.quote(MODEL_RELATIVE)} || test -s {shlex.quote(LOG_RELATIVE)}; then "
        "echo ABORT_EXISTING_EXPERIMENT; exit 31; fi && "
        "export OMP_NUM_THREADS=8 && "
        "( "
        f"nohup {training_command} > {shlex.quote(LOG_RELATIVE)} 2>&1 < /dev/null & "
        "train_pid=$!; echo TRAIN_PID=$train_pid; echo TRAIN_STARTED_AT=$(date -Is); "
        ")"
    )
    queued = (
        "mkdir -p logs; "
        "nohup flock -n /tmp/diffsheg_sync_v1_queue.lock -c "
        f"{shlex.quote(body)} > {shlex.quote(QUEUE_LOG_RELATIVE)} 2>&1 < /dev/null & "
        "echo QUEUE_PID=$!"
    )
    launch_output = run(client, environment(queued))
    time.sleep(2)
    queue_snapshot = run(
        client,
        environment(
            "echo QUEUE_PROCESS; "
            "(pgrep -af '[f]lock.*diffsheg_sync_v1_queue' || true); "
            "echo QUEUE_LOG; "
            f"cat {shlex.quote(QUEUE_LOG_RELATIVE)}"
        ),
    )
    if "WAITING_FOR_FULL_VALIDATION=" not in queue_snapshot:
        raise RuntimeError("Queued launcher did not enter its validation wait state:\n" + queue_snapshot)
    return {
        "preflight": audit,
        "launch": launch_output.strip(),
        "queue_snapshot": queue_snapshot,
    }


def main() -> None:
    args = parse_args()
    client = connect(args.port)
    try:
        if args.action == "preflight":
            result = preflight(client)
        elif args.action == "queue":
            result = queue_after_full_validation(client)
        elif args.action == "start":
            result = start(client, args.wait_seconds)
        else:
            result = {"snapshot": status_snapshot(client)}
    finally:
        client.close()
    print("DIFFSHEG_SYNC_TRAINING=" + json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
