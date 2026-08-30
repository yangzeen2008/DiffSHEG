"""Deploy and validate the bounded SO(3) long-sequence inference repair."""

from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import stat
import time

import paramiko

from download_long_validation_results import (
    CONFIG_PATH,
    CURRENT_SSH_PORT,
    ROOT,
    load_connection_config,
)


REMOTE_ROOT = "/root/DiffSHEG"
SSH_PORT = CURRENT_SSH_PORT
TRANSITION_BLEND = 7
RESULT_TAG = "long3_so3release"
RESULT_RELATIVE = ""
LOG_RELATIVE = ""


def configure(*, port, blend, tag):
    global SSH_PORT, TRANSITION_BLEND, RESULT_TAG, RESULT_RELATIVE, LOG_RELATIVE
    SSH_PORT = int(port)
    TRANSITION_BLEND = int(blend)
    RESULT_TAG = tag
    RESULT_RELATIVE = (
        "results/beat_34/test_on_val/beat_FM_aa_x0_aligned_v1/fixStart24/"
        "BestPCK_e479_sequential_overlap24_" + RESULT_TAG
    )
    LOG_RELATIVE = f"logs/infer_{RESULT_TAG}.log"


def connect():
    config = load_connection_config(CONFIG_PATH)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=config["HOST"],
        port=SSH_PORT,
        username=config["USER"],
        password=config["PASS"],
        timeout=20,
    )
    return client


def run(client, command, *, timeout=None):
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(
            f"Remote command failed with status {status}:\n{output}{error}"
        )
    return output + error


def environment(command):
    payload = (
        "source /root/miniconda3/etc/profile.d/conda.sh && "
        "conda activate diffsheg && "
        f"cd {shlex.quote(REMOTE_ROOT)} && {command}"
    )
    return "bash -lc " + shlex.quote(payload)


def status(client):
    command = (
        f"cd {shlex.quote(REMOTE_ROOT)} && "
        "(pgrep -af '[p]ython.*runner.py' || true); "
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader; "
        f"if test -f {shlex.quote(LOG_RELATIVE)}; then "
        f"tail -n 12 {shlex.quote(LOG_RELATIVE)}; fi"
    )
    print(run(client, command).strip())


def upload(client):
    uploads = (
        (ROOT / "models" / "flow_matching.py", "models/flow_matching.py"),
        (ROOT / "motion_repair_checks.py", "motion_repair_checks.py"),
    )
    sftp = client.open_sftp()
    try:
        for local_path, relative in uploads:
            remote_path = f"{REMOTE_ROOT}/{relative}"
            partial_path = remote_path + ".so3.part"
            sftp.put(str(local_path), partial_path)
            run(
                client,
                f"cd {shlex.quote(REMOTE_ROOT)} && "
                f"cp -p {shlex.quote(relative)} {shlex.quote(relative + '.pre_so3')} && "
                f"mv {shlex.quote(relative + '.so3.part')} {shlex.quote(relative)}",
            )
            print(f"UPLOADED {relative}")
    finally:
        sftp.close()


def test(client):
    output = run(
        client,
        environment("python -m unittest motion_repair_checks.py -v"),
        timeout=180,
    )
    print(output.strip())


def verify_stats(client):
    code = (
        "from types import SimpleNamespace; "
        "from models.flow_matching import FlowMatching; "
        "o=SimpleNamespace(fm_sample_steps=1,fm_solver='euler',"
        f"fm_transition_blend={TRANSITION_BLEND},axis_angle=True,rot_6d=False,split_pos=141,"
        "mean_pose_path='data/BEAT/beat_cache/beat_4english_15_141/train/'); "
        "f=FlowMatching(o); "
        "print('SO3_STATS_READY=%s GESTURE_DIM=%d' % "
        "(f.axis_angle_mean is not None and f.axis_angle_std is not None, "
        "f.gesture_dim))"
    )
    print(run(client, environment("python -c " + shlex.quote(code))).strip())


def start(client):
    running = run(
        client,
        "pgrep -af '[p]ython.*runner.py' || true",
    ).strip()
    if running:
        raise RuntimeError(f"A runner process is already active:\n{running}")

    args = [
        "python", "runner.py",
        "--dataset_name", "beat",
        "--name", "beat_FM_aa_x0_aligned_v1",
        "--mode", "test",
        "--test_on_val",
        "--flow_matching",
        "--fm_expression_condition", "x0",
        "--fm_solver", "rk4",
        "--fm_sample_steps", "50",
        "--fm_transition_blend", str(TRANSITION_BLEND),
        "--n_poses", "34",
        "--batch_size", "60",
        "--addHubert", "True",
        "--encode_hubert", "True",
        "--unidiffuser", "True",
        "--axis_angle", "True",
        "--gpu_id", "0",
        "--beat_cache_name", "beat_4english_15_141",
        "--ckpt", "pck_best.tar",
        "--sequential_test_windows",
        "--overlap_len", "24",
        "--test_window_ranges", "3571:20,3799:20,944:20",
        "--test_result_tag", RESULT_TAG,
        "--no_fgd",
        "--workers", "8",
    ]
    command = " ".join(shlex.quote(value) for value in args)
    launch = (
        f"mkdir -p logs && rm -f {shlex.quote(LOG_RELATIVE)} && "
        f"nohup {command} > {shlex.quote(LOG_RELATIVE)} 2>&1 < /dev/null & "
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
        raise RuntimeError(f"Inference did not start; inspect {LOG_RELATIVE}")
    print(f"STARTED pid={pid} log={LOG_RELATIVE}")


def poll(client):
    result_path = f"{REMOTE_ROOT}/{RESULT_RELATIVE}"
    command = (
        "pid=$(pgrep -f '[p]ython.*runner.py' | head -n 1 || true); "
        "if test -n \"$pid\"; then "
        "echo RUNNING_PID=$pid; ps -p $pid -o etimes=,stat=,pcpu=,pmem=; "
        "else echo RUNNING_PID=none; fi; "
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader; "
        f"for d in gesture expression audio; do if test -d {shlex.quote(result_path)}/$d; "
        f"then echo FILES_$d=$(find {shlex.quote(result_path)}/$d -maxdepth 1 -type f | wc -l); fi; done; "
        f"if test -f {shlex.quote(REMOTE_ROOT + '/' + LOG_RELATIVE)}; then "
        f"tail -c 4000 {shlex.quote(REMOTE_ROOT + '/' + LOG_RELATIVE)} "
        "| tr '\\r' '\\n' | tail -n 12; fi"
    )
    print(run(client, command).strip())


def download_tree(sftp, remote_dir, local_dir):
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


def download(client):
    sftp = client.open_sftp()
    try:
        files, total = download_tree(
            sftp,
            f"{REMOTE_ROOT}/{RESULT_RELATIVE}",
            ROOT / pathlib.PurePosixPath(RESULT_RELATIVE),
        )
    finally:
        sftp.close()
    print(f"DOWNLOAD_COMPLETE files={files} bytes={total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "status", "upload", "test", "verify_stats", "start", "poll", "download"
        ),
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("DIFFSHEG_SSH_PORT", CURRENT_SSH_PORT)),
        help="Current dynamic SSH gateway port",
    )
    parser.add_argument("--blend", type=int, default=7)
    parser.add_argument(
        "--tag", default=None,
        help="Result tag; defaults to long3_so3release for blend 7 and an ablation tag otherwise",
    )
    args = parser.parse_args()
    tag = args.tag or (
        "long3_so3release" if args.blend == 7
        else f"long3_so3release_blend{args.blend}"
    )
    configure(port=args.port, blend=args.blend, tag=tag)
    client = connect()
    try:
        globals()[args.action](client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
