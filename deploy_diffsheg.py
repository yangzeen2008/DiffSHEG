import subprocess
import paramiko
import os
import sys

# ================= 配置区 =================
HOST = "connect.westd.seetacloud.com"
PORT = 22722
USER = "root"
PASS = "Yangzeen2008#"
LOCAL_DIR = r"f:\study\DiffSHEG"
REMOTE_DIR = "/root/DiffSHEG"

# ================= 流程函数 =================
def ssh_run(cmd, print_output=True):
    print(f"\n[Remote Exec] {cmd}")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, PORT, USER, PASS, timeout=10)
    
    stdin, stdout, stderr = ssh.exec_command(cmd)
    
    # 获取实时输出
    while True:
        line = stdout.readline()
        if not line:
            break
        if print_output:
            print(line, end="")
            
    err = stderr.read().decode()
    if err and print_output:
        print("STDERR:", err, file=sys.stderr)
        
    exit_status = stdout.channel.recv_exit_status()
    ssh.close()
    return exit_status

def scp_upload(src, dest_is_dir=True):
    print(f"\n[SCP Upload] 正在上传 {src} 到服务器...")
    # 构造 SCP 命令，使用自带的 scp 工具
    dest_path = REMOTE_DIR if dest_is_dir else ""
    cmd = [
        "scp", "-P", str(PORT), "-r", 
        src, 
        f"{USER}@{HOST}:{dest_path}"
    ]
    # 在 Windows 上由于我们没有配置 SSH 密钥，scp 命令会卡住要求输入密码。
    # 为了全自动化，我们推荐用户手动跑 SCP，或者用更高级的第三方库。
    # 这里我们只生成命令供参考，因为 subprocess 无法轻易注入密码交互。
    print("由于系统安全限制，SCP传输大文件推荐使用以下命令手动执行（需输入密码 Yangzeen2008#）：")
    print(" ".join(cmd))
    print()

def step_1_prepare_server():
    print(">>> 阶段 1: 准备远程服务器环境")
    ssh_run(f"mkdir -p {REMOTE_DIR}/data/BEAT/raw {REMOTE_DIR}/checkpoints")
    
def step_2_setup_env():
    print(">>> 阶段 2: 安装 Conda 与 Python 依赖")
    setup_cmds = f"""
    source /root/miniconda3/etc/profile.d/conda.sh || source /opt/conda/etc/profile.d/conda.sh
    conda create -n diffsheg python=3.9 -y
    conda activate diffsheg
    export OMP_NUM_THREADS=8
    pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -U openmim
    mim install mmcv-full==1.7.2
    pip install loguru einops termcolor scikit-learn pandas soundfile librosa tqdm lmdb transformers==4.37.2
    """
    ssh_run(setup_cmds)

def step_3_transfer_files():
    print(">>> 阶段 3: 请在你的电脑终端复制并执行以下命令上传文件（需输入密码）")
    print("--- 1. 上传代码 ---")
    print(f'scp -P {PORT} -r "{LOCAL_DIR}\\datasets" "{LOCAL_DIR}\\models" "{LOCAL_DIR}\\trainers" "{LOCAL_DIR}\\options" "{LOCAL_DIR}\\*.py" {USER}@{HOST}:{REMOTE_DIR}/')
    
    print("\n--- 2. 上传 Checkpoint (用于断点续训) ---")
    print(f'scp -P {PORT} -r "{LOCAL_DIR}\\checkpoints\\beat" {USER}@{HOST}:{REMOTE_DIR}/checkpoints/')
    
    print("\n--- 3. 上传 22GB 原数据 (建议晚上挂机传) ---")
    print(f'scp -P {PORT} -r "{LOCAL_DIR}\\data\\BEAT\\raw\\*" {USER}@{HOST}:{REMOTE_DIR}/data/BEAT/raw/')

def step_4_start_training():
    print(">>> 阶段 4: 开始数据预处理和训练")
    run_cmd = f"""
    cd {REMOTE_DIR}
    source /root/miniconda3/etc/profile.d/conda.sh || source /opt/conda/etc/profile.d/conda.sh
    conda activate diffsheg
    export OMP_NUM_THREADS=8
    
    echo "1. 重新生成 141维 BVH"
    python preprocess_beat.py
    
    echo "2. 构建带版本信息的动作缓存"
    python runner.py --dataset_name beat --n_poses 34 --mode prepare_cache --cache_splits train_val --rebuild_motion_cache

    echo "3. 按动作 LMDB 索引重新提取 HuBERT 特征"
    export HF_ENDPOINT=https://hf-mirror.com
    python build_hubert_cache.py --split train --force
    python build_hubert_cache.py --split val --force
    
    echo "4. 启动全新训练 (nohup 后台运行)"
    mkdir -p logs
    nohup python runner.py --dataset_name beat --name beat_FM_aa_x0_aligned_sync_v1 --mode train --flow_matching --fm_expression_condition x0 --fm_solver rk4 --fm_sample_steps 50 --fm_transition_blend 4 --n_poses 34 --num_epochs 500 --batch_size 256 --lr 0.0002 --addHubert True --encode_hubert True --unidiffuser True --axis_angle True --add_vel_loss True --vel_loss_weight 100 --acc_loss_weight 50 --jerk_loss_weight 10 --x0_rec_weight 100 --grad_accum_steps 1 --diversity_loss_weight 0 --eval_every_e 10 --max_eval_samples 512 --latest_every_e 10 --save_every_e 50 --no_fgd --gpu_id 0 --beat_cache_name beat_4english_15_141_sync_v1 --seed 1234 --workers 32 --persistent_workers True --prefetch_factor 2 --non_blocking_transfer True > logs/train_beat_FM_aa_x0_aligned_sync_v1.log 2>&1 &
    
    echo "部署完成！可以登录服务器使用 'tail -f {REMOTE_DIR}/logs/train.log' 查看训练进度。"
    """
    print("注意：只有当上面【阶段3】的文件传输全部完成后，才可以执行此操作。")
    print("你可以通过这行代码直接执行训练： ssh_run(run_cmd)")

if __name__ == "__main__":
    print("========= DiffSHEG 云端自动部署 Workflow =========")
    step_1_prepare_server()
    step_2_setup_env()
    step_3_transfer_files()
    step_4_start_training()
