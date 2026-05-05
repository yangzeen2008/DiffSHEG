"""
Generate full-length BVH animations for each model using test_custom_audio mode.
Uses a real audio file from BEAT dataset.
"""
import subprocess, os, sys, glob, shutil

os.chdir(r'f:\study\DiffSHEG')
python = r'f:\study\DiffSHEG\.venv\Scripts\python.exe'

# Use a ~10-second audio clip
test_audio = r'f:\study\DiffSHEG\data\BEAT\raw\beat_english_v0.2.1\beat_english_v0.2.1\1\1_wayne_0_100_100.wav'

configs = [
    {
        "name": "beat_FM_v1",
        "label": "FM_AxisAngle",
        "ckpt": "pck_best.tar",
        "extra": "--flow_matching --fm_sample_steps 50 --overlap_len 4",
    },
    {
        "name": "beat_DDPM_v1",
        "label": "DDPM_AxisAngle",
        "ckpt": "pck_best.tar",
        "extra": "--ddim --timestep_respacing ddim25 --overlap_len 4",
    },
    {
        "name": "beat_FM_6D_v2",
        "label": "FM_6D_v2",
        "ckpt": "pck_best.tar",
        "extra": "--flow_matching --fm_sample_steps 50 --rot_6d --hidden_size_override 512 --n_layer_override 8 --overlap_len 4",
    },
]

os.makedirs('bvh_output', exist_ok=True)

for cfg in configs:
    print(f"\n{'='*60}")
    print(f"[{cfg['label']}] Generating full-length animation...")
    
    cmd = (
        f'{python} runner.py '
        f'--dataset_name beat --name {cfg["name"]} '
        f'--mode test_custom_audio '
        f'--test_audio_path "{test_audio}" '
        f'--n_poses 34 --batch_size 1 --gpu_id 0 '
        f'--beat_cache_name beat_4english_15_141 --workers 0 '
        f'--resume --ckpt {cfg["ckpt"]} --no_fgd '
        f'{cfg["extra"]}'
    )
    
    print(f"  CMD: {cmd[:200]}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
    else:
        print(f"  STDOUT: {result.stdout[-300:]}")
    
    # Find BVH
    bvh_files = glob.glob(f'results/**/bvh/*.bvh', recursive=True)
    new_bvhs = [b for b in bvh_files if '1_wayne_0_100_100' in b]
    if new_bvhs:
        src = sorted(new_bvhs)[-1]
        dst = f'bvh_output/{cfg["label"]}.bvh'
        shutil.copy2(src, dst)
        size = os.path.getsize(dst) / 1024
        print(f"  OK: {dst} ({size:.1f} KB)")

print(f"\n{'='*60}")
print("All BVH:")
for f in sorted(os.listdir('bvh_output')):
    if f.endswith('.bvh'):
        print(f"  {f}: {os.path.getsize(os.path.join('bvh_output', f))/1024:.1f} KB")
