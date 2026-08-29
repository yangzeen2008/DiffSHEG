import subprocess
import os

os.chdir(r'f:\study\DiffSHEG')
python = r'f:\study\DiffSHEG\.venv\Scripts\python.exe'

models = [
    "DDPM_v2.bvh",
    "FM_aa_base.bvh",
    "FM_aa_vel.bvh",
    "FM_aa_velAcc.bvh",
    "FM_6d.bvh"
]

bvh_dir = "bvh_output"

print("Starting batch smoothing of all models...")

for model in models:
    input_path = os.path.join(bvh_dir, model)
    output_name = model.replace(".bvh", "_smooth.bvh")
    output_path = os.path.join(bvh_dir, output_name)
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}, skipping")
        continue
        
    cmd = [
        python, "smooth_adaptive.py",
        "-i", input_path,
        "-o", output_path,
        "--sigma_spine", "1.2",
        "--sigma_arm", "1.5",
        "--smooth_hips_trans", "0.5"
    ]
    
    print(f"\nSmoothing {input_path} -> {output_path}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error smoothing {model}: {result.stderr}")
    else:
        print(f"Success! {result.stdout.strip()}")

print("\nBatch smoothing complete!")
