import paramiko, os, numpy as np, sys
sys.path.insert(0, r'f:\study\DiffSHEG')
from datasets.data_tools import joints_list

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', 48360, 'root', 'Yangzeen2008#')

# Download pid_2 gesture (normalized euler)
sftp = ssh.open_sftp()
remote = '/root/autodl-tmp/DiffSHEG/results/beat_34/test_custom_audio/beat_FM_v2/fixStart4/BestPCK_e899/pid_2/gesture/1_wayne_0_100_100.npy'
local_dir = r'f:\study\DiffSHEG\results\beat_34\test_custom_audio\beat_FM_v2\fixStart4\BestPCK_e899\pid_2\gesture'
os.makedirs(local_dir, exist_ok=True)
local = os.path.join(local_dir, '1_wayne_0_100_100.npy')
sftp.get(remote, local)
sftp.close()
ssh.close()
print(f'Downloaded: {local}')

# Convert to BVH
template_path = r'f:\study\DiffSHEG\data\BEAT\raw\beat_english_v0.2.1\beat_english_v0.2.1\1\1_wayne_0_100_100.bvh'
ori_list = joints_list["beat_joints"]
target_list = joints_list["spine_neck_141"]

with open(template_path, 'r') as f:
    lines = f.readlines()
header = lines[:431]
offset_data = np.fromstring(lines[431], dtype=float, sep=' ')
mean_euler = np.load(r'f:\study\DiffSHEG\data\BEAT\beat_cache\beat_4english_15_141\train\bvh_rot\bvh_mean.npy')
std_euler = np.load(r'f:\study\DiffSHEG\data\BEAT\beat_cache\beat_4english_15_141\train\bvh_rot\bvh_std.npy')

data = np.load(local)
if data.ndim == 3: data = data[0]
euler_deg = data * std_euler + mean_euler
T = euler_deg.shape[0]
print(f'Frames: {T}, shape: {data.shape}')

out_path = os.path.join('bvh_output', 'FM_v2.bvh')
header_copy = header.copy()
header_copy[429] = f'Frames: {T}\n'
with open(out_path, 'w') as f:
    f.writelines(header_copy)
    for i in range(T):
        if i == 0: continue
        rot = offset_data.copy()
        for iii, (k, v) in enumerate(target_list.items()):
            if iii * 3 + 3 <= euler_deg.shape[1]:
                rot[ori_list[k][1]-v:ori_list[k][1]] = euler_deg[i, iii*3:iii*3+3]
        f.write(' '.join(f'{x:.6f}' for x in rot) + '\n')
print(f'Saved: {out_path}')

# Compare arm ranges
arm_joints = {
    'RShoulder': (30, 33), 'RArm(大臂)': (33, 36), 'RArm1(小臂)': (36, 39),
    'LShoulder': (111, 114), 'LArm(大臂)': (114, 117), 'LArm1(小臂)': (117, 120),
}
files = {'Original': 'bvh_output/Original_DiffSHEG.bvh', 'FM_v1(旧)': 'bvh_output/FM_AxisAngle.bvh', 'FM_v2(新)': out_path}

print('\n' + '='*70)
print(f'{"Joint":15s} | {"Original":>10s} | {"FM_v1(旧)":>10s} | {"FM_v2(新)":>10s} | v2/Orig')
print('-'*70)
for jname, (s, e) in arm_joints.items():
    vals = {}
    for label, path in files.items():
        if not os.path.exists(path): continue
        with open(path, 'r') as f:
            bl = f.readlines()
        bd = np.array([np.fromstring(l, dtype=float, sep=' ') for l in bl[431:] if len(l.strip()) > 5])
        vals[label] = (bd[:, s:e].max(axis=0) - bd[:, s:e].min(axis=0)).mean()
    orig = vals.get('Original', 1)
    ratio = vals.get('FM_v2(新)', 0) / orig * 100 if orig > 0 else 0
    print(f'{jname:15s} | {vals.get("Original",0):>9.1f}° | {vals.get("FM_v1(旧)",0):>9.1f}° | {vals.get("FM_v2(新)",0):>9.1f}° | {ratio:>5.1f}%')
