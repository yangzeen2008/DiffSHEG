"""
DiffSHEG 论文图表生成
生成到 F:\study\DiffSHEG\article\figures\
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.linewidth'] = 1.2

out_dir = r'F:\study\DiffSHEG\article\figures'
os.makedirs(out_dir, exist_ok=True)

# Color palette
C_DDPM = '#E74C3C'      # red
C_FM50 = '#2E86C1'      # blue
C_FM10 = '#27AE60'      # green
C_FM1  = '#F39C12'      # orange
C_FM1S = '#8E44AD'      # purple

# ============================================================
# Figure 1: Step Ablation - FGD vs Steps
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
fgd =   [59.2, 71.6, 68.3, 66.1, 58.8, 75.8, 42.6, 60.1, 68.2, 81.7, 44.4]
n_samples = [187, 183, 169, 166, 73, 67, 71, 58, 54, 51, 51]

# Error bars based on std/sqrt(n), approximate std=33
se = [33/np.sqrt(n) for n in n_samples]

ax.errorbar(steps, fgd, yerr=se, fmt='o-', color=C_FM50, linewidth=2, 
            markersize=8, capsize=4, label='FM_v6 (RK4)', zorder=3)

# DDPM reference line
ax.axhline(y=379.3, color=C_DDPM, linestyle='--', linewidth=1.5, alpha=0.7, label='DDPM (379.3)')

# Shade the FM range
ax.fill_between(steps, 40, 85, alpha=0.08, color=C_FM50)
ax.annotate('FM range: 42-82', xy=(25, 82), fontsize=10, color=C_FM50, alpha=0.6)

ax.set_xlabel('Number of Sampling Steps', fontsize=13)
ax.set_ylabel('FGD ↓', fontsize=13)
ax.set_title('Flow Matching: FGD vs Sampling Steps', fontsize=14, fontweight='bold')
ax.set_xlim(0, 52)
ax.set_ylim(30, 100)
ax.set_xticks(steps)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'step_ablation_fgd.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'step_ablation_fgd.pdf'), bbox_inches='tight')
plt.close()
print("1. Step ablation saved")

# ============================================================
# Figure 2a: Training Time Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(5, 4.5))

train_methods = ['DDPM', 'FM_v6']
train_hours = [63.4, 15.0]
train_colors = [C_DDPM, C_FM50]

bars = ax.bar(train_methods, train_hours, color=train_colors, alpha=0.85, 
              edgecolor='white', linewidth=2, width=0.5)
for i, v in enumerate(train_hours):
    ax.text(i, v + 1.5, f'{v}h', ha='center', fontsize=14, fontweight='bold', color=train_colors[i])

ax.set_ylabel('Training Time (hours) ↓', fontsize=13)
ax.set_title('Training Speed Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(0, 80)
ax.grid(True, axis='y', alpha=0.3)
ax.text(0.5, 40, '4.2× faster', ha='center', fontsize=14, color=C_FM50,
        fontweight='bold', style='italic')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'training_speed.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'training_speed.pdf'), bbox_inches='tight')
plt.close()
print("2a. Training speed saved")

# ============================================================
# Figure 2b: Inference Speed vs Quality
# ============================================================
fig, ax1 = plt.subplots(figsize=(7, 5))

methods = ['DDPM\n1000 steps', 'FM\n50 steps', 'FM\n10 steps', 'FM\n1 step']
fps_vals = [16, 107, 267, 533]
fgd_vals = [379.3, 36.5, 68.3, 59.2]
colors = [C_DDPM, C_FM50, C_FM10, C_FM1]

x = np.arange(len(methods))
width = 0.6

bars1 = ax1.bar(x, fps_vals, width, color=colors, alpha=0.85, 
                label='FPS ↑', edgecolor='white', linewidth=1.5)
ax1.set_ylabel('FPS ↑ (Frames Per Second)', fontsize=12, color='#333')
ax1.set_ylim(0, 750)

ax2 = ax1.twinx()
ax2.plot(x, fgd_vals, 'D-', color='#333', markersize=10, linewidth=2, label='FGD ↓', zorder=5)
ax2.set_ylabel('FGD ↓', fontsize=12, color='#333')
ax2.set_ylim(0, 520)

for i, (f, g) in enumerate(zip(fps_vals, fgd_vals)):
    ax1.text(x[i], f + 15, f'{f}', ha='center', fontsize=11, fontweight='bold', color=colors[i])
    ax2.text(x[i] + 0.15, g + 12, f'{g:.1f}', ha='left', fontsize=10, color='#333')

ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=11)
ax1.set_title('Inference Speed vs Quality', fontsize=14, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'inference_speed.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'inference_speed.pdf'), bbox_inches='tight')
plt.close()
print("2b. Inference speed saved")

# ============================================================
# Figure 3: 1-Step Post-processing Comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

versions = ['Original\n(DDPM)', '1-Step\nraw', '1-Step\nσ=1.5', '1-Step\nσ=3.0', '1-Step\nscaled']
colors_pp = [C_DDPM, '#95A5A6', C_FM10, C_FM1S, C_FM50]

# Data
arm_vel =   [11.6, 36.9, 7.1, 3.4, 6.7]
arm_std =   [37.7, 45.6, 38.1, 36.1, 37.7]
arm_range = [96.4, 92.3, 71.5, 61.1, 87.8]

for ax, data, title, ylabel in zip(axes, 
    [arm_vel, arm_std, arm_range],
    ['Arm Velocity Std', 'Arm Position Std', 'Arm Motion Range'],
    ['Vel std ↓', 'Arm std', 'Range (deg)']):
    
    bars = ax.bar(range(len(versions)), data, color=colors_pp, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(data) * 1.18)
    
    # Add value labels
    for i, v in enumerate(data):
        ax.text(i, v + max(data)*0.02, f'{v}', ha='center', fontsize=9, fontweight='bold')
    
    # Reference line for Original
    ax.axhline(y=data[0], color=C_DDPM, linestyle='--', linewidth=1, alpha=0.4)

plt.suptitle('1-Step FM: Post-processing Methods Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'one_step_postprocess.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'one_step_postprocess.pdf'), bbox_inches='tight')
plt.close()
print("3. 1-Step post-processing saved")

# ============================================================
# Figure 4: Main FGD Comparison (bar chart)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

methods = ['DDPM', 'FM_v5\n(Euler)', 'FM_v6\n50-step', 'FM_v6\n1-step']
fgd_mean = [379.3, 57.0, 36.5, 59.2]
colors_fgd = [C_DDPM, '#95A5A6', C_FM50, C_FM1]

bars = ax.barh(range(len(methods)), fgd_mean, color=colors_fgd, alpha=0.85, 
               edgecolor='white', linewidth=2, height=0.6)

for i, v in enumerate(fgd_mean):
    ax.text(v + 5, i, f'{v:.1f}', va='center', fontsize=12, fontweight='bold', color=colors_fgd[i])

ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=12)
ax.set_xlabel('FGD ↓ (lower is better)', fontsize=13)
ax.set_title('FGD Comparison: Same Evaluator', fontsize=14, fontweight='bold')
ax.set_xlim(0, 450)
ax.invert_yaxis()
ax.grid(True, axis='x', alpha=0.3)

# Per-bar "X× lower" annotations (compared to DDPM)
ddpm_fgd = 379.3
for i, v in enumerate(fgd_mean):
    if i == 0:  # DDPM baseline, no annotation
        continue
    ratio = ddpm_fgd / v
    ax.text(v + 60, i, f'{ratio:.1f}× lower', fontsize=11, va='center', 
            color=colors_fgd[i], fontweight='bold', style='italic')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fgd_comparison.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'fgd_comparison.pdf'), bbox_inches='tight')
plt.close()
print("4. FGD comparison saved")

print(f"\nAll figures saved to {out_dir}")
