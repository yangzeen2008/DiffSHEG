import os
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

def parse_bvh_header(lines):
    """
    Parses the BVH header to identify the joint channel mappings.
    Returns:
        header_lines: List of strings representing the hierarchy header.
        joints: List of dicts, each with 'name', 'start_idx', and 'num_channels'.
        data_start_line: The line number where the motion data floats begin.
    """
    joints = []
    current_joint = None
    channel_count = 0
    data_start_line = 0
    header_lines = []

    for i, line in enumerate(lines):
        line_strip = line.strip()
        header_lines.append(line)
        
        if line_strip.startswith('ROOT') or line_strip.startswith('JOINT'):
            current_joint = line_strip.split()[1]
        elif line_strip.startswith('CHANNELS'):
            parts = line_strip.split()
            num_channels = int(parts[1])
            joints.append({
                'name': current_joint,
                'start_idx': channel_count,
                'num_channels': num_channels,
                'channels': parts[2:2 + num_channels],
            })
            channel_count += num_channels
        elif line_strip.startswith('MOTION'):
            # The next two lines are Frames: and Frame Time:
            header_lines.append(lines[i+1])
            header_lines.append(lines[i+2])
            data_start_line = i + 3
            break

    return header_lines, joints, data_start_line

def _unwrap_degrees(values):
    """Choose temporally continuous Euler branches after conversion."""
    return np.rad2deg(np.unwrap(np.deg2rad(values), axis=0))


def smooth_euler_rotation(values, order, sigma, mode='quaternion'):
    """Smooth Euler rotations without averaging across the +/-180 seam."""
    if sigma <= 0:
        return values.copy()

    if mode == 'euler':
        continuous = _unwrap_degrees(values)
        return gaussian_filter1d(continuous, sigma=sigma, axis=0, mode='nearest')

    # Convert to unit quaternions, make quaternion signs temporally
    # continuous, filter in R4, then project back to the unit sphere.
    quaternions = Rotation.from_euler(order, values, degrees=True).as_quat()
    for frame in range(1, len(quaternions)):
        if np.dot(quaternions[frame - 1], quaternions[frame]) < 0:
            quaternions[frame] *= -1.0

    filtered = gaussian_filter1d(quaternions, sigma=sigma, axis=0, mode='nearest')
    norms = np.linalg.norm(filtered, axis=-1, keepdims=True)
    filtered = filtered / np.maximum(norms, 1e-12)
    euler = Rotation.from_quat(filtered).as_euler(order, degrees=True)
    return _unwrap_degrees(euler)


def _smooth_joint_rotation(smoothed_data, motion_data, joint, sigma, rotation_mode):
    channels = joint.get('channels', [])
    start = joint['start_idx']
    rotation_offsets = [
        offset for offset, channel in enumerate(channels)
        if channel.lower().endswith('rotation')
    ]
    if len(rotation_offsets) != 3:
        return

    rotation_indices = [start + offset for offset in rotation_offsets]
    order = ''.join(channels[offset][0].upper() for offset in rotation_offsets)
    smoothed_data[:, rotation_indices] = smooth_euler_rotation(
        motion_data[:, rotation_indices], order, sigma, mode=rotation_mode
    )


def apply_smoothing(motion_data, joints, args):
    """
    Applies Gaussian filtering to different joints based on joint name matching.
    """
    smoothed_data = motion_data.copy()
    num_frames = motion_data.shape[0]
    
    print(f"\nApplying adaptive smoothing to {num_frames} frames...")
    
    for j in joints:
        name = j['name'].lower()
        start = j['start_idx']
        num_ch = j['num_channels']
        
        # Decide sigma based on joint name
        if name == 'hips':
            # Smooth translations linearly and rotations on SO(3).
            channels = j.get('channels', [])
            translation_offsets = [
                offset for offset, channel in enumerate(channels)
                if channel.lower().endswith('position')
            ]
            if translation_offsets:
                if args.smooth_hips_trans > 0:
                    indices = [start + offset for offset in translation_offsets]
                    smoothed_data[:, indices] = gaussian_filter1d(
                        motion_data[:, indices], sigma=args.smooth_hips_trans,
                        axis=0, mode='nearest'
                    )
            _smooth_joint_rotation(
                smoothed_data, motion_data, j, args.sigma_spine, args.rotation_mode
            )
            print(f"  {j['name']:18s} (ROOT) -> Trans sigma={args.smooth_hips_trans}, Rot sigma={args.sigma_spine} ({args.rotation_mode})")

        elif any(x in name for x in ['spine', 'neck', 'chest', 'head']):
            # Spine / Neck joints
            _smooth_joint_rotation(
                smoothed_data, motion_data, j, args.sigma_spine, args.rotation_mode
            )
            print(f"  {j['name']:18s} (Spine/Neck) -> sigma={args.sigma_spine} ({args.rotation_mode})")

        elif any(x in name for x in ['shoulder', 'arm', 'forearm', 'hand', 'finger', 'thumb', 'index', 'middle', 'ring', 'pinky']):
            # Arm / Shoulder / Hand / Fingers
            _smooth_joint_rotation(
                smoothed_data, motion_data, j, args.sigma_arm, args.rotation_mode
            )
            print(f"  {j['name']:18s} (Arm/Hand/Finger) -> sigma={args.sigma_arm} ({args.rotation_mode})")

        else:
            # Other joints (e.g. legs, if present)
            if args.sigma_other > 0:
                _smooth_joint_rotation(
                    smoothed_data, motion_data, j, args.sigma_other, args.rotation_mode
                )
                print(f"  {j['name']:18s} (Other) -> sigma={args.sigma_other} ({args.rotation_mode})")
            else:
                print(f"  {j['name']:18s} (Other) -> Left raw (sigma=0)")
                
    return smoothed_data

def main():
    parser = argparse.ArgumentParser(description="Adaptive Gaussian Smoothing for BVH Animation Files")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input BVH file path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output BVH file path (defaults to input_smooth.bvh)")
    parser.add_argument("--sigma_spine", type=float, default=1.2, help="Smoothing sigma for Spine, Neck, Head (default: 1.2)")
    parser.add_argument("--sigma_arm", type=float, default=1.5, help="Smoothing sigma for Arm, Shoulder, Hand, Fingers (default: 1.5)")
    parser.add_argument("--smooth_hips_trans", type=float, default=0.5, help="Smoothing sigma for Hips Translation (default: 0.5)")
    parser.add_argument("--sigma_other", type=float, default=0.0, help="Smoothing sigma for other joints like legs (default: 0.0)")
    parser.add_argument(
        "--rotation_mode", choices=["quaternion", "euler"], default="quaternion",
        help="Rotation smoothing domain; quaternion avoids +/-180 degree averaging",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return
        
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_smooth{ext}"
        
    print(f"Reading BVH: {args.input}")
    with open(args.input, 'r') as f:
        lines = f.readlines()
        
    # 1. Parse header and joint channels
    header_lines, joints, data_start_line = parse_bvh_header(lines)
    
    # 2. Extract motion data floats
    motion_lines = lines[data_start_line:]
    motion_data = []
    for line in motion_lines:
        if line.strip():
            motion_data.append(np.fromstring(line, dtype=float, sep=' '))
    motion_data = np.array(motion_data)
    
    # 3. Apply smoothing
    smoothed_data = apply_smoothing(motion_data, joints, args)
    
    # 4. Save to output BVH
    print(f"\nSaving smoothed BVH: {args.output}")
    with open(args.output, 'w') as f:
        f.writelines(header_lines)
        for frame in smoothed_data:
            f.write(' '.join(f"{x:.6f}" for x in frame) + '\n')
            
    print("Post-processing complete! [OK]")

if __name__ == '__main__':
    main()
