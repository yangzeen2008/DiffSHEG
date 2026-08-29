import os
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d

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
                'num_channels': num_channels
            })
            channel_count += num_channels
        elif line_strip.startswith('MOTION'):
            # The next two lines are Frames: and Frame Time:
            header_lines.append(lines[i+1])
            header_lines.append(lines[i+2])
            data_start_line = i + 3
            break

    return header_lines, joints, data_start_line

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
            # Hips translation (first 3 channels) is kept raw or smoothed very lightly
            if num_ch >= 6:
                # Translation (channels 0, 1, 2)
                if args.smooth_hips_trans > 0:
                    smoothed_data[:, start:start+3] = gaussian_filter1d(
                        motion_data[:, start:start+3], sigma=args.smooth_hips_trans, axis=0
                    )
                # Rotation (channels 3, 4, 5)
                smoothed_data[:, start+3:start+6] = gaussian_filter1d(
                    motion_data[:, start+3:start+6], sigma=args.sigma_spine, axis=0
                )
                print(f"  {j['name']:18s} (ROOT) -> Trans sigma={args.smooth_hips_trans}, Rot sigma={args.sigma_spine}")
            else:
                smoothed_data[:, start:start+num_ch] = gaussian_filter1d(
                    motion_data[:, start:start+num_ch], sigma=args.sigma_spine, axis=0
                )
                print(f"  {j['name']:18s} -> Rot sigma={args.sigma_spine}")
                
        elif any(x in name for x in ['spine', 'neck', 'chest', 'head']):
            # Spine / Neck joints
            smoothed_data[:, start:start+num_ch] = gaussian_filter1d(
                motion_data[:, start:start+num_ch], sigma=args.sigma_spine, axis=0
            )
            print(f"  {j['name']:18s} (Spine/Neck) -> sigma={args.sigma_spine}")
            
        elif any(x in name for x in ['shoulder', 'arm', 'forearm', 'hand', 'finger', 'thumb', 'index', 'middle', 'ring', 'pinky']):
            # Arm / Shoulder / Hand / Fingers
            smoothed_data[:, start:start+num_ch] = gaussian_filter1d(
                motion_data[:, start:start+num_ch], sigma=args.sigma_arm, axis=0
            )
            print(f"  {j['name']:18s} (Arm/Hand/Finger) -> sigma={args.sigma_arm}")
            
        else:
            # Other joints (e.g. legs, if present)
            if args.sigma_other > 0:
                smoothed_data[:, start:start+num_ch] = gaussian_filter1d(
                    motion_data[:, start:start+num_ch], sigma=args.sigma_other, axis=0
                )
                print(f"  {j['name']:18s} (Other) -> sigma={args.sigma_other}")
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
