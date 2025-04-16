import os
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Copy a limited number of files from source to destination directory')
    parser.add_argument('--src_root', required=True, help='Source root directory path')
    parser.add_argument('--dst_root', default='data', help='Destination root directory path (default: ./data)')
    parser.add_argument('--N', type=int, default=10, help='Maximum number of files to copy per directory (default: 10)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure destination directory exists
    os.makedirs(args.dst_root, exist_ok=True)
    
    for root, dirs, files in os.walk(args.src_root):
        # Sort files and select the first N
        files = sorted([f for f in files if os.path.isfile(os.path.join(root, f))])[:args.N]
        
        # Determine the relative path from the source root
        rel_path = os.path.relpath(root, args.src_root)
        dst_dir = os.path.join(args.dst_root, rel_path)
        
        # Create destination directory if it doesn't exist
        os.makedirs(dst_dir, exist_ok=True)

        for f in files:
            src_file = os.path.join(root, f)
            dst_file = os.path.join(dst_dir, f)
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")

if __name__ == '__main__':
    main()
