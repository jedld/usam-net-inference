import os
import shutil
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Copy a limited number of files from source to destination directory')
    parser.add_argument('--src_root', required=True, help='Source root directory path')
    parser.add_argument('--dst_root', default='data', help='Destination root directory path (default: ./data)')
    parser.add_argument('--N', type=int, default=10, help='Maximum number of files to copy per directory (default: 10)')
    return parser.parse_args()

def main():
    args = parse_args()
    total_files_copied = 0
    total_errors = 0
    error_files = []
    
    # Ensure destination directory exists
    os.makedirs(args.dst_root, exist_ok=True)
    
    print(f"\nStarting file copy process:")
    print(f"Source: {args.src_root}")
    print(f"Destination: {args.dst_root}")
    print(f"Max files per directory: {args.N}\n")
    
    for root, dirs, files in os.walk(args.src_root):
        # Sort files and select the first N
        files = sorted([f for f in files if os.path.isfile(os.path.join(root, f))])[:args.N]
        
        # Determine the relative path from the source root
        rel_path = os.path.relpath(root, args.src_root)
        dst_dir = os.path.join(args.dst_root, rel_path)
        
        # Create destination directory if it doesn't exist
        os.makedirs(dst_dir, exist_ok=True)
        
        dir_files_copied = 0
        for f in files:
            src_file = os.path.join(root, f)
            dst_file = os.path.join(dst_dir, f)
            try:
                shutil.copy2(src_file, dst_file)
                dir_files_copied += 1
                total_files_copied += 1
                print(f"✓ Copied: {src_file} -> {dst_file}")
            except Exception as e:
                total_errors += 1
                error_files.append((src_file, str(e)))
                print(f"✗ Error copying {src_file}: {str(e)}")
        
        if dir_files_copied > 0:
            print(f"  → Copied {dir_files_copied} files from {rel_path}\n")
    
    # Print summary
    print("\nCopy process completed!")
    print(f"Total files copied: {total_files_copied}")
    if total_errors > 0:
        print(f"Total errors: {total_errors}")
        print("\nError details:")
        for src_file, error in error_files:
            print(f"- {src_file}: {error}")
    else:
        print("No errors encountered.")

if __name__ == '__main__':
    main()
