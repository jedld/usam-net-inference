#!/usr/bin/env bash

# Usage: ./run_nsight_profile.sh [stereo_cli.py args]
# Example: ./run_nsight_profile.sh --model-type base --left left.png --right right.png

# Output file for Nsight Compute report
timestamp=$(date +%Y%m%d_%H%M%S)
REPORT="ncu_report_$timestamp.ncu-rep"

# Path to stereo_cli.py (edit if needed)
SCRIPT="stereo_cli.py"

# Run Nsight Compute profiler
ncu --set full --export "$REPORT" --force-overwrite --target-processes all \
    python3 "$SCRIPT" "$@"

echo "\nNsight Compute profiling complete. Report saved to $REPORT" 