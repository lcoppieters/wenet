#!/usr/bin/env python3

import os
import shutil
import time
import sys

# Define the source wav.scp file and the destination directory
wav_scp_path = sys.argv[1]
tmp_dir = "/tmp"

# Create the destination directory if it doesn't exist
os.makedirs(tmp_dir, exist_ok=True)

# Define the path for the new wav.scp file
new_wav_scp_path = os.path.join(tmp_dir, "wav.scp")
# Start the timer
start_time = time.time()
# Open the original and new wav.scp files
with open(wav_scp_path, "r") as wav_scp_file, open(new_wav_scp_path,
                                                   "w") as new_wav_scp_file:
    # Iterate over each line in the original wav.scp file
    for line in wav_scp_file:

        key, original_path = line.strip().split()
        # Define the new path in the /tmp directory
        new_path = os.path.join(
            tmp_dir, os.path.basename(os.path.dirname(original_path)),
            os.path.basename(original_path))
        # Copy the .wav file to the new path
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.copy(original_path, new_path)
        # Write the key and new path to the new wav.scp file
        new_wav_scp_file.write(f"{key} {new_path}\n")
# Stop the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(
    f"All files have been copied to {tmp_dir} and the new wav.scp file is located at {new_wav_scp_path}."
)
print(f"Time taken: {elapsed_time:.2f} seconds.")
