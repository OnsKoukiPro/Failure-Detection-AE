import pandas as pd
import os
import re
from datetime import datetime
from collections import defaultdict

# Docker paths (mounted volumes)
CSV_PATH = "/app/data/overview.csv"
WAV_ROOT_DIR = "/app/data/raw"  # Adjusted to raw subdirectory
OUTPUT_PATH = "/app/results/mapped_dataset.csv"

# --- Step 1: Parse CSV Metadata ---
df = pd.read_csv(CSV_PATH)

# Convert timestamps to datetime (match WAV filenames)
df["Time stamp"] = pd.to_datetime(
    df["Time stamp"],
    format="%d.%m.%Y %H:%M:%S.%f"  # Handles CSV's format
)
# Format to "dd.mm.yyyy_HH:MM:SS" for matching
df["timestamp_str"] = df["Time stamp"].dt.strftime("%d.%m.%Y_%H:%M:%S")

# --- Step 2: Traverse WAV Files & Extract Metadata ---
wav_files = defaultdict(list)

for root, _, files in os.walk(WAV_ROOT_DIR):
    for file in files:
        if file.endswith(".wav"):
            # Extract timestamp from filename using regex
            timestamp_match = re.search(
                r"(\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}\.\d{3})", file
            )
            if not timestamp_match:
                print(f"Skipping {file}: No timestamp found")
                continue

            # Parse timestamp from filename (e.g., "2018-12-06 15-40-50.000")
            timestamp_str_file = timestamp_match.group(1)
            try:
                dt = datetime.strptime(
                    timestamp_str_file,
                    "%Y-%m-%d %H-%M-%S.%f"
                )
            except ValueError:
                print(f"Skipping {file}: Invalid timestamp format")
                continue

            # Convert to CSV's timestamp format ("dd.mm.yyyy_HH:MM:SS")
            timestamp_str = dt.strftime("%d.%m.%Y_%H:%M:%S")

            # Extract measurement, channel, and quality from directory structure
            path_parts = root.split(os.sep)
            # Example path: /app/data/raw/Measurement-1-Channel-1/Frequency not good/
            measurement_channel = path_parts[-2]  # "Measurement-1-Channel-1"
            quality_folder = path_parts[-1]       # "Frequency not good"

            # Split measurement and channel
            if "-Channel-" not in measurement_channel:
                print(f"Skipping {file}: Invalid measurement_channel format")
                continue
            measurement, channel = measurement_channel.rsplit("-Channel-", 1)

            # Store metadata
            wav_files[timestamp_str].append({
                "path": os.path.join(root, file),
                "measurement": measurement,
                "channel": channel,
                "quality_folder": quality_folder
            })

# --- Step 3: Map WAV Files to CSV Rows ---
mapped_data = []

for timestamp_str, entries in wav_files.items():
    # Find matching CSV row(s)
    matches = df[df["timestamp_str"] == timestamp_str]

    if not matches.empty:
        for entry in entries:
            mapped_data.append({
                "wav_path": entry["path"],
                "measurement": entry["measurement"],
                "channel": entry["channel"],
                "quality_folder": entry["quality_folder"],
                "timestamp": matches["Time stamp"].iloc[0],
                "csv_label": matches["INFO_2"].iloc[0],
                "info_1": matches["INFO_1"].iloc[0]
            })
    else:
        print(f"WARNING: No CSV match for timestamp {timestamp_str}")

# Save mapped data
pd.DataFrame(mapped_data).to_csv(OUTPUT_PATH, index=False)
print(f"Mapped data saved to {OUTPUT_PATH}")