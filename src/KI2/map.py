import pandas as pd
import numpy as np
from pathlib import Path
import re

def create_metadata_dataframe(lying_directory, hanging_directory):
    """
    Create a metadata DataFrame from WAV files in the specified directories.

    Parameters:
    lying_directory (str): Path to the directory containing lying position WAV files
    hanging_directory (str): Path to the directory containing hanging position WAV files

    Returns:
    pd.DataFrame: DataFrame containing metadata extracted from filenames
    """
    data = []

    # Process lying position files
    lying_paths = list(Path(lying_directory).glob('*.wav'))
    for file_path in lying_paths:
        process_file(file_path, 'lying', data)

    # Process hanging position files
    hanging_paths = list(Path(hanging_directory).glob('*.wav'))
    for file_path in hanging_paths:
        process_file(file_path, 'hanging', data)

    if not data:
        raise ValueError(f"No WAV files found in the specified directories")

    # Create DataFrame
    df_meta = pd.DataFrame(data)

    # Sort by part_id and timestamp
    df_meta = df_meta.sort_values(by=['part_id', 'timestamp']).reset_index(drop=True)

    return df_meta

def process_file(file_path, condition, data_list):
    """
    Process a single file and add its metadata to the data list.
    """
    filename = file_path.stem  # gets '01_A_2025-08-05 15-23-13.000'

    # Use regex to properly split the filename
    # Updated pattern to handle the space between date and time
    pattern = r'^(\d+)_([A-Za-z]+)_(\d{4}-\d{2}-\d{2}) (\d{2}-\d{2}-\d{2}\.\d{3})$'
    match = re.match(pattern, filename)

    if not match:
        print(f"Skipping file with unexpected naming pattern: {filename}")
        return

    part_id = match.group(1)
    position = match.group(2)
    date_str = match.group(3)
    time_str = match.group(4)

    # Combine date and time
    dt_str = f"{date_str} {time_str}"

    # Convert to datetime object
    try:
        timestamp = pd.to_datetime(dt_str, format='%Y-%m-%d %H-%M-%S.%f')
    except ValueError:
        print(f"Could not parse datetime for file: {filename}")
        return

    data_list.append({
        'file_path': str(file_path),
        'filename': filename,
        'part_id': part_id,
        'position': position,
        'timestamp': timestamp,
        'condition': condition
    })

def analyze_metadata(df_meta):
    """
    Perform basic analysis of the metadata.
    """
    print("=== Metadata Overview ===")
    print(f"Total files: {len(df_meta)}")
    print(f"Date range: {df_meta['timestamp'].min()} to {df_meta['timestamp'].max()}")

    print("\n=== Parts Summary ===")
    part_counts = df_meta['part_id'].value_counts()
    print(f"Number of unique parts: {len(part_counts)}")
    print("Files per part:")
    print(part_counts)

    print("\n=== Positions Summary ===")
    position_counts = df_meta['position'].value_counts()
    print("Files per position:")
    print(position_counts)

    print("\n=== Conditions Summary ===")
    condition_counts = df_meta['condition'].value_counts()
    print("Files per condition:")
    print(condition_counts)

    # Check for parts with both conditions
    parts_with_both_conditions = df_meta.groupby('part_id')['condition'].nunique()
    parts_with_both = parts_with_both_conditions[parts_with_both_conditions > 1]
    if not parts_with_both.empty:
        print("\nParts with both lying and hanging measurements:")
        print(parts_with_both)
    else:
        print("\nNo parts have both lying and hanging measurements.")

def save_metadata(df_meta, output_path):
    """
    Save the metadata DataFrame to a CSV file.
    """
    df_meta.to_csv(output_path, index=False)
    print(f"Metadata saved to: {output_path}")

# Main execution
if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    lying_directory = "/app/data/lying position"  # Update this path
    hanging_directory = "/app/data/hanging position"  # Update this path
    output_csv_path = "/app/results/KI2/metadata.csv"

    try:
        # Create metadata DataFrame
        print("Processing WAV files...")
        df_meta = create_metadata_dataframe(lying_directory, hanging_directory)

        # Display basic information
        print("\nFirst few rows of metadata:")
        print(df_meta.head())

        # Analyze the metadata
        analyze_metadata(df_meta)

        # Save to CSV
        save_metadata(df_meta, output_csv_path)

        print("\nMetadata extraction completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")