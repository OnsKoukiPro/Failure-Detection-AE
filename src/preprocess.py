import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import pandas as pd

# Configuration
INPUT_CSV = "/app/results/mapped_dataset.csv"
OUTPUT_DIR = "/app/results/processed_data"
FILTER_LOW = 1000    # 1 kHz
FILTER_HIGH = 50000  # 50 kHz (102.4 kHz sampling rate)
SILENCE_THRESHOLD = 0.01

def normalize_signal(signal):
    signal = signal.astype(np.float32)
    return signal / np.max(np.abs(signal))

def bandpass_filter(signal, sample_rate, lowcut, highcut):
    nyquist = 0.5 * sample_rate
    highcut = min(highcut, 0.95 * nyquist)
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)

def trim_silence(signal, threshold=SILENCE_THRESHOLD):
    abs_signal = np.abs(signal)
    threshold_value = threshold * np.max(abs_signal)
    start = np.argmax(abs_signal > threshold_value)
    end = len(abs_signal) - np.argmax(abs_signal[::-1] > threshold_value)
    return signal[start:end]

def process_file(wav_path, output_dir):
    sample_rate, signal = wavfile.read(wav_path)
    assert sample_rate == 102400, f"Unexpected sampling rate: {sample_rate} Hz"
    if len(signal) != 51200:
        signal = signal[:51200]
    signal = normalize_signal(signal)
    signal = bandpass_filter(signal, sample_rate, FILTER_LOW, FILTER_HIGH)
    signal = trim_silence(signal)
    base_name = os.path.basename(wav_path).replace(".wav", "")
    output_path = os.path.join(output_dir, f"{base_name}.npy")
    np.save(output_path, signal)
    return output_path

def combine_channels(measurement_group, output_dir):
    combined_data = []
    min_length = None

    # Load signals and find minimum length
    for _, row in measurement_group.iterrows():
        signal = np.load(row["processed_path"])
        combined_data.append(signal)
        if min_length is None or len(signal) < min_length:
            min_length = len(signal)

    # Trim all signals to the shortest length
    trimmed_data = [signal[:min_length] for signal in combined_data]

    # Average the trimmed signals
    combined_signal = np.mean(trimmed_data, axis=0)

    # Generate output path
    measurement_id = measurement_group["measurement"].iloc[0]
    timestamp_str = measurement_group["timestamp"].iloc[0].strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{measurement_id}_{timestamp_str}_combined.npy")
    np.save(output_path, combined_signal)
    return output_path

if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_csv(INPUT_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process all files
    processed_paths = []
    for _, row in df.iterrows():
        output_path = process_file(row["wav_path"], OUTPUT_DIR)
        processed_paths.append(output_path)

    # Update DataFrame with processed paths
    df["processed_path"] = processed_paths

    # Filter data and create binary labels
    filtered_df = df[df["csv_label"] != "invalid"].copy()
    filtered_df["label"] = filtered_df["csv_label"].apply(
        lambda x: 0 if x == "good" else 1
    )

    # Combine valid channels
    combined_paths = []
    for (measurement, timestamp), group in filtered_df.groupby(["measurement", "timestamp"]):
        if len(group) >= 2:
            try:
                combined_path = combine_channels(group, OUTPUT_DIR)
                combined_paths.append(combined_path)
            except Exception as e:
                print(f"Failed to combine {measurement} @ {timestamp}: {str(e)}")

    # Save filtered metadata
    filtered_df.to_csv(os.path.join(OUTPUT_DIR, "processed_metadata.csv"), index=False)