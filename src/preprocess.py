import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import pandas as pd

# Configuration
INPUT_CSV = "/app/results/mapped_dataset.csv"
OUTPUT_DIR = "/app/results/processed_data"
FILTER_LOW = 20000    # 20 kHz
FILTER_HIGH = 200000  # 200 kHz (for 500 kHz sampling rate)
SILENCE_THRESHOLD = 0.01

def normalize_signal(signal):
    signal = signal.astype(np.float32)
    return signal / np.max(np.abs(signal))

def bandpass_filter(signal, sample_rate, lowcut, highcut):
    nyquist = 0.5 * sample_rate
    highcut = min(highcut, 0.95 * nyquist)  # 95% safety margin
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
    print(f"Processing {wav_path} | Sampling rate: {sample_rate} Hz")
    signal = normalize_signal(signal)
    signal = bandpass_filter(signal, sample_rate, FILTER_LOW, FILTER_HIGH)
    signal = trim_silence(signal)
    base_name = os.path.basename(wav_path).replace(".wav", "")
    output_path = os.path.join(output_dir, f"{base_name}.npy")
    np.save(output_path, signal)
    return output_path

def combine_channels(measurement_group, output_dir):
    combined_data = []
    for _, row in measurement_group.iterrows():
        signal = np.load(row["processed_path"])
        combined_data.append(signal)
    combined_signal = np.mean(combined_data, axis=0)
    measurement_id = measurement_group["measurement"].iloc[0]
    timestamp_str = measurement_group["timestamp"].iloc[0].strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{measurement_id}_{timestamp_str}_combined.npy")
    np.save(output_path, combined_signal)
    return output_path

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process only "good" data
    filtered_df = df[df["csv_label"] == "good"]
    processed_paths = []
    for _, row in filtered_df.iterrows():
        output_path = process_file(row["wav_path"], OUTPUT_DIR)
        processed_paths.append(output_path)
    df["processed_path"] = processed_paths
    
    # Update DataFrame with processed paths
    filtered_df["processed_path"] = processed_paths

    # Combine channels by measurement and timestamp
    combined_paths = []
    for (measurement, timestamp), group in filtered_df.groupby(["measurement", "timestamp"]):
        if len(group) >= 2:  # At least two channels
            combined_path = combine_channels(group, OUTPUT_DIR)
            combined_paths.append(combined_path)

    # Save filtered metadata
    filtered_df.to_csv(os.path.join(OUTPUT_DIR, "processed_metadata.csv"), index=False)