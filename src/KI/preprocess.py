import os
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

INPUT_DIR = "/app/data"
OUTPUT_DIR = "/app/results/KI/processed"
TARGET_SR = 51200                # Resample to 51.2 kHz
BANDPASS_RANGE = (1000, 20000)   # Bandpass filter (Hz)
TRIM_DB = 40                     # Silence threshold in dB

# === FUNCTIONS ===
def bandpass_filter(y, sr, lowcut, highcut):
    sos = butter(N=6, Wn=[lowcut, highcut], btype='band', fs=sr, output='sos')
    return sosfilt(sos, y)

def normalize_signal(y):
    return (y - np.mean(y)) / np.std(y)  # z-score normalization

def preprocess_file(file_path, output_path):
    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR)
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=TRIM_DB)
        # Bandpass filter
        y = bandpass_filter(y, sr, *BANDPASS_RANGE)
        # Normalize
        y = normalize_signal(y)
        sf.write(output_path, y, sr)
        print(f"Processed: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def preprocess_dataset(input_dir, output_dir):
    for anchor_folder in os.listdir(input_dir):
        anchor_path = os.path.join(input_dir, anchor_folder)
        if not os.path.isdir(anchor_path):
            continue

        output_anchor_path = os.path.join(output_dir, anchor_folder)
        os.makedirs(output_anchor_path, exist_ok=True)

        for filename in os.listdir(anchor_path):
            if filename.endswith(".wav"):
                input_file = os.path.join(anchor_path, filename)
                output_file = os.path.join(output_anchor_path, filename)
                preprocess_file(input_file, output_file)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preprocess_dataset(INPUT_DIR, OUTPUT_DIR)
