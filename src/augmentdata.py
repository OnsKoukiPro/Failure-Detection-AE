# src/augment.py
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
from scipy.signal import butter, filtfilt
import os

# Add constant definition
SAMPLE_RATE = 102400  # From your preprocessing script

def time_domain_augment(signal):
    """Apply time-domain augmentations"""
    # Gaussian noise
    noisy = signal + np.random.normal(0, 0.01*signal.std(), len(signal))

    # Time shifting (5% of signal length)
    shift = int(0.05 * len(signal))
    shifted = np.roll(signal, shift)

    return [noisy, shifted]

def frequency_domain_augment(signal, sr):
    """Apply frequency-domain augmentations"""
    # Bandpass filter (enhance failure-related frequencies)
    b, a = butter(4, [20000/(sr/2), 50000/(sr/2)], btype='band')
    filtered = filtfilt(b, a, signal)

    return [filtered]

if __name__ == "__main__":
    # Config
    INPUT_CSV = "/app/results/processed_data/processed_metadata.csv"
    OUTPUT_DIR = "/app/results/augmented_data"
    AUGMENT_FACTOR = 5

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Add directory creation

    # Load data
    df = pd.read_csv(INPUT_CSV)
    failure_samples = df[df['label'] == 1]

    # Augment failure samples
    augmented_data = []
    for _, row in tqdm(failure_samples.iterrows(), desc="Augmenting"):
        signal = np.load(row['processed_path'])

        # Apply augmentations
        augmented = []
        augmented += time_domain_augment(signal)
        augmented += frequency_domain_augment(signal, SAMPLE_RATE)  # Now using defined constant

        # Save augmented signals
        for i, aug_signal in enumerate(augmented[:AUGMENT_FACTOR]):
            path = f"{OUTPUT_DIR}/{row['measurement']}_aug{i}_{row['timestamp']}.npy"
            np.save(path, aug_signal)
            augmented_data.append({
                'processed_path': path,
                'label': 1,
                'measurement': row['measurement'],
                'timestamp': row['timestamp']
            })

    # Save augmented metadata
    pd.DataFrame(augmented_data).to_csv(f"{OUTPUT_DIR}/augmented_metadata.csv", index=False)