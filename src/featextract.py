import os
import numpy as np
import pandas as pd
import librosa
import pywt
from scipy.fft import fft
from scipy.signal import find_peaks
from tqdm import tqdm

# Configuration
INPUT_CSV = "/app/results/processed_data/processed_metadata.csv"
OUTPUT_DIR = "/app/results/features"
SAMPLE_RATE = 500000
N_MFCC = 13

def extract_time_domain_features(signal):
    """Extract time-domain features."""
    features = {}

    # Amplitude statistics
    features["amplitude_mean"] = np.mean(np.abs(signal))
    features["amplitude_std"] = np.std(signal)

    # Root Mean Square (RMS)
    features["rms"] = np.sqrt(np.mean(signal ** 2))

    # Zero-Crossing Rate
    features["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(signal.reshape(1, -1))[0][0]

    return features

def extract_freq_domain_features(signal, sample_rate):
    """Extract frequency-domain features."""
    features = {}

    # FFT and spectrum
    fft_vals = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1 / sample_rate)
    positive_freqs = freqs[:len(freqs) // 2]
    positive_fft = fft_vals[:len(fft_vals) // 2]

    # Spectral entropy
    spectral_power = positive_fft / np.sum(positive_fft)
    features["spectral_entropy"] = -np.sum(spectral_power * np.log2(spectral_power + 1e-12))

    # FFT peaks (top 3 dominant frequencies)
    peaks, _ = find_peaks(positive_fft, height=0.1 * np.max(positive_fft))
    dominant_freqs = positive_freqs[peaks]
    for i in range(3):
        features[f"dominant_freq_{i+1}"] = dominant_freqs[i] if i < len(dominant_freqs) else 0.0

    # MFCCs (mean of coefficients)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"mfcc_{i+1}"] = np.mean(mfccs[i])

    return features

def extract_time_freq_features(signal, sample_rate):
    """Extract time-frequency features."""
    features = {}

    # Spectrogram (STFT magnitude)
    stft = librosa.stft(signal)
    stft_mag = np.abs(stft)
    features["spectrogram_mean"] = np.mean(stft_mag)
    features["spectrogram_std"] = np.std(stft_mag)

    # Wavelet Transform (Continuous Wavelet Transform - CWT)
    scales = pywt.frequency2scale('cmor1.5-1.0', np.array([1000, 10000]))  # Adjust scales for AE bursts
    cwtmatr, _ = pywt.cwt(signal, scales, 'cmor1.5-1.0', sampling_period=1/sample_rate)
    features["cwt_energy"] = np.mean(np.abs(cwtmatr))

    return features

if __name__ == "__main__":
    # Load metadata
    df = pd.read_csv(INPUT_CSV)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract features for each signal
    all_features = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        signal = np.load(row["processed_path"])
        features = {}

        # Merge features from all domains
        features.update(extract_time_domain_features(signal))
        features.update(extract_freq_domain_features(signal, SAMPLE_RATE))
        features.update(extract_time_freq_features(signal, SAMPLE_RATE))

        # Add metadata
        features["label"] = row["csv_label"]
        features["measurement"] = row["measurement"]
        all_features.append(features)

    # Save to CSV
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(os.path.join(OUTPUT_DIR, "features.csv"), index=False)
    print(f"Features saved to {OUTPUT_DIR}/features.csv")