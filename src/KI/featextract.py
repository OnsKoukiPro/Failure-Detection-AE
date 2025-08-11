import numpy as np
from scipy.fft import fft
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt
import librosa
import pywt
import pandas as pd
from tqdm import tqdm

def extract_freq_features(signal, sr=51200):
    """
    Extract frequency-domain features robust to 2x frequency spread
    Args:
        signal: Audio signal (1D array)
        sr: Sampling rate (default 51200 Hz)
    Returns:
        Dictionary of frequency spread-robust features
    """
    fft_vals = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/sr)

    # Account for 2x spread - focus on broader frequency range
    positive_mask = (freqs > 0) & (freqs < 20000)
    positive_freqs = freqs[positive_mask]
    positive_fft = fft_vals[positive_mask]

    # Calculate spectral centroid (spread-robust version)
    spectral_centroid = np.sum(positive_freqs * positive_fft) / (np.sum(positive_fft) + 1e-12)

    # Spread-robust features
    features = {
        "freq_spread": np.std(positive_freqs),
        "freq_skew": skew(positive_fft),
        "freq_kurtosis": kurtosis(positive_fft),
        "spectral_centroid": spectral_centroid,
        "freq_range": np.ptp(positive_freqs),
        "freq_entropy": -np.sum(
            (positive_fft / (np.sum(positive_fft) + 1e-12)) *
            np.log2(positive_fft / (np.sum(positive_fft) + 1e-12 + 1e-12)
        ))
    }
    return features

def extract_anchor_features(signal, anchor_point, sr=51200):
    """
    Extract anchor-point specific features
    Args:
        signal: Audio signal (1D array)
        anchor_point: One of ['Anschlagpunkt1', 'Anschlagpunkt2', ...]
        sr: Sampling rate
    Returns:
        Dictionary of anchor-specific features
    """
    # Custom wavelet scales per anchor point based on physics
    anchor_scales = {
        "Anschlagpunkt1": np.linspace(1, 50, 20),    # 0° - broader range
        "Anschlagpunkt2": np.linspace(30, 80, 20),   # 30° - mid-high range
        "Anschlagpunkt3": np.linspace(40, 100, 20),  # 45° - higher frequencies
        "Anschlagpunkt4": np.linspace(20, 60, 20)    # 115° - mid-range
    }

    # Validate anchor point
    if anchor_point not in anchor_scales:
        raise ValueError(f"Unknown anchor point: {anchor_point}")

    # Calculate Continuous Wavelet Transform
    cwtmatr, _ = pywt.cwt(signal, anchor_scales[anchor_point], 'morl')

    # Extract energy statistics
    cwt_abs = np.abs(cwtmatr)
    return {
        "cwt_energy_mean": np.mean(cwt_abs),
        "cwt_energy_std": np.std(cwt_abs),
        "cwt_energy_max": np.max(cwt_abs),
        "cwt_energy_ratio": np.max(cwt_abs) / (np.mean(cwt_abs) + 1e-12)
    }

def extract_spread_robust_mfcc(signal, sr=51200):
    """
    Extract MFCCs adapted for wider frequency spread
    Args:
        signal: Audio signal (1D array)
        sr: Sampling rate
    Returns:
        Dictionary of MFCC features (mean values)
    """
    # Wider frequency bins for spread (200-20000 Hz range)
    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=10,
        fmin=200,
        fmax=20000,
        n_fft=2048,
        hop_length=512,
        lifter=20  # Emphasize higher-order coefficients
    )

    # Return mean and std of each MFCC coefficient
    return {f"mfcc_spread_{i+1}_mean": np.mean(mfccs[i]) for i in range(10)} | \
           {f"mfcc_spread_{i+1}_std": np.std(mfccs[i]) for i in range(10)}

def extract_logmel_features(signal, sr=51200, n_mels=64):
    """Log-mel spectrogram statistics."""
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=n_mels,
        fmin=200,
        fmax=20000
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return {
        "logmel_mean": np.mean(log_mel_spec),
        "logmel_std": np.std(log_mel_spec),
        "logmel_skew": skew(log_mel_spec.flatten()),
        "logmel_kurtosis": kurtosis(log_mel_spec.flatten())
    }

def extract_additional_timefreq(signal, sr=51200):
    """Extra basic spectral + time features."""
    zcr = librosa.feature.zero_crossing_rate(y=signal)[0]
    spec_cent = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    return {
        "zero_crossing_rate_mean": np.mean(zcr),
        "spectral_bandwidth_mean": np.mean(spec_bw),
        "spectral_bandwidth_std": np.std(spec_bw)
    }


def extract_all_features(filepath, anchor_point, sr=51200):
    """
    Full feature extraction pipeline for a single file
    Args:
        filepath: Path to audio file
        anchor_point: Anchor point identifier
        sr: Sampling rate
    Returns:
        Dictionary of all extracted features
    """
    # Load and preprocess audio
    signal, _ = librosa.load(filepath, sr=sr, mono=True)

    # Apply bandpass filter (500Hz-20kHz)
    nyquist = 0.5 * sr
    low = 500 / nyquist
    high = 20000 / nyquist
    b, a = butter(4, [low, high], btype='band')
    signal = filtfilt(b, a, signal)

    # Normalize without amplitude dependence
    signal = signal / (np.max(np.abs(signal)) + 1e-12)

    # Extract all feature types
    features = {}
    features.update(extract_freq_features(signal, sr))
    features.update(extract_anchor_features(signal, anchor_point, sr))
    features.update(extract_spread_robust_mfcc(signal, sr))
    features.update(extract_logmel_features(signal, sr))
    features.update(extract_additional_timefreq(signal, sr))


    return features

def batch_feature_extraction(metadata_csv, output_csv):
    """
    Process all files in metadata CSV
    Args:
        metadata_csv: Path to metadata CSV
        output_csv: Path to save features
    """
    # Load metadata
    df = pd.read_csv(metadata_csv)
    features_list = []

    # Process each file
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            features = extract_all_features(
                row['filepath'],
                row['anchor_point']
            )
            # Add metadata
            features['filepath'] = row['filepath']
            features['anchor_point'] = row['anchor_point']
            features['timestamp'] = row['timestamp']
            features_list.append(features)
        except Exception as e:
            print(f"Error processing {row['filepath']}: {str(e)}")

    # Save to CSV
    feature_df = pd.DataFrame(features_list)
    feature_df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")
    return feature_df

# Example Usage
if __name__ == "__main__":
    # First run metadata extraction from previous step
    # Then run feature extraction
    feature_df = batch_feature_extraction(
        "/app/results/KI/metadata.csv",
        "/app/results/KI/features.csv"
    )