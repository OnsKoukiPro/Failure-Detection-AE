import os
import numpy as np
import pandas as pd
import librosa
import pywt
import nolds
from scipy.fft import fft
from scipy.signal import find_peaks, stft
from scipy.stats import kurtosis, skew
from tqdm import tqdm
import traceback

INPUT_CSV = "/app/results/processed_data/processed_metadata.csv"
OUTPUT_DIR = "/app/results/features"
SAMPLE_RATE = 102400
N_MFCC = 13
MIN_SIGNAL_LENGTH = 1024

def extract_time_domain_features(signal):
    """Extract time-domain features with fallbacks"""
    features = {}
    abs_signal = np.abs(signal)

    features["amplitude_mean"] = np.mean(abs_signal)
    features["amplitude_std"] = np.std(signal)
    features["amplitude_max"] = np.max(abs_signal)
    features["rms"] = np.sqrt(np.mean(signal ** 2))
    features["crest_factor"] = np.max(abs_signal) / (features["rms"] + 1e-12)

    features["kurtosis"] = kurtosis(signal)
    features["skewness"] = skew(signal)

    # Zero crossing rate
    try:
        features["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(
            signal.reshape(1, -1), frame_length=len(signal), center=False
        )[0][0]
    except AttributeError:
        features["zero_crossing_rate"] = np.mean(librosa.zero_crossings(signal))

    features["duration"] = len(signal) / SAMPLE_RATE
    return features

def extract_complexity_features(signal):
    """Extract signal complexity features"""
    features = {}

    if len(signal) < MIN_SIGNAL_LENGTH:
        features["sample_entropy"] = 0
        features["hfd"] = 0
        return features

    try:
        features["sample_entropy"] = nolds.sample_entropy(signal)
        features["hfd"] = nolds.hfd(signal, k_max=10)
    except:
        features["sample_entropy"] = 0
        features["hfd"] = 0

    return features

def extract_freq_domain_features(signal, sample_rate):
    """Extract frequency-domain features"""
    features = {}

    n_fft = min(32768, len(signal))
    if n_fft < 256:
        for i in range(N_MFCC):
            features[f"mfcc_{i+1}"] = 0
        features["spectral_entropy"] = 0
        for i in range(3):
            features[f"dominant_freq_{i+1}"] = 0
        features["spectral_centroid"] = 0
        return features

    try:
        # Compute FFT
        fft_vals = np.abs(fft(signal, n=n_fft))
        freqs = np.fft.fftfreq(n_fft, 1 / sample_rate)
        positive_mask = freqs > 0
        positive_freqs = freqs[positive_mask]
        positive_fft = fft_vals[positive_mask]

        # Spectral entropy
        spectral_power = positive_fft / (np.sum(positive_fft) + 1e-12)
        features["spectral_entropy"] = -np.sum(spectral_power * np.log2(spectral_power + 1e-12))

        # Dominant frequencies
        peaks, _ = find_peaks(positive_fft, height=0.1 * np.max(positive_fft))
        dominant_freqs = positive_freqs[peaks]
        for i in range(3):
            features[f"dominant_freq_{i+1}"] = dominant_freqs[i] if i < len(dominant_freqs) else 0.0
    except:
        features["spectral_entropy"] = 0
        for i in range(3):
            features[f"dominant_freq_{i+1}"] = 0.0

    try:
        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=signal,
            sr=sample_rate,
            n_mfcc=N_MFCC,
            n_fft=min(4096, len(signal)),
            hop_length=min(1024, len(signal)//4),
            fmax=sample_rate//2
        )
        for i in range(N_MFCC):
            features[f"mfcc_{i+1}"] = np.mean(mfccs[i])
    except:
        for i in range(N_MFCC):
            features[f"mfcc_{i+1}"] = 0.0

    try:
        # Spectral centroid
        features["spectral_centroid"] = librosa.feature.spectral_centroid(
            y=signal, sr=sample_rate, n_fft=min(4096, len(signal))
        )[0][0]
    except:
        try:
            # Fallback method
            S = np.abs(librosa.stft(signal, n_fft=min(4096, len(signal))))
            features["spectral_centroid"] = librosa.feature.spectral_centroid(S=S)[0][0]
        except:
            features["spectral_centroid"] = 0

    return features

def extract_time_freq_features(signal, sample_rate):
    """Extract time-frequency features"""
    features = {
        "spectrogram_mean": 0,
        "spectrogram_std": 0,
        "cwt_energy": 0
    }

    n_fft = min(4096, len(signal))
    if n_fft < 256:
        return features

    # STFT-based features
    try:
        _, _, stft_spec = stft(
            signal,
            fs=sample_rate,
            nperseg=n_fft,
            noverlap=n_fft//2
        )
        stft_mag = np.abs(stft_spec)
        features["spectrogram_mean"] = np.mean(stft_mag)
        features["spectrogram_std"] = np.std(stft_mag)
    except:
        pass

    # Wavelet transform
    try:
        central_freq = pywt.central_frequency('cmor1.5-1.0')
        scales = central_freq * sample_rate / np.logspace(
            np.log10(1000),
            np.log10(50000),
            8
        )
        cwtmatr, _ = pywt.cwt(signal, scales, 'cmor1.5-1.0', sampling_period=1/sample_rate)
        features["cwt_energy"] = np.mean(np.abs(cwtmatr))
    except:
        pass

    return features

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded metadata with {len(df)} records")
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        exit(1)

    all_features = []
    error_count = 0
    processed_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            signal = np.load(row["processed_path"])

            if len(signal) == 0:
                print(f"Empty signal: {row['processed_path']}")
                continue

            features = {}
            features.update(extract_time_domain_features(signal))
            features.update(extract_complexity_features(signal))
            features.update(extract_freq_domain_features(signal, SAMPLE_RATE))
            features.update(extract_time_freq_features(signal, SAMPLE_RATE))

            # Metadata
            features["measurement"] = row["measurement"]
            features["timestamp"] = row["timestamp"]

            if "label" in row:
                features["label"] = row["label"]

            all_features.append(features)
            processed_count += 1
        except Exception as e:
            error_count += 1
            print(f"Error processing {row.get('processed_path', 'unknown')}: {str(e)}")
            traceback.print_exc()
            continue

    if processed_count > 0:
        feature_df = pd.DataFrame(all_features)
        output_path = os.path.join(OUTPUT_DIR, "unsupervised_features.csv")
        feature_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully processed {processed_count}/{len(df)} files")
        print(f"Features saved to {output_path}")
    else:
        print("\nNo features extracted - all files failed processing")

    if error_count > 0:
        print(f"Encountered {error_count} errors during processing")

    print("Feature extraction complete")