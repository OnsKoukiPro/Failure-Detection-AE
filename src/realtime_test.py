# src/realtime_test.py
import numpy as np
import joblib
import librosa
import pywt
from time import sleep
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import warnings
import pandas as pd

class RealTimeAEAnalyzer:
    def __init__(self, model_path, sample_rate=102400):
        self.model = joblib.load(model_path)
        self.sample_rate = sample_rate
        self.buffer = []
        self.filter_coeffs = self._create_filter()

    def _create_filter(self):
        """Create bandpass filter coefficients matching preprocessing"""
        nyq = 0.5 * self.sample_rate
        low = 1000 / nyq
        high = 50000 / nyq
        b, a = butter(4, [low, high], btype='band')
        return (b, a)

    def _process_chunk(self, raw_signal):
        """Replicate exact preprocessing steps"""
        # 1. Normalization
        signal = raw_signal.astype(np.float32) / np.max(np.abs(raw_signal))

        # 2. Bandpass filtering
        signal = filtfilt(*self.filter_coeffs, signal)

        # 3. Trim silence (same parameters as preprocessing)
        abs_signal = np.abs(signal)
        threshold = 0.01 * np.max(abs_signal)
        start = np.argmax(abs_signal > threshold)
        end = len(abs_signal) - np.argmax(abs_signal[::-1] > threshold)
        return signal[start:end]

    def _extract_features(self, processed_signal):
        """Complete feature extraction matching training pipeline"""
        features = {}

        # Time-domain features
        features['amplitude_mean'] = np.mean(np.abs(processed_signal))
        features['amplitude_std'] = np.std(processed_signal)
        features['rms'] = np.sqrt(np.mean(processed_signal ** 2))
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            processed_signal.reshape(1, -1))[0][0]

        # Frequency-domain features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fft_vals = np.abs(fft(processed_signal))
            freqs = np.fft.fftfreq(len(processed_signal), 1 / self.sample_rate)

        positive_fft = fft_vals[:len(fft_vals)//2]
        spectral_power = positive_fft / (np.sum(positive_fft) + 1e-12)
        features['spectral_entropy'] = -np.sum(spectral_power * np.log2(spectral_power + 1e-12))

        # MFCC features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mfccs = librosa.feature.mfcc(
                y=processed_signal,
                sr=self.sample_rate,
                n_mfcc=13,
                n_fft=min(131072, len(processed_signal)),
                hop_length=min(131072, len(processed_signal)),
                window='boxcar',
                fmax=self.sample_rate//2
            )
        for i in range(13):
            features[f'mfcc_{i+1}'] = np.mean(mfccs[i])

        # Time-frequency features (CWT)
        central_freq = 1.0
        desired_freqs = np.array([20000, 50000])
        scales = (central_freq * self.sample_rate) / desired_freqs

        try:
            if np.any(scales < 1):
                raise ValueError("Invalid CWT scales")

            cwtmatr, _ = pywt.cwt(processed_signal, scales, 'cmor1.5-1.0',
                                 sampling_period=1/self.sample_rate)
            features['cwt_energy'] = np.mean(np.abs(cwtmatr))
        except Exception as e:
            features['cwt_energy'] = 0.0  # Fallback value
            print(f"CWT warning: {str(e)}")

        return [
            features['mfcc_1'],
            features['spectral_entropy'],
            features['cwt_energy']
        ]

    def process_audio_chunk(self, raw_chunk):
        """Full processing pipeline"""
        processed = self._process_chunk(raw_chunk)

        # Pad/truncate to 51200 samples if needed
        processed = self._adjust_length(processed, target_length=51200)

        features = self._extract_features(processed)

        # Buffer management (5-sample window)
        self.buffer.append(features)
        if len(self.buffer) >= 5:
            X = pd.DataFrame(self.buffer, columns=['mfcc_1', 'spectral_entropy', 'cwt_energy'])
            proba = self.model.predict_proba(X)[:, 1].mean()
            self.buffer = []
            return proba
        return None

    def _adjust_length(self, signal, target_length):
        """Ensure consistent signal length"""
        if len(signal) > target_length:
            return signal[:target_length]
        elif len(signal) < target_length:
            return np.pad(signal, (0, target_length - len(signal)))
        return signal

# Modified signal generation section
if __name__ == "__main__":
    analyzer = RealTimeAEAnalyzer('/app/results/models/failure_detector.pkl')

    # Test parameters
    num_chunks = 10
    failure_mode = True

    for i in range(num_chunks):
        print(f"\n--- Processing chunk {i+1}/{num_chunks} ---")

        # Create base signal with proper scaling
        duration = 0.5  # 0.5 seconds
        freq = 35000 if failure_mode else 15000
        t = np.linspace(0, duration, 51200)

        # Generate scaled sine wave (-0.9 to 0.9 range)
        signal = 0.9 * np.sin(2 * np.pi * freq * t)

        # Add controlled noise (10% of signal range)
        signal += 0.1 * np.random.randn(51200)

        if failure_mode and i % 3 == 0:
            # Add controlled bursts (50% of signal range)
            burst_pos = np.random.randint(10000, 40000)
            signal[burst_pos:burst_pos+1000] += 0.5 * np.random.rand(1000)

        # Clip to prevent overflow and convert to int16
        signal = np.clip(signal, -1.0, 1.0)
        signal_int16 = (signal * 32767).astype(np.int16)

        # Calculate safe RMS
        rms = np.sqrt(np.mean(signal_int16.astype(float)**2))
        print(f"Signal characteristics:")
        print(f"- Length: {len(signal_int16)} samples")
        print(f"- Max amplitude: {np.max(signal_int16):.0f}")
        print(f"- RMS: {rms:.0f}")

        result = analyzer.process_audio_chunk(signal_int16)
        if result is not None:
            status = "FAILURE DETECTED" if result > 0.5 else "Normal operation"
            print(f"Result: {result:.2f} | {status}")

        sleep(0.1)

    print("\nTest completed after processing", num_chunks, "chunks")