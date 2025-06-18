### Overview
This project is used to analyse Acoustic Emissions Data to analyse and predict Material Failure

### Implementation Steps
Step 1-Mapping WAV files to Metadata
Step 2-Preprocessing of WAV files :
    -Normalization
    -Filter noise
    -Trim silence
    -Combine channels
Step 2 bis : Preprocessing / augmenting data (due to imbalance between failure and non-failure classes)
Step 3- Feature Extraction
    - Time-Domain Features: Amplitude, RMS, zero-crossing rate.
    - Frequency-Domain Features:
    - FFT peaks, spectral entropy, Mel-frequency cepstral coefficients (MFCCs).
    - Time-Frequency Features:
        - Spectrograms (using librosa.stft or torchaudio).
        - Wavelet transforms (e.g., Continuous Wavelet Transform for AE bursts)
Step 4- Labeling for Failure Detection
Current Labels: good, invalid, Frequency not good, Quotient not good.
Redefine Labels:
If good = non-failure and Frequency not good/Quotient not good = failure indicators:
Binary labels: 0 (good), 1 (failure).
Step 5- Train-Test Split
Time-Series Split: Since timestamps are sequential, the data will be split chronologically
 (e.g., first 80% for training, next 10% validation, last 10% test).
- classes also present an imbalacement, so Stratified Split is also performed
Step 6- Train Model :
    - Traditional  ML with Random Forest Classifier
    - Convolutional Neural Network for Spectrograms
    - LSTM for time-series