# Acoustic Emissions Analysis for Material Failure Prediction

## Overview

This project aims to **analyze Acoustic Emissions (AE) data** to detect and predict **material failure** events. By leveraging signal processing and machine learning techniques, we aim to distinguish between normal and failure-related acoustic patterns.

---

## Implementation Steps

### Step 1 — Mapping WAV Files to Metadata
- Link each `.wav` file to its associated metadata for traceability and labeling.

---

### Step 2 — Preprocessing of WAV Files
- **Normalization**: Adjust signal amplitude to a standard scale.
- **Noise Filtering**: Apply bandpass or notch filters to remove background noise.
- **Silence Trimming**: Remove inactive periods from start and end.
- **Channel Combining**: Merge stereo into mono if applicable.

---

### Step 2 bis — Data Augmentation (for Class Imbalance)
- Apply transformations (e.g., pitch shift, time stretch) to **augment minority class** (failure cases).
- Ensure balanced representation of failure and non-failure scenarios.

---

### Step 3 — Feature Extraction
- **Time-Domain Features**:
  - Amplitude statistics, RMS, Zero-Crossing Rate
- **Frequency-Domain Features**:
  - FFT peaks, Spectral Entropy, MFCCs
- **Time-Frequency Features**:
  - Spectrograms (e.g., using `librosa.stft`, `torchaudio`)
  - Wavelet Transforms (e.g., Continuous Wavelet Transform)

---

### Step 4 — Labeling for Failure Detection
- Original labels: `good`, `invalid`, `Frequency not good`, `Quotient not good`
- Redefined binary labels:
  - `0` → Non-failure (`good`)
  - `1` → Failure (`Frequency not good`, `Quotient not good`)

---

### Step 5 — Train-Test Split
- **Chronological (Time-Series) Split**:
  - First 80% → Training
  - Next 10% → Validation
  - Last 10% → Test
- **Stratified Sampling**: Performed to address class imbalance.

---

### Step 6 — Model Training
- **Traditional Machine Learning**:
  - `RandomForestClassifier` on extracted features
- **Deep Learning Approaches**:
  - **CNN**: Applied to spectrogram images
  - **LSTM**: Trained on raw or windowed time-series signal
