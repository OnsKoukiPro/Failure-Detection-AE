import pandas as pd
import numpy as np
import librosa
import librosa.display
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Suppress all warnings more aggressively
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Configure TensorFlow to use less memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use a fraction of GPU memory
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]  # Limit to 1GB
        )
    except RuntimeError as e:
        print(e)

# Suppress lazy_loader warnings
import logging
logging.captureWarnings(True)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib

# Create output directory if it doesn't exist
output_base_path = "/app/results/KI2/"
os.makedirs(output_base_path, exist_ok=True)
os.makedirs(os.path.join(output_base_path, "models"), exist_ok=True)
os.makedirs(os.path.join(output_base_path, "plots"), exist_ok=True)
os.makedirs(os.path.join(output_base_path, "results"), exist_ok=True)

def extract_mfcc_features(file_path, sr=22050, n_mfcc=13, max_frames=500):
    """
    Extract MFCC features from audio file and pad/truncate to a consistent length
    This is more memory-efficient than using raw audio
    """
    try:
        # Load audio file
        audio, _ = librosa.load(file_path, sr=sr)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        # Transpose to get (time_steps, n_mfcc)
        mfccs = mfccs.T

        # Pad or truncate to max_frames
        if mfccs.shape[0] < max_frames:
            # Pad with zeros
            pad_width = max_frames - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), 'constant')
        else:
            # Truncate
            mfccs = mfccs[:max_frames, :]

        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((max_frames, n_mfcc))

def load_mfcc_data(file_paths, sr=22050, n_mfcc=13, max_frames=500):
    """
    Load MFCC features from all files
    """
    features = []

    print("Extracting MFCC features from audio files...")
    for i, file_path in enumerate(file_paths):
        if i % 10 == 0:
            print(f"Processing file {i+1}/{len(file_paths)}")

        mfccs = extract_mfcc_features(file_path, sr, n_mfcc, max_frames)
        features.append(mfccs)

    return np.array(features)

def check_data_quality(features):
    """
    Check for NaN, Inf, or other problematic values in the data
    """
    print("Checking data quality...")

    # Check for NaN values
    nan_count = np.sum(np.isnan(features))
    print(f"NaN values in features: {nan_count}")

    # Check for Inf values
    inf_count = np.sum(np.isinf(features))
    print(f"Inf values in features: {inf_count}")

    # Check for extreme values
    max_val = np.max(features)
    min_val = np.min(features)
    print(f"Feature value range: [{min_val}, {max_val}]")

    # Replace any NaN or Inf with zeros
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features

def create_stable_lstm_autoencoder(input_dim, timesteps, encoding_dim=32):
    """
    Create a more stable LSTM autoencoder model with regularization
    """
    # Input layer
    inputs = Input(shape=(timesteps, input_dim))

    # Encoder with regularization
    encoded = LSTM(64, activation='tanh', return_sequences=True,
                  kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))(inputs)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)

    encoded = LSTM(32, activation='tanh', return_sequences=False,
                  kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)

    encoded = Dense(encoding_dim, activation='tanh', kernel_regularizer=l2(0.01))(encoded)

    # Decoder
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(32, activation='tanh', return_sequences=True,
                  kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.2)(decoded)

    decoded = LSTM(64, activation='tanh', return_sequences=True,
                  kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.2)(decoded)

    decoded = TimeDistributed(Dense(input_dim))(decoded)

    # Autoencoder model
    autoencoder = Model(inputs, decoded)

    # Use a more conservative optimizer with gradient clipping
    optimizer = Adam(learning_rate=0.0005, clipvalue=0.5)
    # Alternatively, try RMSprop:
    # optimizer = RMSprop(learning_rate=0.0005, clipvalue=0.5)

    autoencoder.compile(optimizer=optimizer, loss='mse')

    return autoencoder

def train_anomaly_detector(features, condition, test_size=0.2, random_state=42):
    """
    Train an LSTM autoencoder for anomaly detection on MFCC features
    """
    # Split into train and test
    X_train, X_test = train_test_split(features, test_size=test_size, random_state=random_state)

    # Use RobustScaler which is less sensitive to outliers
    scaler = RobustScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

    # Check for NaN values after scaling
    X_train_scaled = check_data_quality(X_train_scaled)
    X_test_scaled = check_data_quality(X_test_scaled)

    # Create and train the model
    timesteps = X_train_scaled.shape[1]
    input_dim = X_train_scaled.shape[2]

    model = create_stable_lstm_autoencoder(input_dim, timesteps)

    print(f"Training LSTM autoencoder for {condition} condition...")

    # Add more callbacks for stability
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

    history = model.fit(
        X_train_scaled, X_train_scaled,
        epochs=100,  # More epochs but with early stopping
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Reconstruct test sequences
    reconstructions = model.predict(X_test_scaled, verbose=0)

    # Calculate reconstruction errors
    test_errors = np.mean(np.square(X_test_scaled - reconstructions), axis=(1, 2))

    # Check for NaN in errors
    if np.any(np.isnan(test_errors)):
        print("Warning: NaN values found in reconstruction errors. Replacing with large value.")
        test_errors = np.nan_to_num(test_errors, nan=1000.0)

    return model, history, X_test_scaled, reconstructions, test_errors, scaler

def plot_results(history, X_test, reconstructions, errors, condition, n_examples=3):
    """
    Plot training history, reconstructions, and error distribution
    """
    # Check if errors contain NaN or Inf
    if np.any(np.isnan(errors)) or np.any(np.isinf(errors)):
        print("Warning: Errors contain NaN or Inf values. Cannot create plots.")
        return []

    # Plot training history
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training History - {condition}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot example reconstructions (first MFCC coefficient only)
    plt.subplot(2, 2, 2)
    example_indices = np.random.choice(len(X_test), min(n_examples, len(X_test)), replace=False)
    for i, idx in enumerate(example_indices):
        plt.plot(X_test[idx][:100, 0], alpha=0.7, label=f'Original {i+1}' if i == 0 else "")
        plt.plot(reconstructions[idx][:100, 0], alpha=0.7, label=f'Reconstructed {i+1}' if i == 0 else "")
    plt.title(f'Example Reconstructions (1st MFCC) - {condition}')
    plt.xlabel('Time Step')
    plt.ylabel('MFCC Value (normalized)')
    plt.legend()

    # Plot error distribution
    plt.subplot(2, 2, 3)
    plt.hist(errors, bins=30)
    plt.title(f'Reconstruction Error Distribution - {condition}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')

    # Plot error vs sequence
    plt.subplot(2, 2, 4)
    plt.plot(errors)
    plt.title(f'Reconstruction Error by Sequence - {condition}')
    plt.xlabel('Sequence Index')
    plt.ylabel('Reconstruction Error')

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_base_path, "plots", f"lstm_results_{condition}.png")
    plt.savefig(plot_path)
    plt.close()

    return [plot_path]

def detect_anomalies(errors, threshold_std=2):
    """
    Detect anomalies based on reconstruction errors
    """
    # Check if errors contain NaN or Inf
    if np.any(np.isnan(errors)) or np.any(np.isinf(errors)):
        print("Warning: Errors contain NaN or Inf values. Using default threshold.")
        threshold = np.mean(errors[np.isfinite(errors)]) + threshold_std * np.std(errors[np.isfinite(errors)])
        anomaly_indices = np.where(errors > threshold)[0]
        return anomaly_indices, threshold

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    threshold = mean_error + threshold_std * std_error

    anomaly_indices = np.where(errors > threshold)[0]

    return anomaly_indices, threshold

def save_anomaly_results(anomaly_indices, errors, threshold, condition, df_meta, test_indices):
    """
    Save anomaly detection results to CSV files
    """
    # Create results DataFrame
    results = []
    for i, error in enumerate(errors):
        is_anomaly = i in anomaly_indices
        results.append({
            'sequence_index': i,
            'reconstruction_error': error,
            'is_anomaly': is_anomaly
        })

    results_df = pd.DataFrame(results)

    # Save results
    results_path = os.path.join(output_base_path, "results", f"anomaly_results_{condition}.csv")
    results_df.to_csv(results_path, index=False)

    # Save detailed anomaly information
    if len(anomaly_indices) > 0:
        anomaly_details = []
        for idx in anomaly_indices:
            # Get the corresponding file from metadata
            file_idx = test_indices[idx]
            anomaly_details.append({
                'sequence_index': idx,
                'reconstruction_error': errors[idx],
                'file_path': df_meta.iloc[file_idx]['file_path'],
                'filename': df_meta.iloc[file_idx]['filename'],
                'part_id': df_meta.iloc[file_idx]['part_id'],
                'position': df_meta.iloc[file_idx]['position'],
                'timestamp': df_meta.iloc[file_idx]['timestamp']
            })

        anomaly_details_df = pd.DataFrame(anomaly_details)
        anomaly_path = os.path.join(output_base_path, "results", f"anomaly_details_{condition}.csv")
        anomaly_details_df.to_csv(anomaly_path, index=False)

    return results_path

# Main execution
if __name__ == "__main__":
    # Load the metadata we created earlier
    metadata_path = os.path.join(output_base_path, "metadata.csv")
    df_meta = pd.read_csv(metadata_path)

    # Configuration - using MFCCs instead of raw audio to save memory
    sample_rate = 22050
    n_mfcc = 13
    max_frames = 500  # Reduced sequence length

    # Process each condition separately
    for condition in ['lying', 'hanging']:
        print(f"\n{'='*50}")
        print(f"Processing {condition} condition")
        print(f"{'='*50}")

        # Filter files for this condition
        condition_indices = df_meta[df_meta['condition'] == condition].index
        condition_files = df_meta[df_meta['condition'] == condition]['file_path'].values

        if len(condition_files) == 0:
            print(f"No files found for {condition} condition. Skipping.")
            continue

        # Load MFCC features (more memory efficient than raw audio)
        features = load_mfcc_data(
            condition_files,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            max_frames=max_frames
        )

        print(f"Features shape: {features.shape}")

        # Check data quality
        features = check_data_quality(features)

        # Train LSTM autoencoder
        model, history, X_test, reconstructions, test_errors, scaler = train_anomaly_detector(
            features,
            condition
        )

        # Plot results and save plots
        plot_paths = plot_results(history, X_test, reconstructions, test_errors, condition)
        print(f"Saved plots to: {plot_paths}")

        # Detect anomalies
        anomaly_indices, threshold = detect_anomalies(test_errors)

        print(f"Found {len(anomaly_indices)} anomalies in {condition} condition (threshold: {threshold:.4f})")

        # Get the indices of test files in the original metadata
        n_total = len(condition_files)
        n_test = len(test_errors)
        test_indices = np.random.choice(condition_indices, n_test, replace=False)

        # Save anomaly results
        results_path = save_anomaly_results(anomaly_indices, test_errors, threshold, condition, df_meta, test_indices)
        print(f"Saved results to: {results_path}")

        # Print anomalous files
        if len(anomaly_indices) > 0:
            print("Anomalous files:")
            for idx in anomaly_indices:
                file_idx = test_indices[idx]
                print(f"  {df_meta.iloc[file_idx]['filename']} (error: {test_errors[idx]:.4f})")

        # Save model
        model_path = os.path.join(output_base_path, "models", f"lstm_autoencoder_{condition}.h5")
        model.save(model_path)
        print(f"Model saved as {model_path}")

        # Save scaler
        scaler_path = os.path.join(output_base_path, "models", f"scaler_{condition}.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved as {scaler_path}")

    print(f"\nAll outputs have been saved to: {output_base_path}")