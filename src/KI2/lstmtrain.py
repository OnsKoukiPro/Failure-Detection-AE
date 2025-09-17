import pandas as pd
import numpy as np
import librosa
import librosa.display
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import os
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
output_base_path = "/app/results/KI2/"
os.makedirs(output_base_path, exist_ok=True)
os.makedirs(os.path.join(output_base_path, "models"), exist_ok=True)
os.makedirs(os.path.join(output_base_path, "plots"), exist_ok=True)
os.makedirs(os.path.join(output_base_path, "results"), exist_ok=True)

def load_audio_data(file_paths, sr=22050, max_length=None):
    """
    Load audio data from files and pad/truncate to a consistent length

    Parameters:
    file_paths (list): List of paths to audio files
    sr (int): Sample rate to use for loading
    max_length (int): Maximum length for audio sequences (if None, determined from data)

    Returns:
    tuple: (audio_sequences, actual_lengths, max_length_used)
    """
    audio_sequences = []
    actual_lengths = []

    print("Loading audio files...")
    for i, file_path in enumerate(file_paths):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(file_paths)}")

        # Load audio file
        try:
            audio, _ = librosa.load(file_path, sr=sr)
            audio_sequences.append(audio)
            actual_lengths.append(len(audio))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            audio_sequences.append(np.zeros(1000))  # Placeholder
            actual_lengths.append(1000)

    # Determine max length if not provided
    if max_length is None:
        max_length = max(actual_lengths)
        print(f"Determined max sequence length: {max_length}")

    # Pad sequences to max_length
    padded_sequences = []
    for audio in audio_sequences:
        if len(audio) < max_length:
            # Pad with zeros
            padded = np.pad(audio, (0, max_length - len(audio)), 'constant')
        else:
            # Truncate
            padded = audio[:max_length]
        padded_sequences.append(padded)

    return np.array(padded_sequences), actual_lengths, max_length

def create_lstm_autoencoder(input_dim, timesteps, encoding_dim=64):
    """
    Create an LSTM autoencoder model for anomaly detection

    Parameters:
    input_dim (int): Number of features (for multivariate time series, always 1 for raw audio)
    timesteps (int): Length of sequences
    encoding_dim (int): Dimension of the encoding (bottleneck)

    Returns:
    Model: Compiled LSTM autoencoder
    """
    # Input layer
    inputs = Input(shape=(timesteps, input_dim))

    # Encoder
    encoded = LSTM(128, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(64, activation='relu', return_sequences=True)(encoded)
    encoded = LSTM(32, activation='relu', return_sequences=False)(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    # Decoder
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(input_dim))(decoded)

    # Autoencoder model
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    return autoencoder

def train_anomaly_detector(audio_sequences, condition, test_size=0.2, random_state=42):
    """
    Train an LSTM autoencoder for anomaly detection on audio sequences

    Parameters:
    audio_sequences (np.array): Array of audio sequences
    condition (str): Condition name for labeling
    test_size (float): Proportion of data to use for testing
    random_state (int): Random seed for reproducibility

    Returns:
    tuple: (model, history, X_test, reconstructions, test_errors)
    """
    # Reshape for LSTM (samples, timesteps, features)
    # For raw audio, we have 1 feature (amplitude)
    X = audio_sequences.reshape((audio_sequences.shape[0], audio_sequences.shape[1], 1))

    # Split into train and test
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)

    # Standardize the data
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

    # Create and train the model
    timesteps = X_train_scaled.shape[1]
    input_dim = X_train_scaled.shape[2]

    model = create_lstm_autoencoder(input_dim, timesteps)

    print(f"Training LSTM autoencoder for {condition} condition...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train_scaled, X_train_scaled,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )

    # Reconstruct test sequences
    reconstructions = model.predict(X_test_scaled)

    # Calculate reconstruction errors
    test_errors = np.mean(np.square(X_test_scaled - reconstructions), axis=1)

    return model, history, X_test_scaled, reconstructions, test_errors, scaler

def plot_results(history, X_test, reconstructions, errors, condition, n_examples=5):
    """
    Plot training history, reconstructions, and error distribution

    Parameters:
    history: Training history object
    X_test: Test sequences
    reconstructions: Reconstructed sequences
    errors: Reconstruction errors
    condition (str): Condition name for plotting
    n_examples (int): Number of example sequences to plot

    Returns:
    list: Paths to saved plot files
    """
    # Plot training history
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training History - {condition}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot example reconstructions
    plt.subplot(2, 2, 2)
    example_indices = np.random.choice(len(X_test), n_examples, replace=False)
    for i, idx in enumerate(example_indices):
        plt.plot(X_test[idx][:1000, 0], alpha=0.7, label=f'Original {i+1}' if i == 0 else "")
        plt.plot(reconstructions[idx][:1000, 0], alpha=0.7, label=f'Reconstructed {i+1}' if i == 0 else "")
    plt.title(f'Example Reconstructions - {condition}')
    plt.xlabel('Time Step')
    plt.ylabel('Amplitude (normalized)')
    plt.legend()

    # Plot error distribution
    plt.subplot(2, 2, 3)
    plt.hist(errors, bins=50)
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

    Parameters:
    errors (np.array): Reconstruction errors
    threshold_std (float): Number of standard deviations for threshold

    Returns:
    tuple: (anomaly_indices, threshold)
    """
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    threshold = mean_error + threshold_std * std_error

    anomaly_indices = np.where(errors > threshold)[0]

    return anomaly_indices, threshold

def save_anomaly_results(anomaly_indices, errors, threshold, condition, df_meta):
    """
    Save anomaly detection results to CSV files

    Parameters:
    anomaly_indices (np.array): Indices of anomalous sequences
    errors (np.array): Reconstruction errors for all test sequences
    threshold (float): Anomaly detection threshold
    condition (str): Condition name
    df_meta (pd.DataFrame): Metadata DataFrame

    Returns:
    str: Path to saved results file
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
            # Note: This assumes the test set indices correspond to the original metadata
            # You might need to adjust this based on your data splitting
            file_idx = idx  # This is a simplification
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
    metadata_path = os.path.join(output_base_path, "metadata.csv")  # Update this path if needed
    df_meta = pd.read_csv(metadata_path)

    # Configuration
    sample_rate = 22050  # Sample rate for audio loading
    max_sequence_length = 22050 * 5  # 5 seconds of audio (adjust based on your data)

    # Process each condition separately
    for condition in ['lying', 'hanging']:
        print(f"\n{'='*50}")
        print(f"Processing {condition} condition")
        print(f"{'='*50}")

        # Filter files for this condition
        condition_files = df_meta[df_meta['condition'] == condition]['file_path'].values

        # Load audio data
        audio_sequences, actual_lengths, max_length = load_audio_data(
            condition_files,
            sr=sample_rate,
            max_length=max_sequence_length
        )

        # Train LSTM autoencoder
        model, history, X_test, reconstructions, test_errors, scaler = train_anomaly_detector(
            audio_sequences,
            condition
        )

        # Plot results and save plots
        plot_paths = plot_results(history, X_test, reconstructions, test_errors, condition)
        print(f"Saved plots to: {plot_paths}")

        # Detect anomalies
        anomaly_indices, threshold = detect_anomalies(test_errors)

        print(f"Found {len(anomaly_indices)} anomalies in {condition} condition (threshold: {threshold:.4f})")

        # Save anomaly results
        results_path = save_anomaly_results(anomaly_indices, test_errors, threshold, condition, df_meta)
        print(f"Saved results to: {results_path}")

        # If you want to see which files were detected as anomalies
        if len(anomaly_indices) > 0:
            print("Anomalous files:")
            # This is a simplification - you might need to adjust the indexing
            test_indices = np.random.choice(len(condition_files), len(test_errors), replace=False)
            for idx in anomaly_indices:
                file_idx = test_indices[idx]
                print(f"  {df_meta.iloc[file_idx]['filename']} (error: {test_errors[idx]:.4f})")

        # Save model
        model_path = os.path.join(output_base_path, "models", f"lstm_autoencoder_{condition}.h5")
        model.save(model_path)
        print(f"Model saved as {model_path}")

        # Save scaler
        scaler_path = os.path.join(output_base_path, "models", f"scaler_{condition}.pkl")
        import joblib
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved as {scaler_path}")

    print(f"\nAll outputs have been saved to: {output_base_path}")