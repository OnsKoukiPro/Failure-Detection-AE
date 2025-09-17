import pandas as pd
import numpy as np
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def extract_features(file_path, n_mfcc=13, sr=22050):
    """
    Load a WAV file, extract MFCCs, and return a summary stat vector (mean, std, etc.)

    Parameters:
    file_path (str): Path to the WAV file
    n_mfcc (int): Number of MFCC coefficients to extract
    sr (int): Sample rate for audio loading

    Returns:
    np.array: Feature vector containing MFCC statistics
    """
    try:
        # Load audio file. Use `sr` to resample for consistency.
        audio, sample_rate = librosa.load(file_path, sr=sr)

        # Extract MFCCs. `n_mfcc` is the number of coefficients to return.
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        # To create a single feature vector per file, we take statistics over time.
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        mfccs_max = np.max(mfccs, axis=1)
        mfccs_min = np.min(mfccs, axis=1)

        # Combine all stats into one long feature vector
        feature_vector = np.concatenate((mfccs_mean, mfccs_std, mfccs_max, mfccs_min))
        return feature_vector

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_all_features(df_meta, n_mfcc=13, sr=22050):
    """
    Extract features for all files in the metadata DataFrame

    Parameters:
    df_meta (pd.DataFrame): Metadata DataFrame with file paths
    n_mfcc (int): Number of MFCC coefficients to extract
    sr (int): Sample rate for audio loading

    Returns:
    tuple: (X, feature_names) where X is the feature matrix and feature_names is a list of feature names
    """
    print("Extracting features from audio files...")

    # Apply feature extraction to all files
    X_list = []
    failed_files = []

    for idx, row in df_meta.iterrows():
        feature_vector = extract_features(row['file_path'], n_mfcc, sr)
        if feature_vector is not None:
            X_list.append(feature_vector)
        else:
            failed_files.append(row['file_path'])

    if failed_files:
        print(f"Failed to process {len(failed_files)} files:")
        for f in failed_files:
            print(f"  {f}")

    X = np.array(X_list)  # This is our feature matrix

    # Create feature names
    feature_names = []
    stats = ['mean', 'std', 'max', 'min']
    for stat in stats:
        for i in range(n_mfcc):
            feature_names.append(f"mfcc_{i}_{stat}")

    print(f"Feature matrix shape: {X.shape}")  # (n_samples, n_features)

    return X, feature_names

def visualize_features(X, df_meta, condition):
    """
    Visualize the extracted features using UMAP

    Parameters:
    X (np.array): Feature matrix
    df_meta (pd.DataFrame): Metadata DataFrame
    condition (str): Condition to visualize ('lying', 'hanging', or 'all')
    """
    # Filter data based on condition
    if condition != 'all':
        mask = df_meta['condition'] == condition
        X_filtered = X[mask]
        df_filtered = df_meta[mask]
        title_suffix = f"({condition} condition)"
    else:
        X_filtered = X
        df_filtered = df_meta
        title_suffix = "(all conditions)"

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    # Reduce dimensionality with UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'UMAP1': X_umap[:, 0],
        'UMAP2': X_umap[:, 1],
        'part_id': df_filtered['part_id'].values,
        'position': df_filtered['position'].values
    })

    # Plot by part_id
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    unique_parts = plot_df['part_id'].unique()
    colors = sns.color_palette('tab20', len(unique_parts))
    for i, part_id in enumerate(unique_parts):
        part_data = plot_df[plot_df['part_id'] == part_id]
        plt.scatter(part_data['UMAP1'], part_data['UMAP2'],
                   color=colors[i], label=part_id, alpha=0.7)
    plt.title(f'UMAP Projection by Part ID {title_suffix}')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot by position
    plt.subplot(1, 2, 2)
    unique_positions = plot_df['position'].unique()
    colors = sns.color_palette('tab10', len(unique_positions))
    for i, position in enumerate(unique_positions):
        pos_data = plot_df[plot_df['position'] == position]
        plt.scatter(pos_data['UMAP1'], pos_data['UMAP2'],
                   color=colors[i], label=position, alpha=0.7)
    plt.title(f'UMAP Projection by Position {title_suffix}')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

def save_features(X, feature_names, df_meta, output_path):
    """
    Save the extracted features to a CSV file

    Parameters:
    X (np.array): Feature matrix
    feature_names (list): List of feature names
    df_meta (pd.DataFrame): Metadata DataFrame
    output_path (str): Path to save the CSV file
    """
    # Create a DataFrame with the features
    features_df = pd.DataFrame(X, columns=feature_names)

    # Add metadata columns
    features_df['file_path'] = df_meta['file_path'].values
    features_df['filename'] = df_meta['filename'].values
    features_df['part_id'] = df_meta['part_id'].values
    features_df['position'] = df_meta['position'].values
    features_df['timestamp'] = df_meta['timestamp'].values
    features_df['condition'] = df_meta['condition'].values

    # Save to CSV
    features_df.to_csv(output_path, index=False)
    print(f"Features saved to: {output_path}")

# Main execution
if __name__ == "__main__":
    # Load the metadata we created earlier
    metadata_path = "/app/results/KI2/metadata.csv"  # Update this path if needed
    df_meta = pd.read_csv(metadata_path)
    df_meta['timestamp'] = pd.to_datetime(df_meta['timestamp'])

    # Configuration
    n_mfcc = 13  # Number of MFCC coefficients to extract
    sample_rate = 22050  # Sample rate for audio loading
    output_features_path = "/app/results/KI2/audio_features.csv"

    # Extract features
    X, feature_names = extract_all_features(df_meta, n_mfcc, sample_rate)

    # Visualize features for different conditions
    visualize_features(X, df_meta, 'lying')
    visualize_features(X, df_meta, 'hanging')
    visualize_features(X, df_meta, 'all')

    # Save features to CSV
    save_features(X, feature_names, df_meta, output_features_path)

    print("Feature extraction completed successfully!")