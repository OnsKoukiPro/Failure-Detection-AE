import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
output_base_path = "/app/results/KI2/"
os.makedirs(output_base_path, exist_ok=True)
os.makedirs(os.path.join(output_base_path, "models"), exist_ok=True)
os.makedirs(os.path.join(output_base_path, "plots"), exist_ok=True)
os.makedirs(os.path.join(output_base_path, "results"), exist_ok=True)

def load_features(features_path):
    """
    Load the extracted MFCC features
    """
    features_df = pd.read_csv(features_path)

    # Separate features and metadata
    feature_columns = [col for col in features_df.columns if col.startswith('mfcc_')]
    X = features_df[feature_columns].values
    metadata = features_df.drop(columns=feature_columns)

    return X, metadata, feature_columns

def train_isolation_forest(X, metadata, condition, contamination=0.1):
    """
    Train an Isolation Forest model for anomaly detection
    """
    print(f"Training Isolation Forest for {condition} condition...")

    # Filter data for the specific condition
    condition_mask = metadata['condition'] == condition
    X_condition = X[condition_mask]
    metadata_condition = metadata[condition_mask]

    if len(X_condition) == 0:
        print(f"No data found for {condition} condition. Skipping.")
        return None, None, None, None

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_condition)

    # Train Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    # Fit the model
    iso_forest.fit(X_scaled)

    # Predict anomalies (-1 for anomalies, 1 for normal)
    predictions = iso_forest.predict(X_scaled)

    # Convert to binary (0 for normal, 1 for anomaly)
    binary_predictions = np.where(predictions == -1, 1, 0)

    # Get anomaly scores (the lower, the more abnormal)
    anomaly_scores = iso_forest.decision_function(X_scaled)

    return iso_forest, scaler, binary_predictions, anomaly_scores, metadata_condition

def plot_isolation_forest_results(anomaly_scores, predictions, metadata, condition):
    """
    Plot and save visualization of Isolation Forest results
    """
    # Create results DataFrame
    results_df = metadata.copy()
    results_df['anomaly_score'] = anomaly_scores
    results_df['prediction'] = predictions

    # Plot anomaly score distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(anomaly_scores, bins=50, alpha=0.7)
    plt.axvline(np.mean(anomaly_scores), color='r', linestyle='--', label='Mean score')
    plt.title(f'Isolation Forest Anomaly Score Distribution - {condition}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot anomalies by part_id if available
    if 'part_id' in results_df.columns:
        plt.subplot(1, 2, 2)
        anomalies_by_part = results_df[results_df['prediction'] == 1]['part_id'].value_counts()
        anomalies_by_part.plot(kind='bar', alpha=0.7)
        plt.title(f'Anomalies by Part ID - {condition}')
        plt.xlabel('Part ID')
        plt.ylabel('Number of Anomalies')
        plt.xticks(rotation=45)

    plt.tight_layout()
    plot_path = os.path.join(output_base_path, "plots", f"isolation_forest_results_{condition}.png")
    plt.savefig(plot_path)
    plt.close()

    # Plot anomalies over time if timestamp is available
    if 'timestamp' in results_df.columns:
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
        results_df = results_df.sort_values('timestamp')

        plt.figure(figsize=(12, 6))
        plt.scatter(results_df['timestamp'], results_df['anomaly_score'],
                   c=results_df['prediction'], cmap='coolwarm', alpha=0.7)
        plt.title(f'Isolation Forest Anomalies Over Time - {condition}')
        plt.xlabel('Time')
        plt.ylabel('Anomaly Score')
        plt.colorbar(label='Anomaly (1) / Normal (0)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        time_plot_path = os.path.join(output_base_path, "plots", f"isolation_forest_time_{condition}.png")
        plt.savefig(time_plot_path)
        plt.close()

    return results_df, plot_path

def save_isolation_forest_results(results_df, condition):
    """
    Save Isolation Forest results to CSV files
    """
    # Save results
    results_path = os.path.join(output_base_path, "results", f"isolation_forest_results_{condition}.csv")
    results_df.to_csv(results_path, index=False)

    # Save detailed anomaly information
    if len(results_df[results_df['prediction'] == 1]) > 0:
        anomaly_details = results_df[results_df['prediction'] == 1].copy()
        anomaly_path = os.path.join(output_base_path, "results", f"isolation_forest_anomaly_details_{condition}.csv")
        anomaly_details.to_csv(anomaly_path, index=False)

    return results_path

def main():
    # Configuration
    features_path = os.path.join(output_base_path, "audio_features.csv")

    # Load features
    X, metadata, feature_columns = load_features(features_path)

    # Process each condition separately
    for condition in ['lying', 'hanging']:
        print(f"\n{'='*50}")
        print(f"Processing {condition} condition with Isolation Forest")
        print(f"{'='*50}")

        # Train Isolation Forest
        iso_forest, scaler, predictions, scores, metadata_condition = train_isolation_forest(
            X, metadata, condition
        )

        if iso_forest is None:
            continue

        # Plot and save results
        results_df, plot_path = plot_isolation_forest_results(scores, predictions, metadata_condition, condition)
        print(f"Saved plots to: {plot_path}")

        # Save results
        results_path = save_isolation_forest_results(results_df, condition)
        print(f"Saved results to: {results_path}")

        # Print summary
        n_anomalies = np.sum(predictions)
        n_normal = len(predictions) - n_anomalies
        print(f"Found {n_anomalies} anomalies and {n_normal} normal samples")

        # Print anomalous files
        if n_anomalies > 0:
            print("Anomalous files:")
            anomalous_files = results_df[results_df['prediction'] == 1]
            for _, row in anomalous_files.iterrows():
                print(f"  {row['filename']} (score: {row['anomaly_score']:.4f})")

        # Save model
        model_path = os.path.join(output_base_path, "models", f"isolation_forest_{condition}.pkl")
        joblib.dump(iso_forest, model_path)
        print(f"Model saved as {model_path}")

        # Save scaler
        scaler_path = os.path.join(output_base_path, "models", f"scaler_isolation_forest_{condition}.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved as {scaler_path}")

    print(f"\nAll outputs have been saved to: {output_base_path}")

if __name__ == "__main__":
    main()