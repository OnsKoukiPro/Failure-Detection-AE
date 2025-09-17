import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
output_base_path = "/app/results/KI2/"
os.makedirs(os.path.join(output_base_path, "comparison"), exist_ok=True)

def load_metadata():
    """Load the metadata file"""
    metadata_path = os.path.join(output_base_path, "metadata.csv")
    if os.path.exists(metadata_path):
        return pd.read_csv(metadata_path)
    else:
        print("Metadata file not found")
        return None

def load_results(condition, model_type):
    """
    Load results for a specific model and condition
    """
    if model_type == "lstm":
        # For LSTM, we need to handle both file types
        details_path = os.path.join(output_base_path, "results", f"anomaly_details_{condition}.csv")
        results_path = os.path.join(output_base_path, "results", f"anomaly_results_{condition}.csv")

        # Check which files exist
        details_exists = os.path.exists(details_path)
        results_exists = os.path.exists(results_path)

        if not details_exists and not results_exists:
            print(f"No LSTM results found for {condition} condition")
            return None

        # Load metadata to get all files for this condition
        metadata = load_metadata()
        if metadata is None:
            return None

        condition_files = metadata[metadata['condition'] == condition]['filename'].unique()

        # Create a base dataframe with all files
        lstm_results = pd.DataFrame({'filename': condition_files})
        lstm_results['is_anomaly'] = False
        lstm_results['reconstruction_error'] = np.nan

        # Add details from anomaly_details file if it exists
        if details_exists:
            details_df = pd.read_csv(details_path)
            for _, row in details_df.iterrows():
                filename = row['filename']
                error = row['reconstruction_error']
                lstm_results.loc[lstm_results['filename'] == filename, 'is_anomaly'] = True
                lstm_results.loc[lstm_results['filename'] == filename, 'reconstruction_error'] = error

        # Add results from anomaly_results file if it exists
        if results_exists:
            results_df = pd.read_csv(results_path)
            # This file doesn't have filenames, so we can't use it directly for comparison
            # But we can use it to get statistics if needed
            pass

        return lstm_results

    else:
        # For Isolation Forest, load the main results file
        results_path = os.path.join(output_base_path, "results", f"isolation_forest_results_{condition}.csv")

        if not os.path.exists(results_path):
            print(f"Results file not found: {results_path}")
            return None

        return pd.read_csv(results_path)

def create_complete_results(condition, lstm_results, iso_forest_results):
    """
    Create a complete comparison DataFrame by merging results from both models
    """
    # Get all unique filenames from both result sets
    all_filenames = set()
    if lstm_results is not None and 'filename' in lstm_results.columns:
        all_filenames.update(lstm_results['filename'].unique())
    if iso_forest_results is not None and 'filename' in iso_forest_results.columns:
        all_filenames.update(iso_forest_results['filename'].unique())

    # Create a base DataFrame with all filenames
    comparison_df = pd.DataFrame({'filename': list(all_filenames)})

    # Merge LSTM results
    if lstm_results is not None and 'filename' in lstm_results.columns:
        comparison_df = pd.merge(comparison_df, lstm_results[['filename', 'reconstruction_error', 'is_anomaly']],
                                on='filename', how='left')
        comparison_df.rename(columns={'reconstruction_error': 'lstm_score', 'is_anomaly': 'lstm_prediction'}, inplace=True)
    else:
        comparison_df['lstm_score'] = np.nan
        comparison_df['lstm_prediction'] = False

    # Merge Isolation Forest results
    if iso_forest_results is not None and 'filename' in iso_forest_results.columns:
        # Check if the required columns exist
        if 'anomaly_score' not in iso_forest_results.columns:
            iso_forest_results.rename(columns={iso_forest_results.columns[1]: 'anomaly_score'}, inplace=True)
        if 'prediction' not in iso_forest_results.columns:
            iso_forest_results.rename(columns={iso_forest_results.columns[2]: 'prediction'}, inplace=True)

        comparison_df = pd.merge(comparison_df, iso_forest_results[['filename', 'anomaly_score', 'prediction']],
                                on='filename', how='left')
        comparison_df.rename(columns={'anomaly_score': 'iso_score', 'prediction': 'iso_prediction'}, inplace=True)
    else:
        comparison_df['iso_score'] = np.nan
        comparison_df['iso_prediction'] = False

    # Fill NaN values
    comparison_df['lstm_prediction'] = comparison_df['lstm_prediction'].fillna(False).astype(bool)
    comparison_df['iso_prediction'] = comparison_df['iso_prediction'].fillna(False).astype(bool)

    return comparison_df

def compare_models(condition):
    """
    Compare LSTM Autoencoder and Isolation Forest results for a specific condition
    """
    print(f"\nComparing models for {condition} condition")

    # Load results from both models
    lstm_results = load_results(condition, "lstm")
    iso_forest_results = load_results(condition, "iso_forest")

    if lstm_results is None and iso_forest_results is None:
        print(f"No results found for {condition} condition. Skipping.")
        return None

    # Create complete comparison DataFrame
    comparison_df = create_complete_results(condition, lstm_results, iso_forest_results)

    # Calculate agreement
    agreement = np.mean(comparison_df['lstm_prediction'] == comparison_df['iso_prediction'])
    print(f"Model agreement: {agreement:.2%}")

    # Create confusion matrix between models
    conf_matrix = confusion_matrix(comparison_df['lstm_prediction'].astype(int),
                                  comparison_df['iso_prediction'].astype(int))

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title(f'Model Comparison - {condition}\n(Isolation Forest vs LSTM Autoencoder)')
    plt.xlabel('Isolation Forest Prediction')
    plt.ylabel('LSTM Autoencoder Prediction')
    plt.tight_layout()
    conf_matrix_path = os.path.join(output_base_path, "comparison", f"confusion_matrix_{condition}.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f"Confusion matrix saved to: {conf_matrix_path}")

    # Plot score correlation (only if both scores are available)
    if not comparison_df['lstm_score'].isna().all() and not comparison_df['iso_score'].isna().all():
        plt.figure(figsize=(10, 6))
        valid_idx = ~comparison_df['lstm_score'].isna() & ~comparison_df['iso_score'].isna()
        plt.scatter(comparison_df.loc[valid_idx, 'iso_score'], comparison_df.loc[valid_idx, 'lstm_score'],
                   alpha=0.7, c=comparison_df.loc[valid_idx, 'lstm_prediction'].astype(int) +
                                comparison_df.loc[valid_idx, 'iso_prediction'].astype(int),
                   cmap='viridis')
        plt.xlabel('Isolation Forest Anomaly Score')
        plt.ylabel('LSTM Autoencoder Reconstruction Error')
        plt.title(f'Score Correlation - {condition}')
        plt.colorbar(label='Sum of Predictions (0-2)')
        plt.tight_layout()
        correlation_path = os.path.join(output_base_path, "comparison", f"score_correlation_{condition}.png")
        plt.savefig(correlation_path)
        plt.close()
        print(f"Score correlation plot saved to: {correlation_path}")

    # Analyze cases where models disagree
    disagreements = comparison_df[comparison_df['lstm_prediction'] != comparison_df['iso_prediction']]
    print(f"Number of disagreements: {len(disagreements)}")

    if len(disagreements) > 0:
        print("Sample disagreements:")
        for _, row in disagreements.head().iterrows():
            print(f"  {row['filename']}: LSTM={row['lstm_prediction']}, ISO={row['iso_prediction']}")

    # Save comparison results
    comparison_path = os.path.join(output_base_path, "comparison", f"model_comparison_{condition}.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison results saved to: {comparison_path}")

    return comparison_df

def main():
    # Compare models for each condition
    for condition in ['lying', 'hanging']:
        compare_models(condition)

    print("\nModel comparison completed!")

if __name__ == "__main__":
    main()