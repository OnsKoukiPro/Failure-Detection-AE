import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
output_base_path = "/app/results/KI2/"
os.makedirs(os.path.join(output_base_path, "ensemble"), exist_ok=True)

def load_model_results(condition):
    """
    Load results from both LSTM and Isolation Forest models
    """
    # Load LSTM results
    lstm_path = os.path.join(output_base_path, "comparison", f"model_comparison_{condition}.csv")
    if not os.path.exists(lstm_path):
        print(f"LSTM results not found for {condition} condition")
        return None

    results_df = pd.read_csv(lstm_path)
    return results_df

def create_confidence_based_ensemble(results_df, condition):
    """
    Create ensemble predictions based on model confidence
    """
    # Calculate confidence measures for each model
    if 'lstm_score' in results_df.columns and not results_df['lstm_score'].isna().all():
        lstm_scores = results_df['lstm_score'].fillna(0)
        max_lstm = np.max(lstm_scores)
        if max_lstm > 0:
            # For LSTM, higher reconstruction error = higher confidence in anomaly
            lstm_confidence = lstm_scores / max_lstm
        else:
            lstm_confidence = np.zeros(len(results_df))
    else:
        lstm_confidence = np.zeros(len(results_df))

    if 'iso_score' in results_df.columns and not results_df['iso_score'].isna().all():
        iso_scores = results_df['iso_score'].fillna(0)
        inverted_iso = -iso_scores
        min_iso, max_iso = np.min(inverted_iso), np.max(inverted_iso)
        if max_iso > min_iso:
            # For Isolation Forest, more negative scores = higher confidence in anomaly
            # We invert and normalize to get a confidence measure
            iso_confidence = (inverted_iso - min_iso) / (max_iso - min_iso)
        else:
            iso_confidence = np.zeros(len(results_df))
    else:
        iso_confidence = np.zeros(len(results_df))

    # Create confidence-weighted predictions
    lstm_preds = results_df.get('lstm_prediction', 0).fillna(0)
    iso_preds = results_df.get('iso_prediction', 0).fillna(0)

    lstm_weighted = lstm_preds * lstm_confidence
    iso_weighted = iso_preds * iso_confidence

    # Combined confidence score
    confidence_scores = lstm_weighted + iso_weighted

    # Check if we have valid confidence scores
    if np.all(np.isnan(confidence_scores)) or (np.max(confidence_scores) - np.min(confidence_scores)) == 0:
        # Fall back to simple majority voting if confidence scores are invalid
        confidence_scores_normalized = (lstm_preds.astype(int) + iso_preds.astype(int)) / 2
        threshold = 0.5  # At least one model predicts anomaly
    else:
        # Normalize confidence scores
        confidence_scores_normalized = (confidence_scores - np.min(confidence_scores)) / (
            np.max(confidence_scores) - np.min(confidence_scores))

        # Determine threshold (using top 10% as anomalies)
        threshold = np.percentile(confidence_scores_normalized, 90)

    # Create ensemble predictions
    ensemble_predictions = (confidence_scores_normalized > threshold).astype(int)

    # Add to results dataframe
    results_df['confidence_score'] = confidence_scores_normalized
    results_df['confidence_prediction'] = ensemble_predictions

    return results_df, threshold

def analyze_ensemble_results(ensemble_df, condition, threshold):
    """
    Analyze and visualize ensemble results
    """
    print(f"\nAnalyzing ensemble results for {condition} condition...")

    # Check if we have valid confidence scores
    if ensemble_df['confidence_score'].isna().all() or ensemble_df['confidence_score'].nunique() == 1:
        print("Warning: Confidence scores are invalid or constant. Using simple majority voting.")
        # Fall back to majority voting
        if 'lstm_prediction' in ensemble_df.columns and 'iso_prediction' in ensemble_df.columns:
            ensemble_df['confidence_prediction'] = (
                (ensemble_df['lstm_prediction'].fillna(0).astype(int) +
                 ensemble_df['iso_prediction'].fillna(0).astype(int)) >= 1
            ).astype(int)

        # Count anomalies
        n_anomalies = np.sum(ensemble_df['confidence_prediction'])
        n_normal = len(ensemble_df) - n_anomalies
        print(f"Using majority voting: {n_anomalies} anomalies detected")

        # Skip plotting since we don't have valid confidence scores
        return ensemble_df

    # Count anomalies
    n_anomalies = np.sum(ensemble_df['confidence_prediction'])
    n_normal = len(ensemble_df) - n_anomalies
    print(f"Ensemble detected {n_anomalies} anomalies and {n_normal} normal samples")

    # Compare with individual models
    if 'lstm_prediction' in ensemble_df.columns and 'iso_prediction' in ensemble_df.columns:
        lstm_anomalies = np.sum(ensemble_df['lstm_prediction'])
        iso_anomalies = np.sum(ensemble_df['iso_prediction'])

        print(f"LSTM detected {lstm_anomalies} anomalies")
        print(f"Isolation Forest detected {iso_anomalies} anomalies")

        # Agreement analysis
        agreement_both = np.sum((ensemble_df['lstm_prediction'] == 1) &
                               (ensemble_df['iso_prediction'] == 1))
        agreement_ensemble_both = np.sum((ensemble_df['confidence_prediction'] == 1) &
                                        (ensemble_df['lstm_prediction'] == 1) &
                                        (ensemble_df['iso_prediction'] == 1))

        print(f"Both models agreed on {agreement_both} anomalies")
        print(f"Ensemble agrees with both models on {agreement_ensemble_both} anomalies")

    # Plot ensemble score distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(ensemble_df['confidence_score'].dropna(), bins=50, alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
    plt.title(f'Confidence Score Distribution - {condition}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot score comparison
    plt.subplot(1, 2, 2)
    if 'lstm_score' in ensemble_df.columns and 'iso_score' in ensemble_df.columns:
        valid_data = ensemble_df.dropna(subset=['lstm_score', 'iso_score', 'confidence_score'])
        if len(valid_data) > 0:
            plt.scatter(valid_data['lstm_score'], valid_data['iso_score'],
                       c=valid_data['confidence_score'], cmap='viridis', alpha=0.7)
            plt.xlabel('LSTM Score (Reconstruction Error)')
            plt.ylabel('Isolation Forest Score (Inverted)')
            plt.title(f'Score Comparison - {condition}')
            plt.colorbar(label='Confidence Score')

    plt.tight_layout()
    plot_path = os.path.join(output_base_path, "ensemble", f"ensemble_results_{condition}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Ensemble plot saved to: {plot_path}")

    return ensemble_df

def save_ensemble_results(ensemble_df, condition):
    """
    Save ensemble results to CSV file
    """
    results_path = os.path.join(output_base_path, "ensemble", f"ensemble_results_{condition}.csv")
    ensemble_df.to_csv(results_path, index=False)
    print(f"Ensemble results saved to: {results_path}")

    # Save detailed anomaly information
    if len(ensemble_df[ensemble_df['confidence_prediction'] == 1]) > 0:
        anomaly_details = ensemble_df[ensemble_df['confidence_prediction'] == 1].copy()
        anomaly_path = os.path.join(output_base_path, "ensemble", f"ensemble_anomaly_details_{condition}.csv")
        anomaly_details.to_csv(anomaly_path, index=False)

    return results_path

def create_ensemble_strategy(condition):
    """
    Main function to create ensemble predictions for a condition
    """
    print(f"\n{'='*50}")
    print(f"Creating ensemble for {condition} condition")
    print(f"{'='*50}")

    # Load model results
    results_df = load_model_results(condition)
    if results_df is None:
        return

    # Create ensemble predictions using confidence-based approach
    ensemble_df, threshold = create_confidence_based_ensemble(results_df, condition)

    # Analyze results
    ensemble_df = analyze_ensemble_results(ensemble_df, condition, threshold)

    # Save results
    results_path = save_ensemble_results(ensemble_df, condition)

    # Print anomalous files
    anomalous_files = ensemble_df[ensemble_df['confidence_prediction'] == 1]
    if len(anomalous_files) > 0:
        print("Ensemble anomalous files:")
        for _, row in anomalous_files.head().iterrows():
            print(f"  {row['filename']} (score: {row['confidence_score']:.3f})")

    return ensemble_df

def main():
    """
    Main function to run ensemble analysis for all conditions
    """
    print("Using confidence-based ensemble method")

    # Process each condition
    for condition in ['lying', 'hanging']:
        create_ensemble_strategy(condition)

    print("\nEnsemble analysis completed!")

if __name__ == "__main__":
    main()