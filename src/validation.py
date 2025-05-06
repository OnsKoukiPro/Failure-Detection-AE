import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import argparse
import os
import sys

def load_and_preprocess(csv_path):
    """Load CSV and convert array-like columns"""
    try:
        df = pd.read_csv(csv_path)

        # Custom parser for spectral_entropy
        def parse_spectral_entropy(value):
            try:
                if isinstance(value, str):
                    # Clean malformed arrays
                    cleaned = value.strip('[]').replace('\n', ' ').split()
                    return list(map(float, cleaned))
                return value
            except:
                return np.nan

        if 'spectral_entropy' in df.columns:
            df['spectral_entropy'] = df['spectral_entropy'].apply(parse_spectral_entropy)

        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

def analyze_predictions(df, output_dir):
    """Perform comprehensive analysis and visualization"""
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)

    # Basic comparison dataframe
    results = df[['measurement', 'label',
                 'rf_prediction', 'rf_probability',
                 'cnn_prediction', 'cnn_probability']].copy()

    # Add analysis columns
    results['rf_correct'] = results['label'] == results['rf_prediction']
    results['cnn_correct'] = results['label'] == results['cnn_prediction']
    results['models_agree'] = results['rf_prediction'] == results['cnn_prediction']
    results['probability_diff'] = abs(results['rf_probability'] - results['cnn_probability'])

    # Save analysis results
    results.to_csv(os.path.join(output_dir, 'analysis_results.csv'), index=False)

    # Generate visualizations
    plt.figure(figsize=(15, 6))

    # Accuracy comparison
    plt.subplot(1, 2, 1)
    rf_acc = results['rf_correct'].mean()
    cnn_acc = results['cnn_correct'].mean()
    plt.bar(['Random Forest', 'CNN'], [rf_acc, cnn_acc])
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')

    # Probability distribution
    plt.subplot(1, 2, 2)
    plt.scatter(results['rf_probability'], results['cnn_probability'],
               c=results['label'], alpha=0.6)
    plt.colorbar(label='True Label')
    plt.xlabel('RF Probability')
    plt.ylabel('CNN Probability')
    plt.title('Prediction Probability Comparison')

    # Save plots
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()

    # Generate text report
    report = f"""Model Performance Report:

Random Forest:
- Accuracy: {rf_acc:.2%}
- False Positives: {(results['rf_prediction'] > results['label']).sum()}
- False Negatives: {(results['rf_prediction'] < results['label']).sum()}

CNN:
- Accuracy: {cnn_acc:.2%}
- False Positives: {(results['cnn_prediction'] > results['label']).sum()}
- False Negatives: {(results['cnn_prediction'] < results['label']).sum()}

Agreement Analysis:
- Models agree on {results['models_agree'].mean():.2%} of cases
- Both wrong on {(~results['rf_correct'] & ~results['cnn_correct']).sum()} measurements
"""

    # Save report
    with open(os.path.join(output_dir, 'performance_report.txt'), 'w') as f:
        f.write(report)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze model predictions')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to predictions CSV file')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                       help='Output directory for results')
    args = parser.parse_args()

    # Check for matplotlib
    try:
        df = load_and_preprocess(args.csv_path)
        results = analyze_predictions(df, args.output_dir)
        print(f"Analysis complete! Results saved to {args.output_dir}")
    except Exception as e:
        print(f"Critical error: {str(e)}")
        sys.exit(1)

    # Load and analyze data
    df = load_and_preprocess(args.csv_path)
    results = analyze_predictions(df, args.output_dir)

    print(f"Analysis complete! Results saved to {args.output_dir}")
    print("\nSample results:")
    print(results.head())