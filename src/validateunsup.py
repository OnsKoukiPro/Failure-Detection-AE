import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)
import os

# Configuration
ANOMALY_RESULTS_PATH = "/app/results/models/anomaly_detection_results.csv"
ORIGINAL_MAPPED_PATH = "/app/results/mapped_dataset.csv"
COMPARISON_DIR = "/app/results/comparison"
PLOTS_DIR = "/app/results/plots"

# Create output directories
os.makedirs(COMPARISON_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_and_prepare_data():
    """Load and merge anomaly results with original mapped data"""
    # Load anomaly detection results
    anomaly_df = pd.read_csv(ANOMALY_RESULTS_PATH)

    # Convert timestamp to match original format
    anomaly_df['timestamp'] = pd.to_datetime(anomaly_df['timestamp'])

    # Load original mapped data
    original_df = pd.read_csv(ORIGINAL_MAPPED_PATH)
    original_df['timestamp'] = pd.to_datetime(original_df['timestamp'])

    # Create a composite key for merging
    anomaly_df['composite_key'] = anomaly_df['measurement'] + '_' + anomaly_df['timestamp'].astype(str)
    original_df['composite_key'] = original_df['measurement'] + '_' + original_df['timestamp'].astype(str)

    # Merge datasets
    merged_df = anomaly_df.merge(
        original_df[['composite_key', 'csv_label', 'info_1', 'quality_folder']],
        on='composite_key',
        how='left'
    )

    # Clean up
    merged_df.drop(columns=['composite_key'], inplace=True)

    return merged_df

def calculate_metrics(df):
    """Calculate performance metrics and confusion matrix"""
    # Convert labels to binary format
    df['true_label'] = df['csv_label'].apply(lambda x: 0 if x == "good" else 1)

    # Confusion matrix
    cm = confusion_matrix(df['true_label'], df['is_anomaly'])
    tn, fp, fn, tp = cm.ravel()

    # Metrics dictionary
    metrics = {
        'confusion_matrix': cm,
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'true_negative_rate': tn / (tn + fp) if (tn + fp) > 0 else 0
    }

    return metrics

def generate_visualizations(df, metrics):
    """Create visualizations comparing results"""
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'],
                annot=True, fmt='d',
                cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Good', 'Bad'])
    plt.title('Confusion Matrix: Anomaly Detection vs True Labels')
    plt.xlabel('Predicted Anomaly')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(df['true_label'], df['anomaly_score'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'), dpi=300)
    plt.close()

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(df['true_label'], df['anomaly_score'])
    avg_precision = average_precision_score(df['true_label'], df['anomaly_score'])

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkgreen', lw=2,
             label=f'Precision-Recall (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig(os.path.join(PLOTS_DIR, 'precision_recall_curve.png'), dpi=300)
    plt.close()

    # 4. Anomaly Score Distribution by True Label
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='csv_label', y='anomaly_score', data=df)
    plt.title('Anomaly Score Distribution by True Label')
    plt.xlabel('True Label')
    plt.ylabel('Anomaly Score')
    plt.savefig(os.path.join(PLOTS_DIR, 'score_by_label.png'), dpi=300)
    plt.close()

    # 5. Cluster Distribution by True Label
    plt.figure(figsize=(10, 6))
    sns.countplot(x='cluster', hue='csv_label', data=df)
    plt.title('Cluster Distribution by True Label')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title='True Label')
    plt.savefig(os.path.join(PLOTS_DIR, 'cluster_by_label.png'), dpi=300)
    plt.close()

def generate_report(df, metrics):
    """Generate detailed comparison report"""
    report = f"""Anomaly Detection Performance Report
============================================

Dataset Summary:
- Total samples: {len(df)}
- True "good" samples: {len(df[df['csv_label'] == "good"])}
- True "bad" samples: {len(df[df['csv_label'] == "bad"])}
- Detected anomalies: {df['is_anomaly'].sum()}

Performance Metrics:
- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1 Score: {metrics['f1_score']:.4f}
- False Positive Rate: {metrics['false_positive_rate']:.4f}
- True Negative Rate: {metrics['true_negative_rate']:.4f}

Confusion Matrix:
                Predicted Normal  Predicted Anomaly
Actual Good        {metrics['confusion_matrix'][0, 0]}               {metrics['confusion_matrix'][0, 1]}
Actual Bad         {metrics['confusion_matrix'][1, 0]}               {metrics['confusion_matrix'][1, 1]}

Top Anomalies Missed (False Negatives):
"""
    # Get top false negatives (high anomaly scores but not flagged as anomaly)
    false_negatives = df[(df['true_label'] == 1) & (~df['is_anomaly'])]
    false_negatives = false_negatives.sort_values('anomaly_score', ascending=True).head(10)

    if not false_negatives.empty:
        report += false_negatives[['measurement', 'timestamp', 'anomaly_score', 'info_1']].to_string(index=False)
    else:
        report += "None found"

    report += "\n\nTop False Positives:"
    # Get top false positives
    false_positives = df[(df['true_label'] == 0) & (df['is_anomaly'])]
    false_positives = false_positives.sort_values('anomaly_score', ascending=False).head(10)

    if not false_positives.empty:
        report += "\n" + false_positives[['measurement', 'timestamp', 'anomaly_score', 'info_1']].to_string(index=False)
    else:
        report += "\nNone found"

    with open(os.path.join(COMPARISON_DIR, 'performance_report.txt'), 'w') as f:
        f.write(report)

    df.to_csv(os.path.join(COMPARISON_DIR, 'detailed_comparison.csv'), index=False)

    return report

def main():
    print("Loading and preparing data...")
    df = load_and_prepare_data()

    print("Calculating performance metrics...")
    metrics = calculate_metrics(df)

    print("Generating visualizations...")
    generate_visualizations(df, metrics)

    print("Generating report...")
    report = generate_report(df, metrics)

    print("\n" + "="*50)
    print("COMPARISON COMPLETE")
    print("="*50)
    print(report)
    print("\nOutputs saved to:")
    print(f"- Detailed comparison CSV: {os.path.join(COMPARISON_DIR, 'detailed_comparison.csv')}")
    print(f"- Performance report: {os.path.join(COMPARISON_DIR, 'performance_report.txt')}")
    print(f"- Visualizations: {PLOTS_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()