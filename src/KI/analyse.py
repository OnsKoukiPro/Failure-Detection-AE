import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from datetime import datetime, timedelta


import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):  # ✅ NEW
            return obj.isoformat()
        elif isinstance(obj, pd.Timedelta):  # ✅ Also useful
            return str(obj)
        else:
            return super().default(obj)


# 1. Threshold Optimization & Anomaly Detection
def optimize_thresholds(df, output_dir):
    """
    Determine optimal thresholds per anchor point and detection method
    """
    os.makedirs(output_dir, exist_ok=True)
    thresholds = {}
    anomaly_flags = pd.DataFrame()

    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    for anchor in df['anchor_point'].unique():
        anchor_df = df[df['anchor_point'] == anchor].copy()

        # Method 1: Percentile-based (domain-informed)
        thresholds[f"{anchor}_percentile"] = {
            "iso": np.percentile(anchor_df['iso_score'], 5),
            "lof": np.percentile(anchor_df['lof_score'], 95),
            "anchor": np.percentile(anchor_df['anchor_score'], 95),
            "ensemble": np.percentile(anchor_df['ensemble_score'], 95)
        }

        # Method 2: Precision-Recall optimization (if partial labels exist)
        if 'label' in df.columns:
            prec, recall, threshs = precision_recall_curve(
                anchor_df['label'],
                -anchor_df['ensemble_score']  # Higher score = more anomalous
            )
            f1_scores = 2 * (prec * recall) / (prec + recall + 1e-12)
            best_idx = np.argmax(f1_scores)
            thresholds[f"{anchor}_optimized"] = {
                "ensemble": threshs[best_idx]
            }

        # Apply thresholds (using conservative ensemble approach)
        anchor_threshold = thresholds[f"{anchor}_percentile"]["ensemble"]
        anchor_df['anomaly_flag'] = anchor_df['ensemble_score'] > anchor_threshold
        anomaly_flags = pd.concat([anomaly_flags, anchor_df])

    # Save thresholds and anomaly flags
    with open(os.path.join(output_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2, cls=NumpyEncoder)

    anomaly_flags.to_csv(os.path.join(output_dir, "anomaly_detections.csv"), index=False)
    return thresholds, anomaly_flags

def visualize_anomalies(df, thresholds, output_dir):
    """Generate diagnostic visualizations"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Score distributions with thresholds
    plt.figure(figsize=(15, 10))
    anchors = df['anchor_point'].unique()
    n_anchors = len(anchors)

    for i, anchor in enumerate(anchors, 1):
        plt.subplot(int(np.ceil(n_anchors / 2)), 2, i)
        anchor_df = df[df['anchor_point'] == anchor]
        sns.histplot(anchor_df['ensemble_score'], kde=True)

        # Add threshold line
        if f"{anchor}_percentile" in thresholds:
            plt.axvline(
                thresholds[f"{anchor}_percentile"]["ensemble"],
                color='r',
                linestyle='--',
                label='Threshold'
            )

        plt.title(f"{anchor} - Ensemble Score Distribution")
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distributions.png"))
    plt.close()

    # 2. Temporal anomaly progression
    plt.figure(figsize=(15, 8))
    for anchor in anchors:
        anchor_df = df[df['anchor_point'] == anchor].sort_values('timestamp')

        # Plot scores
        plt.plot(
            anchor_df['timestamp'],
            anchor_df['ensemble_score'],
            label=anchor,
            alpha=0.7
        )

        # Plot anomalies
        anomalies = anchor_df[anchor_df['anomaly_flag']]
        if not anomalies.empty:
            plt.scatter(
                anomalies['timestamp'],
                anomalies['ensemble_score'],
                color='red',
                s=50,
                label=f'{anchor} Anomalies'
            )

    # Add reference line
    if anchors.size > 0 and f"{anchors[0]}_percentile" in thresholds:
        plt.axhline(
            thresholds[f"{anchors[0]}_percentile"]["ensemble"],
            color='k',
            linestyle='--',
            label='Reference Threshold'
        )

    plt.legend()
    plt.title("Anomaly Scores Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "temporal_anomalies.png"))
    plt.close()

    # 3. Anchor point comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='anchor_point',
        y='ensemble_score',
        data=df,
        showfliers=False
    )

    # Add anomaly points
    anomalies_df = df[df['anomaly_flag']]
    if not anomalies_df.empty:
        sns.stripplot(
            x='anchor_point',
            y='ensemble_score',
            data=anomalies_df,
            color='red',
            size=5,
            jitter=0.2
        )

    plt.title("Anomaly Score Distribution by Anchor Point")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "anchor_comparison.png"))
    plt.close()

# 2. Failure Pattern Analysis & Reporting
def analyze_failure_patterns(anomaly_df, output_dir):
    """Identify failure patterns and generate reports"""
    os.makedirs(output_dir, exist_ok=True)
    report = {}

    # 1. Basic anomaly statistics
    report["summary"] = {
        "total_samples": len(anomaly_df),
        "total_anomalies": anomaly_df['anomaly_flag'].sum(),
        "anomaly_rate": anomaly_df['anomaly_flag'].mean()
    }

    # Anchor-specific stats
    anchor_stats = anomaly_df.groupby('anchor_point')['anomaly_flag'].agg(['sum', 'mean', 'std'])
    report["by_anchor"] = anchor_stats.to_dict()

    # 2. Temporal clustering of anomalies
    if not pd.api.types.is_datetime64_any_dtype(anomaly_df['timestamp']):
        anomaly_df['timestamp'] = pd.to_datetime(anomaly_df['timestamp'], errors='coerce')

    anomaly_df = anomaly_df.sort_values(['anchor_point', 'timestamp'])
    anomaly_df['time_diff'] = anomaly_df.groupby('anchor_point')['timestamp'].diff()

    # Identify anomaly clusters
    consecutive = anomaly_df[anomaly_df['anomaly_flag']].copy()
    consecutive['time_gap'] = consecutive['time_diff'] > timedelta(minutes=5)
    consecutive['cluster_id'] = consecutive.groupby('anchor_point')['time_gap'].cumsum()

    cluster_report = consecutive.groupby(['anchor_point', 'cluster_id']).agg(
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        duration=('timestamp', lambda x: (x.max() - x.min()).total_seconds()/60),
        anomaly_count=('cluster_id', 'count'),
        avg_score=('ensemble_score', 'mean'),
        max_score=('ensemble_score', 'max')
    ).reset_index()

    report["anomaly_clusters"] = cluster_report.to_dict(orient='records')

    # 3. Critical anomaly identification
    critical_anomalies = []
    for anchor in anchor_stats.index:
        anchor_anomalies = anomaly_df[
            (anomaly_df['anchor_point'] == anchor) &
            (anomaly_df['anomaly_flag'])
        ]
        if len(anchor_anomalies) > 0:
            anchor_mean = anchor_anomalies['ensemble_score'].mean()
            anchor_std = anchor_anomalies['ensemble_score'].std()
            critical = anchor_anomalies[
                anchor_anomalies['ensemble_score'] > anchor_mean + 2*anchor_std
            ]
            critical_anomalies.append(critical)

    if critical_anomalies:
        critical_df = pd.concat(critical_anomalies).sort_values('ensemble_score', ascending=False)
    else:
        critical_df = pd.DataFrame()

    # 4. Save reports
    with open(os.path.join(output_dir, "anomaly_report.json"), "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    cluster_report.to_csv(os.path.join(output_dir, "anomaly_clusters.csv"), index=False)

    if not critical_df.empty:
        critical_df.to_csv(os.path.join(output_dir, "critical_anomalies.csv"), index=False)

    return report

# Main execution function
def main():
    # Configuration
    INPUT_CSV = "/app/results/KI/models/model_predictions.csv"  # From training step
    OUTPUT_DIR = "/app/results/KI/analysis"

    # Load model predictions
    df = pd.read_csv(INPUT_CSV)

    # 1. Threshold optimization and anomaly detection
    thresholds, anomaly_df = optimize_thresholds(df, OUTPUT_DIR)

    # 2. Visualization
    visualize_anomalies(anomaly_df, thresholds, OUTPUT_DIR)

    # 3. Failure pattern analysis
    report = analyze_failure_patterns(anomaly_df, OUTPUT_DIR)

    print(f"Analysis complete. Results saved to {OUTPUT_DIR}")
    print(f"Total anomalies detected: {report['summary']['total_anomalies']}")

if __name__ == "__main__":
    main()