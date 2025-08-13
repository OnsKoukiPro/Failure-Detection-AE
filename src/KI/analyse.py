import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from datetime import datetime, timedelta
import joblib

# Additional imports
import shap
import ruptures as rpt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Timedelta):
            return str(obj)
        else:
            return super().default(obj)


# 1. Threshold Optimization & Anomaly Detection
def optimize_thresholds(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    thresholds = {}
    anomaly_flags = pd.DataFrame()

    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    for anchor in df['anchor_point'].unique():
        anchor_df = df[df['anchor_point'] == anchor].copy()

        thresholds[f"{anchor}_percentile"] = {
            "iso": np.percentile(anchor_df['iso_score'], 5),
            "lof": np.percentile(anchor_df['lof_score'], 95),
            "anchor": np.percentile(anchor_df['anchor_score'], 95),
            "ensemble": np.percentile(anchor_df['ensemble_score'], 95)
        }

        if 'label' in df.columns:
            prec, recall, threshs = precision_recall_curve(
                anchor_df['label'],
                -anchor_df['ensemble_score']
            )
            f1_scores = 2 * (prec * recall) / (prec + recall + 1e-12)
            best_idx = np.argmax(f1_scores)
            thresholds[f"{anchor}_optimized"] = {
                "ensemble": threshs[best_idx]
            }

        anchor_threshold = thresholds[f"{anchor}_percentile"]["ensemble"]
        anchor_df['anomaly_flag'] = anchor_df['ensemble_score'] > anchor_threshold
        anomaly_flags = pd.concat([anomaly_flags, anchor_df])

    with open(os.path.join(output_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2, cls=NumpyEncoder)

    anomaly_flags.to_csv(os.path.join(output_dir, "anomaly_detections.csv"), index=False)
    return thresholds, anomaly_flags

def visualize_anomalies(df, thresholds, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(15, 10))
    anchors = df['anchor_point'].unique()
    n_anchors = len(anchors)

    for i, anchor in enumerate(anchors, 1):
        plt.subplot(int(np.ceil(n_anchors / 2)), 2, i)
        anchor_df = df[df['anchor_point'] == anchor]
        sns.histplot(anchor_df['ensemble_score'], kde=True)
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

    plt.figure(figsize=(15, 8))
    for anchor in anchors:
        anchor_df = df[df['anchor_point'] == anchor].sort_values('timestamp')
        plt.plot(
            anchor_df['timestamp'],
            anchor_df['ensemble_score'],
            label=anchor,
            alpha=0.7
        )
        anomalies = anchor_df[anchor_df['anomaly_flag']]
        if not anomalies.empty:
            plt.scatter(
                anomalies['timestamp'],
                anomalies['ensemble_score'],
                color='red',
                s=50,
                label=f'{anchor} Anomalies'
            )

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

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='anchor_point',
        y='ensemble_score',
        data=df,
        showfliers=False
    )
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
    os.makedirs(output_dir, exist_ok=True)
    report = {}

    report["summary"] = {
        "total_samples": len(anomaly_df),
        "total_anomalies": anomaly_df['anomaly_flag'].sum(),
        "anomaly_rate": anomaly_df['anomaly_flag'].mean()
    }

    anchor_stats = anomaly_df.groupby('anchor_point')['anomaly_flag'].agg(['sum', 'mean', 'std'])
    report["by_anchor"] = anchor_stats.to_dict()

    if not pd.api.types.is_datetime64_any_dtype(anomaly_df['timestamp']):
        anomaly_df['timestamp'] = pd.to_datetime(anomaly_df['timestamp'], errors='coerce')

    anomaly_df = anomaly_df.sort_values(['anchor_point', 'timestamp'])
    anomaly_df['time_diff'] = anomaly_df.groupby('anchor_point')['timestamp'].diff()

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

    with open(os.path.join(output_dir, "anomaly_report.json"), "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    cluster_report.to_csv(os.path.join(output_dir, "anomaly_clusters.csv"), index=False)

    if not critical_df.empty:
        critical_df.to_csv(os.path.join(output_dir, "critical_anomalies.csv"), index=False)

    return report


# ========== NEW: SHAP Feature Importance ==========

def shap_feature_importance(model_path, features_csv, output_dir, sample_size=100):
    """
    Compute SHAP values for feature importance using Isolation Forest
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model and data
    model_data = joblib.load(model_path)
    iso_forest = model_data['iso_forest']
    scaler = model_data['scaler']
    feature_names = model_data['feature_columns']

    df = pd.read_csv(features_csv)
    feature_columns = feature_names
    X = df[feature_columns].values
    X_scaled = scaler.transform(X)

    # Use a sample for SHAP due to performance
    if sample_size < len(X_scaled):
        sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[sample_idx]
    else:
        X_sample = X_scaled

    explainer = shap.Explainer(iso_forest, X_sample, feature_names=feature_names)
    shap_values = explainer(X_sample)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features=X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    plt.close()

    # Save raw SHAP values
    np.save(os.path.join(output_dir, "shap_values.npy"), shap_values.values)
    print(f"SHAP feature importance saved to {output_dir}")


# ========== NEW: Multivariate Temporal Analysis ==========

def temporal_correlation_heatmap(df, output_dir):
    """
    Cross-anchor correlation heatmap of ensemble scores to detect systemic issues
    """
    os.makedirs(output_dir, exist_ok=True)
    pivot = df.pivot(index='timestamp', columns='anchor_point', values='ensemble_score')
    corr = pivot.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Cross-Anchor Ensemble Score Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_anchor_correlation.png"))
    plt.close()


def lead_lag_analysis(df, output_dir, max_lag=10):
    """
    Simple lead-lag cross correlation between anchors
    """
    os.makedirs(output_dir, exist_ok=True)
    pivot = df.pivot(index='timestamp', columns='anchor_point', values='ensemble_score').fillna(0)

    anchors = pivot.columns
    lags = range(-max_lag, max_lag + 1)

    lead_lag_results = {}

    for a1 in anchors:
        for a2 in anchors:
            if a1 == a2:
                continue
            corr_vals = []
            for lag in lags:
                shifted = pivot[a2].shift(lag).fillna(0)
                corr = pivot[a1].corr(shifted)
                corr_vals.append(corr)
            lead_lag_results[(a1, a2)] = corr_vals

            plt.figure(figsize=(8, 4))
            plt.plot(lags, corr_vals)
            plt.title(f"Lead-Lag Correlation: {a1} vs {a2}")
            plt.xlabel("Lag")
            plt.ylabel("Correlation")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"lead_lag_{a1}_vs_{a2}.png"))
            plt.close()

    return lead_lag_results


def change_point_detection(df, output_dir):
    """
    Detect change points in ensemble scores over time using ruptures
    """
    os.makedirs(output_dir, exist_ok=True)

    pivot = df.pivot(index='timestamp', columns='anchor_point', values='ensemble_score').fillna(method='ffill').fillna(method='bfill')
    results = {}

    for anchor in pivot.columns:
        signal = pivot[anchor].values
        model = "rbf"
        algo = rpt.Pelt(model=model).fit(signal)
        change_points = algo.predict(pen=5)

        results[anchor] = change_points

        plt.figure(figsize=(10, 4))
        rpt.display(signal, change_points, figsize=(10, 4))
        plt.title(f"Change Point Detection for {anchor}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"change_points_{anchor}.png"))
        plt.close()

    # Save change points to JSON
    with open(os.path.join(output_dir, "change_points.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def time_series_clustering(df, output_dir, n_clusters=3):
    """
    Cluster anomaly score time series using DTW KMeans (tslearn)
    """
    os.makedirs(output_dir, exist_ok=True)
    pivot = df.pivot(index='timestamp', columns='anchor_point', values='ensemble_score').fillna(0)

    # Normalize time series
    scaler = TimeSeriesScalerMeanVariance()
    ts_data = scaler.fit_transform(pivot.T.values)  # shape: (anchors, timestamps, 1)

    # Cluster
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
    labels = model.fit_predict(ts_data)

    cluster_map = dict(zip(pivot.columns, labels))

    # Plot clustering result
    plt.figure(figsize=(10, 6))
    for label in range(n_clusters):
        members = [anchor for anchor, cluster in cluster_map.items() if cluster == label]
        for anchor in members:
            plt.plot(pivot.index, pivot[anchor], label=f"{anchor} (Cluster {label})")
    plt.title("Time Series Clustering of Ensemble Scores by Anchor")
    plt.xlabel("Timestamp")
    plt.ylabel("Ensemble Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_series_clustering.png"))
    plt.close()

    # Save cluster assignments
    with open(os.path.join(output_dir, "ts_clusters.json"), "w") as f:
        json.dump(cluster_map, f, indent=2, cls=NumpyEncoder)

    return cluster_map


def main():
    INPUT_CSV = "/app/results/KI/models/model_predictions.csv"
    OUTPUT_DIR = "/app/results/KI/analysis"
    MODEL_PATH = "/app/results/KI/models/ensemble_model.pkl"
    FEATURES_CSV = "/app/results/KI/features.csv"

    # Load model predictions
    df = pd.read_csv(INPUT_CSV)

    # 1. Threshold optimization and anomaly detection
    thresholds, anomaly_df = optimize_thresholds(df, OUTPUT_DIR)

    # 2. Visualization of anomalies and patterns
    visualize_anomalies(anomaly_df, thresholds, OUTPUT_DIR)

    # 3. Failure pattern analysis
    report = analyze_failure_patterns(anomaly_df, OUTPUT_DIR)

    # 4. SHAP feature importance
    shap_feature_importance(MODEL_PATH, FEATURES_CSV, OUTPUT_DIR)

    # 5. Multivariate temporal analysis
    temporal_correlation_heatmap(anomaly_df, OUTPUT_DIR)
    lead_lag_analysis(anomaly_df, OUTPUT_DIR)
    change_point_detection(anomaly_df, OUTPUT_DIR)
    time_series_clustering(anomaly_df, OUTPUT_DIR)

    print(f"Analysis complete. Results saved to {OUTPUT_DIR}")
    print(f"Total anomalies detected: {report['summary']['total_anomalies']}")

if __name__ == "__main__":
    main()
