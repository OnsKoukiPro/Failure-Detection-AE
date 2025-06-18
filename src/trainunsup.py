import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import joblib
import os
import ast

FEATURES_PATH = "/app/results/features/unsupervised_features.csv"
OUTPUT_DIR = "/app/results/models"
PLOTS_DIR = "/app/results/plots"
MODEL_NAME = "ae_anomaly_detector"
CONTAMINATION = 0.05
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def clean_feature_values(value):
    """Convert string representations to actual numerical values"""
    if isinstance(value, str):
        if value.startswith('[') and value.endswith(']'):
            try:
                return ast.literal_eval(value)[0]
            except:
                return 0.0
        try:
            return float(value)
        except:
            return 0.0
    return value

def load_and_preprocess_data():
    """Load features and prepare for modeling"""
    print("Loading features...")
    feature_df = pd.read_csv(FEATURES_PATH)

    feature_columns = [col for col in feature_df.columns
                      if col not in ["measurement", "timestamp", "label"]]

    for col in feature_columns:
        feature_df[col] = feature_df[col].apply(clean_feature_values)

    metadata_cols = ["measurement", "timestamp"]
    if "label" in feature_df.columns:
        metadata_cols.append("label")
    metadata = feature_df[metadata_cols]

    # Extract features (exclude metadata)
    X = feature_df.drop(columns=metadata_cols)

    print(f"Original features shape: {X.shape}")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "feature_scaler.pkl"))

    return X_scaled, metadata, X.columns.tolist()

def visualize_feature_space(X_reduced, labels=None, filename="pca_projection.png"):
    """Visualize reduced feature space"""
    plt.figure(figsize=(12, 8))

    if labels is None:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5, s=10)
        plt.title("PCA Projection of Acoustic Features")
    else:
        unique_labels = np.unique(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                X_reduced[mask, 0],
                X_reduced[mask, 1],
                alpha=0.5,
                s=10,
                color=colors[i],
                label=f'Cluster {label}'
            )
        plt.title("PCA Projection with Cluster Labels")
        plt.legend()

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close()

def train_unsupervised_model(X_reduced):
    """Train unsupervised models"""
    print("\nTraining unsupervised models...")

    # Approach 1: Isolation Forest
    iso_forest = IsolationForest(
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=200,
        verbose=0
    )
    iso_forest.fit(X_reduced)
    iso_scores = iso_forest.decision_function(X_reduced)
    iso_anomalies = iso_scores < np.quantile(iso_scores, CONTAMINATION)

    # Approach 2: DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    clusters = dbscan.fit_predict(X_reduced)

    # Evaluate clustering
    if len(np.unique(clusters)) > 1:
        silhouette = silhouette_score(X_reduced, clusters)
        print(f"DBSCAN Silhouette Score: {silhouette:.4f}")

    joblib.dump(iso_forest, os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_isolation_forest.pkl"))
    joblib.dump(dbscan, os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_dbscan.pkl"))

    return {
        "iso_scores": iso_scores,
        "iso_anomalies": iso_anomalies,
        "clusters": clusters
    }

def analyze_results(metadata, results, feature_names):
    """Combine results with metadata and perform analysis"""
    print("\nAnalyzing results...")

    results_df = metadata.copy()
    results_df["anomaly_score"] = results["iso_scores"]
    results_df["is_anomaly"] = results["iso_anomalies"]
    results_df["cluster"] = results["clusters"]

    results_df.to_csv(os.path.join(OUTPUT_DIR, "anomaly_detection_results.csv"), index=False)

    # Analysis 1: Anomaly distribution by cluster
    plt.figure(figsize=(10, 6))
    sns.countplot(
        x="cluster",
        hue="is_anomaly",
        data=results_df,
        palette={True: "red", False: "blue"}
    )
    plt.title("Anomaly Distribution Across Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.legend(title="Anomaly", labels=["Normal", "Anomaly"])
    plt.savefig(os.path.join(PLOTS_DIR, "anomaly_cluster_distribution.png"), dpi=300)
    plt.close()

    # Analysis 2: Anomaly score distribution (FIXED)
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=results_df,
        x="anomaly_score",
        bins=50,
        kde=True,
        hue="is_anomaly",
        palette={True: "red", False: "blue"},
        element="step"
    )
    plt.axvline(
        x=np.quantile(results_df["anomaly_score"], CONTAMINATION),
        color="r",
        linestyle="--",
        label="Anomaly Threshold"
    )
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "anomaly_score_distribution.png"), dpi=300)
    plt.close()

    # Analysis 3: Cluster characteristics
    if "label" in results_df.columns:
        cluster_summary = results_df.groupby("cluster").agg({
            "label": ["mean", "count"],
            "anomaly_score": "mean"
        })
        print("\nCluster Summary with Failure Labels:")
        print(cluster_summary)

    return results_df

def main():
    X_scaled, metadata, feature_names = load_and_preprocess_data()

    print("\nApplying dimensionality reduction...")
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_scaled)
    print(f"Reduced from {X_scaled.shape[1]} to {X_reduced.shape[1]} dimensions")

    joblib.dump(pca, os.path.join(OUTPUT_DIR, "pca_model.pkl"))

    visualize_feature_space(X_reduced, filename="pca_initial.png")

    results = train_unsupervised_model(X_reduced)

    visualize_feature_space(X_reduced, results["clusters"], filename="pca_clusters.png")

    results_df = analyze_results(metadata, results, feature_names)

    print("\n" + "="*50)
    print("Anomaly Detection Complete!")
    print(f"Total measurements: {len(results_df)}")
    print(f"Anomalies detected: {results_df['is_anomaly'].sum()} ({results_df['is_anomaly'].mean()*100:.2f}%)")
    print(f"Results saved to: {os.path.join(OUTPUT_DIR, 'anomaly_detection_results.csv')}")
    print("="*50)

if __name__ == "__main__":
    main()