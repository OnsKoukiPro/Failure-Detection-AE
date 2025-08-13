import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.covariance import MinCovDet

def train_unsupervised_models(features_csv, output_dir, contamination=0.05):
    """
    Train unsupervised anomaly detection models with enhanced stability
    and covariance handling
    """
    # Load features and metadata
    df = pd.read_csv(features_csv)
    feature_columns = [col for col in df.columns if col not in
                      ['filepath', 'anchor_point', 'timestamp']]
    X = df[feature_columns].values
    metadata = df[['filepath', 'anchor_point', 'timestamp']]

    # Global standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === 1. Enhanced Isolation Forest ===
    iso_forest = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        max_features=0.6,  # More conservative to handle correlations
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    iso_forest.fit(X_scaled)
    print("Isolation Forest training complete")

    # === 2. Robust Anchor-specific Models ===
    anchor_models = {}
    anchor_pipelines = {}

    for anchor in ["Anschlagpunkt1", "Anschlagpunkt2", "Anschlagpunkt3", "Anschlagpunkt4"]:
        anchor_mask = metadata["anchor_point"] == anchor
        X_anchor = X_scaled[anchor_mask]

        if len(X_anchor) < 10:  # Minimum sample threshold
            print(f"Skipping {anchor} - insufficient samples ({len(X_anchor)})")
            continue

        # Feature selection to remove constant features
        selector = VarianceThreshold(threshold=0.01)  # Remove near-constant features
        try:
            X_anchor_selected = selector.fit_transform(X_anchor)

            # Handle potential low-rank after selection
            n_features = X_anchor_selected.shape[1]
            n_components = min(10, n_features - 1) if n_features > 1 else n_features

            if n_components < 1:
                print(f"Skipping {anchor} - no valid features after selection")
                continue

            # Dimensionality reduction
            pca = PCA(n_components=n_components, svd_solver='arpack')
            X_anchor_pca = pca.fit_transform(X_anchor_selected)

            # Robust covariance estimation
            try:
                model = EllipticEnvelope(
                    contamination=contamination,
                    assume_centered=False,
                    support_fraction=0.85
                )
                model.fit(X_anchor_pca)
            except:
                # Fallback to Minimum Covariance Determinant
                model = MinCovDet(support_fraction=0.85)
                model.fit(X_anchor_pca)

            anchor_models[anchor] = model
            anchor_pipelines[anchor] = (selector, pca)
            print(f"Elliptic Envelope trained for {anchor} ({len(X_anchor)} samples)")

        except Exception as e:
            print(f"Error training {anchor} model: {str(e)}")

    # === 3. Regularized LOF ===
    # Regularized covariance matrix
    cov = np.cov(X_scaled.T) + 1e-4 * np.eye(X_scaled.shape[1])
    VI = np.linalg.pinv(cov)  # Pseudo-inverse for stability

    lof = LocalOutlierFactor(
        n_neighbors=min(35, len(X_scaled)//2),  # Adaptive neighborhood size
        contamination=contamination,
        novelty=True,
        metric='mahalanobis',
        metric_params={'VI': VI},
        n_jobs=-1
    )
    lof.fit(X_scaled)
    print("LOF training complete with regularized Mahalanobis metric")

    # === Create Ensemble Predictions ===
    iso_scores = iso_forest.decision_function(X_scaled)
    lof_scores = -lof.negative_outlier_factor_
    anchor_scores = np.zeros(len(X_scaled))

    for anchor, model in anchor_models.items():
        anchor_mask = metadata["anchor_point"] == anchor
        selector, pca = anchor_pipelines[anchor]

        try:
            X_anchor = X_scaled[anchor_mask]
            X_anchor_selected = selector.transform(X_anchor)
            X_anchor_pca = pca.transform(X_anchor_selected)

            scores = model.decision_function(X_anchor_pca)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
            anchor_scores[anchor_mask] = scores
        except Exception as e:
            print(f"Error scoring {anchor}: {str(e)}")

    iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
    lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())

    ensemble_scores = (
        0.4 * iso_scores_norm +
        0.3 * lof_scores_norm +
        0.3 * anchor_scores
    )

    # Final normalization
    ensemble_scores = (ensemble_scores - ensemble_scores.min()) / \
                     (ensemble_scores.max() - ensemble_scores.min())

    # === Save Models & Results ===
    os.makedirs(output_dir, exist_ok=True)

    # Save models
    joblib.dump({
        'iso_forest': iso_forest,
        'anchor_models': anchor_models,
        'anchor_pipelines': anchor_pipelines,
        'lof': lof,
        'scaler': scaler,
        'feature_columns': feature_columns  # Save feature names for SHAP
    }, os.path.join(output_dir, "ensemble_model.pkl"))

    results_df = metadata.copy()
    results_df['iso_score'] = iso_scores_norm
    results_df['lof_score'] = lof_scores_norm
    results_df['anchor_score'] = anchor_scores
    results_df['ensemble_score'] = ensemble_scores
    results_df.to_csv(os.path.join(output_dir, "model_predictions.csv"), index=False)

    print(f"Trained models and predictions saved to {output_dir}")
    return results_df

if __name__ == "__main__":
    train_unsupervised_models(
        features_csv="/app/results/KI/features.csv",
        output_dir="/app/results/KI/models",
        contamination=0.05
    )
