# test_model.py
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import argparse
import os

def test_model(model_path, test_csv, model_type):
    """Test model with type handling"""
    # Load model
    if model_type == 'rf':
        model = joblib.load(model_path)
    elif model_type == 'cnn':
        model = tf.keras.models.load_model(model_path)
    else:
        raise ValueError("Invalid model_type. Choose 'rf' or 'cnn'")

    # Load and prepare data
    test_df = pd.read_csv(test_csv)
    X_test = test_df[['mfcc_1', 'spectral_entropy', 'cwt_energy']]
    y_test = test_df['label']

    # Predictions
    if model_type == 'rf':
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    elif model_type == 'cnn':
        # Reshape for CNN input
        X_test_cnn = X_test.values.reshape(-1, X_test.shape[1], 1)
        y_proba = model.predict(X_test_cnn).flatten()
        y_pred = (y_proba > 0.5).astype(int)

    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nROC AUC: {roc_auc_score(y_test, y_proba):.2f}")

    return y_proba

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['rf', 'cnn'], required=True)
    args = parser.parse_args()

    # Set model path based on type
    model_dir = '/app/results/models'
    if args.model_type == 'rf':
        model_path = os.path.join(model_dir, 'failure_detector.pkl')
    else:
        model_path = os.path.join(model_dir, 'failure_cnn.h5')

    # Test splits
    splits = [
        '/app/results/splits/stratified_test.csv', #Stratified random split preserving class ratios
        '/app/results/splits/time_series_test.csv', #Split data chronologically based on timestamp
        '/app/results/splits/measurement_test.csv'  #Split by measurement groups (Measurement 1 vs 2)
    ]

    for split in splits:
        if os.path.exists(split):
            print(f"\n{'='*40}\nTesting {args.model_type} on: {split}\n{'='*40}")
            test_model(model_path, split, args.model_type)
        else:
            print(f"Skipping non-existent split: {split}")