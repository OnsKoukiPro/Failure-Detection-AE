import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import argparse
import joblib
import os

def load_data(train_csv, test_csv):
    """Load and preprocess data with feature selection"""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Feature selection
    important_features = ['mfcc_1', 'spectral_entropy', 'cwt_energy']

    X_train = train_df[important_features]
    y_train = train_df['label']
    X_test = test_df[important_features]
    y_test = test_df['label']

    return X_train, y_train, X_test, y_test, test_df

def train_random_forest(X_train, y_train):
    """Traditional ML model with hyperparameter tuning"""
    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(class_weight='balanced', random_state=42)
    )

    param_grid = {
        'randomforestclassifier__n_estimators': [100, 200],
        'randomforestclassifier__max_depth': [5, 8, 10],
        'randomforestclassifier__min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1')
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_cnn(X_train, y_train):
    """Deep learning model with improved architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((X_train.shape[1], 1)),
        tf.keras.layers.Conv1D(32, 2, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(1),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.2,
        class_weight={0: 0.28, 1: 0.72}
    )
    return model

def evaluate_model(model, X_test, y_test, model_type='rf'):
    """Comprehensive model evaluation"""
    if model_type == 'rf':
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    elif model_type == 'cnn':
        y_proba = model.predict(X_test).flatten()
        y_pred = (y_proba > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_pred, y_proba

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    args = parser.parse_args()

    # Load data with original test dataframe
    X_train, y_train, X_test, y_test, test_df = load_data(args.train_csv, args.test_csv)

    # Train and evaluate Random Forest
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_pred, rf_proba = evaluate_model(rf_model, X_test, y_test, 'rf')

    # Train and evaluate CNN
    print("\nTraining CNN...")
    cnn_model = train_cnn(X_train.values, y_train.values)
    cnn_pred, cnn_proba = evaluate_model(cnn_model, X_test.values, y_test.values, 'cnn')

    # Create comparison dataframe
    comparison_df = test_df.copy()
    comparison_df['rf_prediction'] = rf_pred
    comparison_df['rf_probability'] = rf_proba
    comparison_df['cnn_prediction'] = cnn_pred
    comparison_df['cnn_probability'] = cnn_proba

    # Save results
    os.makedirs('/app/results/models', exist_ok=True)
    joblib.dump(rf_model, '/app/results/models/rf_model.pkl')
    cnn_model.save('/app/results/models/cnn_model.h5')

    os.makedirs('/app/results/comparison', exist_ok=True)
    comparison_df.to_csv('/app/results/comparison/predictions_comparison.csv', index=False)
    print("\nComparison CSV saved at /app/results/comparison/predictions_comparison.csv")