# split_data.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration
FEATURES_CSV = "/app/results/features/features.csv"
OUTPUT_DIR = "/app/results/splits"
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # Of remaining data after test split
RANDOM_STATE = 42
TIME_SPLIT_DATE = "2018-12-10"  # Date to split between train and test

def plot_class_distribution(y, title, filepath):
    plt.figure()
    pd.Series(y).value_counts(normalize=True).plot(kind='bar')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Proportion')
    plt.xticks([0, 1], ['Non-Failure', 'Failure'], rotation=0)
    plt.savefig(filepath)
    plt.close()

def time_series_split(df):
    """Split data chronologically based on timestamp"""
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    # Split points
    total_samples = len(df_sorted)
    test_split_idx = int(total_samples * (1 - TEST_SIZE - VAL_SIZE))
    val_split_idx = test_split_idx + int(total_samples * VAL_SIZE)

    return {
        'train': df_sorted.iloc[:test_split_idx],
        'val': df_sorted.iloc[test_split_idx:val_split_idx],
        'test': df_sorted.iloc[val_split_idx:]
    }

def stratified_split(df):
    """Stratified random split preserving class ratios"""
    # First split out test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df['label'],
        random_state=RANDOM_STATE
    )

    # Then split val from train
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_SIZE/(1-TEST_SIZE),  # Relative to train_val size
        stratify=train_val_df['label'],
        random_state=RANDOM_STATE
    )

    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

def measurement_split(df):
    """Split by measurement groups (Measurement 1 vs 2)"""
    return {
        'train': df[df['measurement'].str.contains('Measurement-1')],
        'test': df[df['measurement'].str.contains('Measurement-2')]
    }

if __name__ == "__main__":
    # Load features
    df = pd.read_csv(FEATURES_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Ensure datetime type

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate different splits
    splits = {
        'time_series': time_series_split(df),
        'stratified': stratified_split(df),
        'measurement': measurement_split(df)
    }

    # Save splits and plot distributions
    for split_name, split_data in splits.items():
        # Save to CSV
        for dataset in ['train', 'val', 'test']:
            if dataset in split_data:
                split_data[dataset].to_csv(
                    os.path.join(OUTPUT_DIR, f"{split_name}_{dataset}.csv"),
                    index=False
                )

        # Plot class distributions
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, dataset in enumerate(['train', 'val', 'test']):
            if dataset in split_data:
                y = split_data[dataset]['label']
                pd.Series(y).value_counts(normalize=True).plot(
                    kind='bar', ax=axes[i], title=f'{split_name} {dataset}'
                )
                axes[i].set_xticks([0, 1], ['Non-Failure', 'Failure'], rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{split_name}_distributions.png"))
        plt.close()

    # Verification steps
    print("\nVerification:")
    try:
        # Check stratified split distribution
        stratified_train = pd.read_csv(os.path.join(OUTPUT_DIR, "stratified_train.csv"))
        print("\nStratified Train Class Distribution:")
        print(stratified_train['label'].value_counts(normalize=True).round(2))

        # Check time series temporal order
        time_train = pd.read_csv(os.path.join(OUTPUT_DIR, "time_series_train.csv"))
        time_train['timestamp'] = pd.to_datetime(time_train['timestamp'])
        print(f"\nTime Series Train Range: {time_train['timestamp'].min()} to {time_train['timestamp'].max()}")

    except Exception as e:
        print(f"\nVerification error: {str(e)}")

    print(f"\nSplits saved to {OUTPUT_DIR}")