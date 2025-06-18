import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = "/app/results/features/unsupervised_features.csv"
OUTPUT_DIR = "/app/results/eda_rms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)

def clean_feature(value):
    """Convert feature values from string representation to float"""
    if isinstance(value, str):
        try:
            if value.startswith('[') and value.endswith(']'):
                return float(ast.literal_eval(value)[0])
            return float(value)
        except:
            return np.nan
    return value

print("Loading features...")
df = pd.read_csv(FEATURES_PATH)

numerical_cols = [
    'amplitude_mean', 'amplitude_std', 'amplitude_max', 'rms', 'crest_factor',
    'kurtosis', 'skewness', 'zero_crossing_rate', 'duration', 'sample_entropy',
    'hfd', 'spectral_entropy', 'dominant_freq_1', 'dominant_freq_2', 'dominant_freq_3',
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8',
    'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'spectral_centroid',
    'spectrogram_mean', 'spectrogram_std', 'cwt_energy'
]

for col in numerical_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_feature)

numeric_df = df[numerical_cols].dropna()

scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)

if 'label' in df.columns:
    scaled_df['label'] = df['label'].values

# 1. Correlation Heatmap
print("Generating correlation heatmap...")
plt.figure(figsize=(20, 18))
corr_matrix = numeric_df.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

heatmap = sns.heatmap(
    corr_matrix,
    mask=mask,
    vmin=-1, vmax=1,
    cmap='coolwarm',
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    annot_kws={"size": 8}
)
plt.title('Feature Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=300)
plt.close()

# 2. Feature Distributions
print("Generating feature distributions...")
melted_df = pd.melt(scaled_df.drop(columns=['label'] if 'label' in scaled_df else scaled_df))

plt.figure(figsize=(16, 12))
sns.boxplot(x="variable", y="value", data=melted_df)
plt.xticks(rotation=90)
plt.title('Distribution of Standardized Features', fontsize=16)
plt.ylabel('Standardized Value')
plt.xlabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_distributions.png"), dpi=300)
plt.close()

# 3. Pair Plots for Key Features (selecting 6 most important)
print("Identifying important features...")
# Find features with highest absolute correlation with amplitude_mean
target_feature = 'rms' #adjustable depending on target feature
if target_feature in corr_matrix.columns:
    amplitude_corrs = corr_matrix[target_feature].abs().sort_values(ascending=False)
    top_features = amplitude_corrs.index[1:7].tolist()
else:
    # Default to these features if amplitude_mean or target feature not available
    top_features = ['rms', 'crest_factor', 'spectral_entropy',
                   'spectral_centroid', 'cwt_energy', 'zero_crossing_rate']

# Create pair plot
print("Generating pair plots...")
if 'label' in scaled_df.columns:
    pair_plot = sns.pairplot(
        scaled_df[top_features + ['label']],
        hue='label',
        palette={0: "blue", 1: "red"},
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'w'},
        diag_kws={'alpha': 0.7, 'fill': True}
    )
    pair_plot._legend.set_title('Failure Status')
    for t, l in zip(pair_plot._legend.texts, ['Good (0)', 'Bad (1)']):
        t.set_text(l)
else:
    pair_plot = sns.pairplot(
        scaled_df[top_features],
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'w'},
        diag_kws={'alpha': 0.7, 'fill': True}
    )

pair_plot.fig.suptitle('Pairwise Feature Relationships', fontsize=16, y=1.02)
pair_plot.savefig(os.path.join(OUTPUT_DIR, "feature_pair_plot.png"), dpi=300)
plt.close()

# 4. Feature vs. Failure Status
if 'label' in df.columns:
    print("Generating feature vs. failure status plots...")
    feature_comp_dir = os.path.join(OUTPUT_DIR, "feature_vs_failure")
    os.makedirs(feature_comp_dir, exist_ok=True)

    # Convert labels to strings for proper palette mapping
    df['label_str'] = df['label'].astype(str)

    # violin plots for each feature
    for feature in numerical_cols:
        if feature in df.columns:
            plt.figure(figsize=(10, 6))
            sns.violinplot(
                x='label_str',
                y=feature,
                data=df,
                palette={'0': "blue", '1': "red"},
                inner="quartile"
            )
            plt.title(f'{feature} Distribution by Failure Status', fontsize=14)
            plt.xlabel('Failure Status', fontsize=12)
            plt.ylabel(feature, fontsize=12)
            plt.xticks([0, 1], ['Good', 'Bad'])
            plt.tight_layout()
            plt.savefig(os.path.join(feature_comp_dir, f"{feature}_vs_failure.png"), dpi=150)
            plt.close()

# 5. Time-Series Analysis
if 'timestamp' in df.columns:
    print("Generating time-series analysis...")
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_df = df.sort_values('timestamp')

        # time plots for key features
        time_features = ['amplitude_mean', 'spectral_entropy', 'cwt_energy', 'rms']

        plt.figure(figsize=(16, 12))
        for i, feature in enumerate(time_features, 1):
            if feature in time_df.columns:
                plt.subplot(len(time_features), 1, i)
                plt.plot(time_df['timestamp'], time_df[feature], alpha=0.7)
                if 'label' in time_df.columns:
                    # Mark failure points
                    failures = time_df[time_df['label'] == 1]
                    plt.scatter(
                        failures['timestamp'],
                        failures[feature],
                        color='red',
                        s=30,
                        label='Failure' if i==1 else ""
                    )
                plt.title(f'{feature} Over Time', fontsize=12)
                plt.ylabel(feature)
                if i == 1 and 'label' in time_df.columns:
                    plt.legend()
        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "time_series_features.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating time-series plots: {str(e)}")

# 6. 3D Scatter Plot of Key Features
print("Generating 3D scatter plot...")
try:
    from mpl_toolkits.mplot3d import Axes3D

    # Select three most relevant features
    top_3d_features = top_features[:3] if len(top_features) >= 3 else top_features

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    if 'label' in scaled_df.columns:
        # Plot good and bad separately
        good = scaled_df[scaled_df['label'] == 0]
        bad = scaled_df[scaled_df['label'] == 1]

        ax.scatter(
            good[top_3d_features[0]],
            good[top_3d_features[1]],
            good[top_3d_features[2]],
            c='blue',
            alpha=0.6,
            s=30,
            label='Good'
        )
        ax.scatter(
            bad[top_3d_features[0]],
            bad[top_3d_features[1]],
            bad[top_3d_features[2]],
            c='red',
            alpha=0.8,
            s=40,
            label='Bad'
        )
        ax.legend()
    else:
        ax.scatter(
            scaled_df[top_3d_features[0]],
            scaled_df[top_3d_features[1]],
            scaled_df[top_3d_features[2]],
            c=scaled_df.get('cwt_energy', 0.5),  # Color by energy if available
            cmap='viridis',
            alpha=0.6,
            s=30
        )

    ax.set_xlabel(top_3d_features[0])
    ax.set_ylabel(top_3d_features[1])
    ax.set_zlabel(top_3d_features[2])
    plt.title(f"3D Feature Space: {', '.join(top_3d_features)}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3d_feature_space.png"), dpi=300)
    plt.close()
except ImportError:
    print("Skipping 3D plot - mpl_toolkits not available")
except Exception as e:
    print(f"Error generating 3D plot: {str(e)}")

print(f"EDA complete! Results saved to: {OUTPUT_DIR}")