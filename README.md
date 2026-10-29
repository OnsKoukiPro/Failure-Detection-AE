<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OnsKoukiPro/Failure-Detection-AE/main/.github/assets/banner-dark.svg?v=r1"><img src="https://raw.githubusercontent.com/OnsKoukiPro/Failure-Detection-AE/main/.github/assets/banner-light.svg?v=r1" alt="Acoustic Failure: listens to the acoustic emissions of a part under test and flags the one that is about to fail" width="100%"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OnsKoukiPro/Failure-Detection-AE/main/.github/assets/stats-dark.svg?v=r1"><img src="https://raw.githubusercontent.com/OnsKoukiPro/Failure-Detection-AE/main/.github/assets/stats-light.svg?v=r1" alt="200 recordings scored, 33 features each, 3 detectors plus an ensemble, 12 flagged" width="100%"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OnsKoukiPro/Failure-Detection-AE/main/.github/assets/s1-dark.svg?v=r1"><img src="https://raw.githubusercontent.com/OnsKoukiPro/Failure-Detection-AE/main/.github/assets/s1-light.svg?v=r1" alt="01 / Pipeline: from a sound to a verdict" width="100%"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OnsKoukiPro/Failure-Detection-AE/main/.github/assets/pipeline-dark.svg?v=r1"><img src="https://raw.githubusercontent.com/OnsKoukiPro/Failure-Detection-AE/main/.github/assets/pipeline-light.svg?v=r1" alt="WAV recordings are preprocessed, turned into time, frequency and time-frequency features, scored by IsolationForest, LOF and per-anchor models, then combined in an ensemble" width="100%"></picture>

Each recording is cleaned (normalise, band-pass, trim, mono) and turned into time-domain, spectral, MFCC and wavelet features. Unsupervised detectors score each one, and an ensemble with a per-impact-point threshold makes the call. Supervised paths (random forest, CNN on spectrograms, LSTM) live in `src/` and `src/KI2/`.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OnsKoukiPro/Failure-Detection-AE/main/.github/assets/s2-dark.svg?v=r1"><img src="https://raw.githubusercontent.com/OnsKoukiPro/Failure-Detection-AE/main/.github/assets/s2-light.svg?v=r1" alt="02 / Results: what it heard" width="100%"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OnsKoukiPro/Failure-Detection-AE/main/.github/assets/lanes-dark.svg?v=r1"><img src="https://raw.githubusercontent.com/OnsKoukiPro/Failure-Detection-AE/main/.github/assets/lanes-light.svg?v=r1" alt="Ensemble anomaly score per impact point over 200 recordings; 12 flagged above the threshold" width="100%"></picture>

<details>
<summary><b>More plots</b></summary>

| | |
|---|---|
| <img src="results/KI/analysis/temporal_anomalies.png" alt="Temporal anomalies"> | <img src="results/KI/analysis/score_distributions.png" alt="Score distributions"> |
| <img src="results/KI/analysis/cross_anchor_correlation.png" alt="Cross-anchor correlation"> | <img src="results/KI/analysis/shap_summary.png" alt="SHAP feature importance"> |

</details>

<details>
<summary><b>Run it</b></summary>

Everything runs in Docker through the `makefile`, one target per step:

```bash
make map-ki        # 1. link each WAV to its metadata
make process-ki    # 2. normalise, filter, trim
make help          # all steps: features, training, validation, analysis
```

Labels: `good` is 0, `Frequency not good` and `Quotient not good` are 1. Splits are chronological (80 / 10 / 10) and stratified for the class imbalance.

</details>
