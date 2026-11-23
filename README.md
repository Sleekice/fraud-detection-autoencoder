# Credit Card Fraud Detection with PyOD AutoEncoder

Anomaly detection experiment using PyOD's PyTorch AutoEncoder on the anonymized credit card dataset.

## Data
Download `creditcard.csv` from https://www.kaggle.com/datasets/whenamancodes/fraud-detection and save it as `data/creditcard.csv`. This binary-label dataset has `Time`, `Amount`, `V1` to `V28`, and `Class` columns. The CSV is excluded from GitHub.

## Quick start
Use Python 3.10–3.12 and run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.download_dataset
python -m src.fraud_detection --data data/creditcard.csv --epochs 10
```

For a faster smoke test, use `--epochs 2 --max-normal-train 5000`.

## Outputs
After the real dataset is loaded and training finishes, the script prints the fraud-class precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix. It also writes `outputs/metrics.json` and `outputs/precision_recall_curve.png`. Capture the terminal output as a genuine screenshot and insert it into the Word report.

## Method
1. Stratified 80/20 train/test split; never train on test records.
2. Train an AutoEncoder on up to 50,000 randomly sampled normal-only training transactions.
3. Fit `StandardScaler` on normal training data; apply the same fit to held-out data.
4. Rank transactions by reconstruction-based anomaly scores, and flag scores higher than PyOD's learned threshold. `contamination=0.01` is a modeling choice, not the known fraud prevalence.
5. Evaluate against labeled held-out data. Labels are not model inputs.

The benchmark is not a production fraud detection system. Fraud costs, temporal drift, false positive rates, and changing customer behavior require further study. A randomly shuffled split can overstate future-deployment performance compared with a time-ordered evaluation.

## References
- PyOD AutoEncoder API: https://pyod.readthedocs.io/en/latest/pyod.models.tabular.html
- PyOD installation: https://pyod.readthedocs.io/en/latest/install.html
- Dataset: https://www.kaggle.com/datasets/whenamancodes/fraud-detection
# fraud-detection-autoencoder
