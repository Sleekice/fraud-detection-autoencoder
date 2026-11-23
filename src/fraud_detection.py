"""Detect anomalous credit-card transactions using a PyOD AutoEncoder.

Usage: python -m src.fraud_detection --data data/creditcard.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def run_experiment(data_path: Path, output_dir: Path, max_normal_train: int, epochs: int) -> dict:
    """Train on normal transactions and evaluate fraud detection on held-out data."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Download creditcard.csv from Kaggle "
            "and place it in the data directory."
        )

    frame = pd.read_csv(data_path)
    required = {"Class", "Time", "Amount", *(f"V{i}" for i in range(1, 29))}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Dataset has missing values in required columns")
    if not set(frame["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must contain binary labels: 0 (normal), 1 (fraud)")

    # Labels are excluded from training features and used only for evaluation.
    features = frame.drop(columns=["Class"])
    labels = frame["Class"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.20, random_state=RANDOM_STATE,
        stratify=labels,
    )

    # Use normal-only training to learn typical transaction behavior.
    normal_train = x_train.loc[y_train == 0]
    if max_normal_train > 0 and len(normal_train) > max_normal_train:
        normal_train = normal_train.sample(n=max_normal_train, random_state=RANDOM_STATE)

    # Fit the scaler ONLY on training data to avoid test-set leakage.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(normal_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    model = AutoEncoder(
        contamination=0.01,  # A chosen training-score cutoff, not the real fraud prevalence.
        preprocessing=False,  # Already standardized using training data only.
        hidden_neuron_list=[64, 32],
        epoch_num=epochs,
        batch_size=256,
        lr=0.001,
        dropout_rate=0.1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    model.fit(x_train_scaled)

    # Larger decision_function scores indicate more anomalous observations.
    scores = model.decision_function(x_test_scaled)
    threshold = float(model.threshold_)
    predictions = (scores > threshold).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["Normal", "Fraud"], zero_division=0, output_dict=True,
    )
    results = {
        "dataset_rows": int(len(frame)),
        "dataset_fraud_rows": int(labels.sum()),
        "training_normal_rows": int(len(normal_train)),
        "test_rows": int(len(y_test)),
        "test_fraud_rows": int(y_test.sum()),
        "epochs": epochs,
        "anomaly_threshold": threshold,
        "confusion_matrix_0_normal_1_fraud": matrix.tolist(),
        "precision_fraud": float(report["Fraud"]["precision"]),
        "recall_fraud": float(report["Fraud"]["recall"]),
        "f1_fraud": float(report["Fraud"]["f1-score"]),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc_average_precision": float(average_precision_score(y_test, scores)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    precision, recall, _ = precision_recall_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(recall, precision, linewidth=2)
    ax.set(xlabel="Recall", ylabel="Precision", title="AutoEncoder — Precision–Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close(fig)

    print("\n=== PYOD AUTOENCODER FRAUD DETECTION RESULTS ===")
    print(f"Dataset: {results['dataset_rows']:,} transactions; "
          f"{results['dataset_fraud_rows']:,} labeled frauds")
    print(f"Training: {results['training_normal_rows']:,} normal transactions")
    print(f"Test: {results['test_rows']:,} transactions; {results['test_fraud_rows']:,} frauds")
    print(f"Fraud precision: {results['precision_fraud']:.4f}")
    print(f"Fraud recall:    {results['recall_fraud']:.4f}")
    print(f"Fraud F1-score:  {results['f1_fraud']:.4f}")
    print(f"ROC-AUC:         {results['roc_auc']:.4f}")
    print(f"PR-AUC (AP):     {results['pr_auc_average_precision']:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {matrix.tolist()}")
    print(f"Saved outputs to: {output_dir.resolve()}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--max-normal-train", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    run_experiment(args.data, args.output, args.max_normal_train, args.epochs)


if __name__ == "__main__":
    main()
