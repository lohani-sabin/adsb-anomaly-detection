
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import json

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, accuracy_score
)


def load_frozen_models():
    # Load all frozen artifacts from build phase
    print("\nLoading frozen models...")
    
    required = ["output/scaler.pkl", "output/iso_model.pkl"]
    for f in required:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing required artifact: {f}. Run build.py first.")
    
    scaler = joblib.load("output/scaler.pkl")
    iso_model = joblib.load("output/iso_model.pkl")
    
    models = {'scaler': scaler, 'iso_model': iso_model}
    
    # Try to load LSTM
    lstm_path = "output/lstm_model.keras"
    if os.path.exists(lstm_path):
        try:
            from tensorflow.keras.models import load_model
            models['lstm_model'] = load_model(lstm_path)
            print(f"\tLoaded LSTM from {lstm_path}")
        except Exception as e:
            print(f"\tWARNING: Could not load LSTM: {e}")
    else:
        print("\tLSTM model not found (optional)")
    
    print("\tAll models loaded successfully")
    return models


def predict_isolation_forest(iso_model, scaler, X_test):
    # Get anomaly scores and predictions from Isolation Forest."""
    X_scaled = scaler.transform(X_test)
    
    # anomaly_score: lower = more anomalous (sklearn convention)
    # We negate so higher = more anomalous (standard convention)
    anomaly_scores = -iso_model.score_samples(X_scaled)
    
    # Default predictions: -1 = anomaly, 1 = normal
    raw_preds = iso_model.predict(X_scaled)
    # Convert to 1 = anomaly, 0 = normal
    binary_preds = (raw_preds == -1).astype(int)
    
    return binary_preds, anomaly_scores


def predict_lstm_autoencoder(lstm_model, scaler, X_test, seq_length=10):
    # Get reconstruction errors from LSTM Autoencoder.
    X_scaled = scaler.transform(X_test)
    
    # Need enough data for sequences
    if len(X_scaled) < seq_length:
        print(f"  WARNING: Test set too small for LSTM sequences ({len(X_scaled)} < {seq_length})")
        return None, None
    
    # Create sequences
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)
    
    X_seq = create_sequences(X_scaled, seq_length)
    
    # Predict reconstructions
    reconstructed = lstm_model.predict(X_seq, verbose=0)
    
    # Reconstruction error (MSE) per sequence
    mse = np.mean(np.power(X_seq - reconstructed, 2), axis=(1, 2))
    
    # Map sequence errors back to original samples
    # The error at position i corresponds to the sequence ending at i+seq_length-1
    scores = np.zeros(len(X_scaled))
    scores[seq_length - 1:] = mse
    
    # For the first seq_length-1 samples, use the first sequence error
    scores[:seq_length - 1] = mse[0] if len(mse) > 0 else 0
    
    # Threshold: 99th percentile (top 1% flagged as anomalies)
    threshold = np.percentile(scores, 99)
    binary_preds = (scores > threshold).astype(int)
    
    return binary_preds, scores


def compute_metrics(y_true, y_pred, anomaly_scores=None):
    # Compute precision, recall, FPR, F1, and AUC-ROC.
    tp = int(sum((y_true == 1) & (y_pred == 1)))
    fp = int(sum((y_true == 0) & (y_pred == 1)))
    tn = int(sum((y_true == 0) & (y_pred == 0)))
    fn = int(sum((y_true == 1) & (y_pred == 0)))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    metrics = {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'fpr': round(fpr, 4),
        'f1_score': round(f1, 4),
        'accuracy': round(accuracy, 4),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }
    
    # AUC-ROC
    if anomaly_scores is not None and len(set(y_true)) > 1:
        fpr_list, tpr_list, _ = roc_curve(y_true, anomaly_scores)
        metrics['auc_roc'] = round(auc(fpr_list, tpr_list), 4)
    else:
        metrics['auc_roc'] = None
    
    return metrics


def generate_confusion_matrix(y_true, y_pred, model_name, save_path):
    # Generate and save confusion matrix plot.
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'],
           title=f'Confusion Matrix - {model_name}',
           ylabel='True Label', xlabel='Predicted Label')
    
    # Add text annotations
    thresh = cm.max() / 2. if cm.max() > 0 else 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def generate_roc_curve(y_true, iso_scores, lstm_scores, save_path):
    # Generate ROC curve comparing both models.
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Isolation Forest
    if iso_scores is not None and len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, iso_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'Isolation Forest (AUC = {roc_auc:.3f})', linewidth=2)
    
    # LSTM
    if lstm_scores is not None and len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, lstm_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'LSTM Autoencoder (AUC = {roc_auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve Comparison', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\tSaved: {save_path}")


def generate_per_attack_breakdown(results_df, save_path):
    # Generate bar chart of per-attack performance.
    attack_types = results_df['attack_type'].unique()
    
    metrics_by_attack = []
    for atype in sorted(attack_types):
        subset = results_df[results_df['attack_type'] == atype]
        if len(subset) == 0:
            continue
        
        y_true_sub = subset['true_label'].values
        y_pred_iso_sub = subset['iso_pred'].values
        
        entry = {
            'attack_type': atype,
            'count': len(subset),
            'iso_precision': precision_score(y_true_sub, y_pred_iso_sub, zero_division=0),
            'iso_recall': recall_score(y_true_sub, y_pred_iso_sub, zero_division=0),
            'iso_f1': f1_score(y_true_sub, y_pred_iso_sub, zero_division=0),
        }
        
        if 'lstm_pred' in subset.columns:
            y_pred_lstm_sub = subset['lstm_pred'].values
            entry['lstm_precision'] = precision_score(y_true_sub, y_pred_lstm_sub, zero_division=0)
            entry['lstm_recall'] = recall_score(y_true_sub, y_pred_lstm_sub, zero_division=0)
            entry['lstm_f1'] = f1_score(y_true_sub, y_pred_lstm_sub, zero_division=0)
        
        metrics_by_attack.append(entry)
    
    metrics_df = pd.DataFrame(metrics_by_attack)
    if len(metrics_df) == 0:
        print("  WARNING: No per-attack data to plot")
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_df))
    width = 0.15
    
    bars1 = ax.bar(x - 2.5*width, metrics_df['iso_precision'], width, label='Iso Precision', alpha=0.9)
    bars2 = ax.bar(x - 1.5*width, metrics_df['iso_recall'], width, label='Iso Recall', alpha=0.9)
    bars3 = ax.bar(x - 0.5*width, metrics_df['iso_f1'], width, label='Iso F1', alpha=0.9)
    
    if 'lstm_precision' in metrics_df.columns:
        bars4 = ax.bar(x + 0.5*width, metrics_df['lstm_precision'], width, label='LSTM Precision', alpha=0.9)
        bars5 = ax.bar(x + 1.5*width, metrics_df['lstm_recall'], width, label='LSTM Recall', alpha=0.9)
        bars6 = ax.bar(x + 2.5*width, metrics_df['lstm_f1'], width, label='LSTM F1', alpha=0.9)
    
    ax.set_xlabel('Attack Type', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Per-Attack Performance Breakdown', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['attack_type'], rotation=15, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def generate_report_text(metrics, per_attack_metrics, output_path):
    # Generate human-readable report text.
    lines = [
        "=" * 60,
        "AIRCRAFT NETWORK FORENSIC EVALUATION REPORT",
        "=" * 60,
        "",
        "OVERALL PERFORMANCE",
        "-" * 40,
        "",
        "Isolation Forest:",
        f"\tPrecision:\t{metrics['iso']['precision']:.4f}",
        f"\tRecall:\t{metrics['iso']['recall']:.4f}",
        f"\tFPR:\t{metrics['iso']['fpr']:.4f}",
        f"\tF1 Score:\t{metrics['iso']['f1_score']:.4f}",
        f"\tAccuracy:\t{metrics['iso']['accuracy']:.4f}",
        f"\tAUC-ROC:\t{metrics['iso'].get('auc_roc', 'N/A')}",
        f"\tTP: {metrics['iso']['tp']} | FP: {metrics['iso']['fp']} | TN: {metrics['iso']['tn']} | FN: {metrics['iso']['fn']}",
        ""
    ]
    
    if 'lstm' in metrics:
        lines.extend([
            "LSTM Autoencoder:",
            f"\tPrecision:\t{metrics['lstm']['precision']:.4f}",
            f"\tRecall:\t{metrics['lstm']['recall']:.4f}",
            f"\tFPR:\t{metrics['lstm']['fpr']:.4f}",
            f"\tF1 Score:\t{metrics['lstm']['f1_score']:.4f}",
            f"\tAccuracy:\t{metrics['lstm']['accuracy']:.4f}",
            f"\tAUC-ROC:\t{metrics['lstm'].get('auc_roc', 'N/A')}",
            f"\tTP: {metrics['lstm']['tp']} | FP: {metrics['lstm']['fp']} | TN: {metrics['lstm']['tn']} | FN: {metrics['lstm']['fn']}",
            ""
        ])
    
    lines.extend([
        "PER-ATTACK BREAKDOWN",
        "-" * 40,
        ""
    ])
    
    for am in per_attack_metrics:
        lines.append(f"{am['attack_type']} (n={am['count']}):")
        lines.append(f"\tIso Forest:\tPrecision={am['iso_precision']:.4f}, Recall={am['iso_recall']:.4f}, F1={am['iso_f1']:.4f}")
        if 'lstm_precision' in am:
            lines.append(f"\tLSTM:\tPrecision={am['lstm_precision']:.4f}, Recall={am['lstm_recall']:.4f}, F1={am['lstm_f1']:.4f}")
        lines.append("")
    
    lines.extend([
        "OPERATIONAL LIMITATIONS",
        "-" * 40,
        "- LSTM requires sufficient sequence length; may miss early anomalies",
        "- Threshold selection affects trade-off between precision and recall",
        "- FPR at 3% still generates many alerts at high-traffic airports",
        "- Cross-airport generalization requires retraining on local patterns",
        "- Synthetic attacks inherit realism from cloned real trajectories",
        "- Model comparison assumes identical test conditions for both algorithms"
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\tSaved: {output_path}")


def run_evaluation(features, labels, icaos):
    # Main evaluation pipeline.
    print("\n" + "=" * 50)
    print("PHASE 5: EVALUATION")
    print("=" * 50)
    
    # Load frozen models
    models = load_frozen_models()
    scaler = models['scaler']
    iso_model = models['iso_model']
    
    # Filter to test set if available
    if os.path.exists("output/test_icaos.npy"):
        test_icaos = np.load("output/test_icaos.npy", allow_pickle=True)
        print(f"\nFiltering to {len(test_icaos)} test aircraft from build phase")
        
        mask = icaos.isin(test_icaos)
        features = features[mask].reset_index(drop=True)
        labels = labels[mask].reset_index(drop=True)
        icaos = icaos[mask].reset_index(drop=True)
        print(f"Evaluating on {len(features)} test aircraft")
    else:
        print("\nWARNING: test_icaos.npy not found. Evaluating on full dataset.")
    
    if len(features) == 0:
        print("ERROR: No test aircraft to evaluate.")
        return
    
    # Get predictions from Isolation Forest
    print("\nRunning Isolation Forest...")
    iso_preds, iso_scores = predict_isolation_forest(iso_model, scaler, features)
    print(f"\tFlagged {sum(iso_preds)} of {len(iso_preds)} aircraft as anomalous")
    
    # Get predictions from LSTM Autoencoder
    lstm_preds = None
    lstm_scores = None
    if 'lstm_model' in models:
        print("\nRunning LSTM Autoencoder...")
        lstm_preds, lstm_scores = predict_lstm_autoencoder(
            models['lstm_model'], scaler, features
        )
        if lstm_preds is not None:
            print(f"\tFlagged {sum(lstm_preds)} of {len(lstm_preds)} aircraft as anomalous")
    
    # Build results dataframe
    results = pd.DataFrame({
        'icao24': icaos.values,
        'true_label': labels.values,
        'iso_pred': iso_preds,
        'iso_score': iso_scores,
    })
    
    if lstm_preds is not None:
        results['lstm_pred'] = lstm_preds
        results['lstm_score'] = lstm_scores
    
    # Map attack types from hybrid dataset
    if os.path.exists("output/hybrid_dataset.csv"):
        hybrid = pd.read_csv("output/hybrid_dataset.csv")
        attack_map = hybrid.groupby('icao24')['attack_type'].first().to_dict()
        results['attack_type'] = results['icao24'].map(attack_map).fillna('unknown')
    else:
        results['attack_type'] = results['true_label'].apply(lambda x: 'attack' if x == 1 else 'normal')
    
    # Compute overall metrics
    y_true = results['true_label'].values
    
    iso_metrics = compute_metrics(y_true, iso_preds, iso_scores)
    print(f"\nIsolation Forest Results:")
    print(f"\tPrecision: {iso_metrics['precision']:.4f} | Recall: {iso_metrics['recall']:.4f} | FPR: {iso_metrics['fpr']:.4f} | F1: {iso_metrics['f1_score']:.4f}")
    
    metrics = {'iso': iso_metrics}
    
    if lstm_preds is not None:
        lstm_metrics = compute_metrics(y_true, lstm_preds, lstm_scores)
        print(f"\nLSTM Autoencoder Results:")
        print(f"\tPrecision: {lstm_metrics['precision']:.4f} | Recall: {lstm_metrics['recall']:.4f} | FPR: {lstm_metrics['fpr']:.4f} | F1: {lstm_metrics['f1_score']:.4f}")
        metrics['lstm'] = lstm_metrics
    
    # Per-attack metrics
    print("\nComputing per-attack breakdown...")
    attack_types = results['attack_type'].unique()
    per_attack_metrics = []
    for atype in sorted(attack_types):
        subset = results[results['attack_type'] == atype]
        if len(subset) == 0:
            continue
        
        y_true_sub = subset['true_label'].values
        y_pred_iso_sub = subset['iso_pred'].values
        
        pam = {
            'attack_type': atype,
            'count': len(subset),
            'iso_precision': precision_score(y_true_sub, y_pred_iso_sub, zero_division=0),
            'iso_recall': recall_score(y_true_sub, y_pred_iso_sub, zero_division=0),
            'iso_f1': f1_score(y_true_sub, y_pred_iso_sub, zero_division=0)
        }
        
        if 'lstm_pred' in subset.columns:
            y_pred_lstm_sub = subset['lstm_pred'].values
            pam['lstm_precision'] = precision_score(y_true_sub, y_pred_lstm_sub, zero_division=0)
            pam['lstm_recall'] = recall_score(y_true_sub, y_pred_lstm_sub, zero_division=0)
            pam['lstm_f1'] = f1_score(y_true_sub, y_pred_lstm_sub, zero_division=0)
        
        per_attack_metrics.append(pam)
    
    # Save all outputs
    os.makedirs("output/report", exist_ok=True)
    
    print("\nGenerating reports...")
    
    # 1. metrics.json
    with open("output/report/metrics.json", 'w') as f:
        json.dump({
            'overall': metrics,
            'per_attack': per_attack_metrics
        }, f, indent=2)
    print("\tSaved: output/report/metrics.json")
    
    # 2. metrics.txt
    generate_report_text(metrics, per_attack_metrics, "output/report/metrics.txt")
    
    # 3. Confusion matrices
    generate_confusion_matrix(y_true, iso_preds, "Isolation Forest", "output/report/confusion_matrix_iso.png")
    if lstm_preds is not None:
        generate_confusion_matrix(y_true, lstm_preds, "LSTM Autoencoder", "output/report/confusion_matrix_lstm.png")
    
    # 4. ROC curve
    generate_roc_curve(y_true, iso_scores, lstm_scores, "output/report/roc_curve.png")
    
    # 5. Per-attack breakdown
    generate_per_attack_breakdown(results, "output/report/per_attack_breakdown.png")
    
    # 6. Anomaly scores CSV
    results.to_csv("output/report/anomaly_scores.csv", index=False)
    print("\tSaved: output/report/anomaly_scores.csv")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("Reports saved to output/report/")
    print("\nKey files:")
    print("\t- metrics.txt (human-readable summary)")
    print("\t- metrics.json (machine-readable scores)")
    print("\t- anomaly_scores.csv (per-aircraft verdicts)")
    print("\t- confusion_matrix_*.png")
    print("\t- roc_curve.png")
    print("\t- per_attack_breakdown.png")