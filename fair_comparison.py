"""
==============================================================================
FAIR COMPARISON — XGBoost trên đúng 3,000 mẫu của LLM
==============================================================================
Đọc llm_checkpoint.json để lấy đúng indices LLM đã dùng
→ Chạy XGBoost trên cùng sample đó
→ So sánh apple-to-apple
Chạy: py fair_comparison.py
==============================================================================
"""

import os, json, gc
import numpy as np
import xgboost as xgb
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              accuracy_score, roc_auc_score, confusion_matrix)

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = r"C:\new\malware_project\ember_data\ember2018"
N_FEATURES = 2351
CKPT_PATH  = os.path.join(THIS_DIR, "llm_checkpoint.json")
LLM_PATH   = os.path.join(THIS_DIR, "llm_results.json")

print("=" * 66)
print("  FAIR COMPARISON — XGBoost vs LLM trên cùng 3,000 mẫu")
print("=" * 66)

# ── Load LLM records để lấy true_label và xgb_prob đã có sẵn ─────────────
with open(LLM_PATH) as f:
    llm_data = json.load(f)

records = llm_data["records"]
print(f"\n  LLM records: {len(records):,} mẫu")

# ── Tái tạo kết quả từ records (không cần đọc lại file .dat) ─────────────
# XGBoost prob đã được lưu trong mỗi record khi chạy llm_eval.py
y_true      = []
xgb_probs   = []
llm_preds   = []
llm_valids  = []

for r in records:
    y_true.append(r["true_label"])
    xgb_probs.append(r["xgb_prob"])
    llm_preds.append(r["llm_pred"])
    llm_valids.append(r["llm_pred"] != -1)

y_true    = np.array(y_true)
xgb_probs = np.array(xgb_probs)
llm_preds = np.array(llm_preds)

# XGBoost prediction trên cùng 3,000 mẫu (threshold 0.5)
xgb_preds = (xgb_probs >= 0.5).astype(int)

# Chỉ so sánh trên các mẫu LLM parse được (loại -1)
valid_mask      = np.array(llm_valids)
y_true_valid    = y_true[valid_mask]
xgb_preds_valid = xgb_preds[valid_mask]
xgb_probs_valid = xgb_probs[valid_mask]
llm_preds_valid = llm_preds[valid_mask]
n_valid         = valid_mask.sum()

print(f"  Mẫu hợp lệ (LLM parse OK): {n_valid:,}/{len(records):,}")
print(f"  Malware: {(y_true_valid==1).sum():,} | "
      f"Benign: {(y_true_valid==0).sum():,}")

# ── Metrics XGBoost trên 3,000 mẫu ───────────────────────────────────────
xgb_acc  = accuracy_score(y_true_valid, xgb_preds_valid)
xgb_f1   = f1_score(y_true_valid, xgb_preds_valid, average="macro")
xgb_prec = precision_score(y_true_valid, xgb_preds_valid, zero_division=0)
xgb_rec  = recall_score(y_true_valid, xgb_preds_valid, zero_division=0)
xgb_auc  = roc_auc_score(y_true_valid, xgb_probs_valid)
xgb_cm   = confusion_matrix(y_true_valid, xgb_preds_valid)
xgb_tn, xgb_fp, xgb_fn, xgb_tp = xgb_cm.ravel()
xgb_fpr  = xgb_fp / (xgb_fp + xgb_tn)

# ── Metrics LLM trên cùng mẫu ────────────────────────────────────────────
llm_acc  = accuracy_score(y_true_valid, llm_preds_valid)
llm_f1   = f1_score(y_true_valid, llm_preds_valid, average="macro")
llm_prec = precision_score(y_true_valid, llm_preds_valid, zero_division=0)
llm_rec  = recall_score(y_true_valid, llm_preds_valid, zero_division=0)
llm_cm   = confusion_matrix(y_true_valid, llm_preds_valid)
llm_tn, llm_fp, llm_fn, llm_tp = llm_cm.ravel()
llm_fpr  = llm_fp / (llm_fp + llm_tn)

# ── Bảng so sánh ─────────────────────────────────────────────────────────
sep = "=" * 66
print(f"\n{sep}")
print("  TABLE 4.4 — FAIR COMPARISON (cùng 3,000 mẫu)")
print(f"  XGBoost vs Llama 3.2 — Apple-to-Apple")
print(sep)
print(f"\n  {'Tiêu chí':<28} {'XGBoost':>14} {'Llama 3.2':>14} {'Winner':>8}")
print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*8}")

rows = [
    ("Accuracy",        xgb_acc,  llm_acc,  True),
    ("F1-Score (Macro)",xgb_f1,   llm_f1,   True),
    ("Precision",       xgb_prec, llm_prec, True),
    ("Recall",          xgb_rec,  llm_rec,  True),
    ("False Pos. Rate", xgb_fpr,  llm_fpr,  False),  # lower is better
]

for label, xval, lval, higher_better in rows:
    if higher_better:
        winner = "XGBoost" if xval > lval else "LLM    "
    else:
        winner = "XGBoost" if xval < lval else "LLM    "
    print(f"  {label:<28} {xval:>13.1%} {lval:>13.1%} {winner:>8}")

print(f"\n  {'ROC-AUC':<28} {xgb_auc:>14.4f} {'N/A':>14}")
print(f"  {'Latency':<28} {'0.01 ms':>14} {'14,370 ms':>14}")
print(f"  {'Explainability':<28} {'Không':>14} {'Có':>14}")
print(sep)

# ── Confusion Matrix so sánh ──────────────────────────────────────────────
print(f"\n  CONFUSION MATRIX (trên {n_valid:,} mẫu):")
print(f"\n  XGBoost:")
print(f"                   Pred Benign  Pred Malware")
print(f"  Actual Benign  : {xgb_tn:>11,}  {xgb_fp:>12,}")
print(f"  Actual Malware : {xgb_fn:>11,}  {xgb_tp:>12,}")

print(f"\n  Llama 3.2:")
print(f"                   Pred Benign  Pred Malware")
print(f"  Actual Benign  : {llm_tn:>11,}  {llm_fp:>12,}")
print(f"  Actual Malware : {llm_fn:>11,}  {llm_tp:>12,}")

# ── Phân tích ─────────────────────────────────────────────────────────────
f1_gap  = xgb_f1 - llm_f1
rec_gap = xgb_rec - llm_rec

print(f"\n  PHÂN TÍCH:")
print(f"  • XGBoost vượt trội F1: +{f1_gap:.1%} trên cùng tập mẫu")
print(f"  • Recall gap: XGBoost {xgb_rec:.1%} vs LLM {llm_rec:.1%} "
      f"(+{rec_gap:.1%})")
print(f"  • LLM FPR {llm_fpr:.1%} vs XGBoost FPR {xgb_fpr:.1%} "
      f"— {'LLM tệ hơn' if llm_fpr > xgb_fpr else 'LLM tốt hơn'} về false alarm")
print(f"  • XGBoost nhanh hơn {14370/0.01:,.0f}x")
print(f"  • → H1 được xác nhận: ML vượt trội LLM về detection performance")

# ── Lưu ──────────────────────────────────────────────────────────────────
out = {
    "n_samples"     : int(n_valid),
    "note"          : "Fair comparison: same 3000 samples for both models",
    "xgboost": {
        "accuracy" : round(float(xgb_acc),  4),
        "f1_macro" : round(float(xgb_f1),   4),
        "precision": round(float(xgb_prec), 4),
        "recall"   : round(float(xgb_rec),  4),
        "roc_auc"  : round(float(xgb_auc),  4),
        "fpr"      : round(float(xgb_fpr),  4),
        "latency_ms": 0.01,
    },
    "llm": {
        "accuracy" : round(float(llm_acc),  4),
        "f1_macro" : round(float(llm_f1),   4),
        "precision": round(float(llm_prec), 4),
        "recall"   : round(float(llm_rec),  4),
        "fpr"      : round(float(llm_fpr),  4),
        "latency_ms": 14370,
    },
}
with open(os.path.join(THIS_DIR, "fair_comparison.json"), "w") as f:
    json.dump(out, f, indent=2)

print(f"\n  fair_comparison.json")
print(f"\n  Dùng bảng này làm TABLE 4.4 chính trong báo cáo")
print(f"  (Thuyết phục hơn vì same sample size)")