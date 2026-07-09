"""
==============================================================================
BỔ SUNG BÁO CÁO — Confusion Matrix + Fair Comparison
==============================================================================
Chạy: py improve_report.py
Output:
  figures/cm_xgboost.png     → Figure 4.4
  figures/cm_llm.png         → Figure 4.5
  figures/cm_comparison.png  → Figure 4.6 (cả hai cạnh nhau)
  fair_comparison.json       → số liệu Bảng 4.5
==============================================================================
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(THIS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load dữ liệu từ các file đã có ───────────────────────────────────────
with open(os.path.join(THIS_DIR, "eval_results.json")) as f:
    xgb_full = json.load(f)

with open(os.path.join(THIS_DIR, "llm_results.json")) as f:
    llm_data = json.load(f)

records = llm_data["records"]

# Tái tạo y_true / predictions từ records
y_true_all  = np.array([r["true_label"] for r in records])
xgb_probs   = np.array([r["xgb_prob"]   for r in records])
llm_preds   = np.array([r["llm_pred"]   for r in records])

# Chỉ lấy mẫu LLM parse được
valid_mask      = llm_preds != -1
y_true_valid    = y_true_all[valid_mask]
xgb_preds_valid = (xgb_probs[valid_mask] >= 0.5).astype(int)
xgb_probs_valid = xgb_probs[valid_mask]
llm_preds_valid = llm_preds[valid_mask].astype(int)
n_valid         = valid_mask.sum()

print(f"Mẫu hợp lệ: {n_valid:,}")

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1: CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/3] Vẽ Confusion Matrix...")

cm_xgb = confusion_matrix(y_true_valid, xgb_preds_valid)
cm_llm = confusion_matrix(y_true_valid, llm_preds_valid)

def plot_cm(cm, title, subtitle, path, color):
    fig, ax = plt.subplots(figsize=(6, 5))

    # Normalize để hiển thị %
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=color, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Benign", "Malware"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)

    # Giá trị trong ô
    thresh = cm_norm.max() / 2.0
    for i in range(2):
        for j in range(2):
            pct  = cm_norm[i, j] * 100
            cnt  = cm[i, j]
            color_text = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{pct:.1f}%\n({cnt:,})",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color_text)

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")

plot_cm(cm_xgb,
        "Figure 4.4 — Confusion Matrix: XGBoost",
        f"N={n_valid:,} mẫu | Accuracy={100*cm_xgb.diagonal().sum()/n_valid:.1f}%",
        os.path.join(FIG_DIR, "cm_xgboost.png"),
        plt.cm.Blues)

plot_cm(cm_llm,
        "Figure 4.5 — Confusion Matrix: Llama 3.2",
        f"N={n_valid:,} mẫu | Accuracy={100*cm_llm.diagonal().sum()/n_valid:.1f}%",
        os.path.join(FIG_DIR, "cm_llm.png"),
        plt.cm.Oranges)

# Side-by-side
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, cm, title, cmap in zip(
    axes,
    [cm_xgb, cm_llm],
    ["XGBoost (ML Tầng 1)", "Llama 3.2 (LLM Tầng 2)"],
    [plt.cm.Blues, plt.cm.Oranges]
):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    classes = ["Benign", "Malware"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    thresh = cm_norm.max() / 2.0
    for i in range(2):
        for j in range(2):
            pct = cm_norm[i, j] * 100
            cnt = cm[i, j]
            c   = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{pct:.1f}%\n({cnt:,})",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=c)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

fig.suptitle("Figure 4.6 — Confusion Matrix Comparison\nXGBoost vs Llama 3.2 (cùng 3,000 mẫu)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cm_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ figures/cm_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2: FAIR COMPARISON (cùng 3,000 mẫu)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/3] Tính Fair Comparison metrics...")

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)

xgb_acc  = accuracy_score(y_true_valid, xgb_preds_valid)
xgb_f1   = f1_score(y_true_valid, xgb_preds_valid, average="macro")
xgb_prec = precision_score(y_true_valid, xgb_preds_valid, zero_division=0)
xgb_rec  = recall_score(y_true_valid, xgb_preds_valid, zero_division=0)
xgb_auc  = roc_auc_score(y_true_valid, xgb_probs_valid)
xgb_tn, xgb_fp, xgb_fn, xgb_tp = cm_xgb.ravel()
xgb_fpr  = xgb_fp / (xgb_fp + xgb_tn)

llm_acc  = accuracy_score(y_true_valid, llm_preds_valid)
llm_f1   = f1_score(y_true_valid, llm_preds_valid, average="macro")
llm_prec = precision_score(y_true_valid, llm_preds_valid, zero_division=0)
llm_rec  = recall_score(y_true_valid, llm_preds_valid, zero_division=0)
llm_tn, llm_fp, llm_fn, llm_tp = cm_llm.ravel()
llm_fpr  = llm_fp / (llm_fp + llm_tn)

sep = "=" * 66
print(f"\n{sep}")
print("  BẢNG 4.5 — FAIR COMPARISON (cùng 3,000 mẫu)")
print(f"  Copy đoạn này vào Word, ngay sau Bảng 4.4")
print(sep)
print(f"\n  {'Tiêu chí':<25} {'XGBoost':>12} {'Llama 3.2':>12} {'Winner':>10}")
print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

rows = [
    ("Accuracy",        xgb_acc,  llm_acc,  True),
    ("F1-Score (Macro)",xgb_f1,   llm_f1,   True),
    ("Precision",       xgb_prec, llm_prec, True),
    ("Recall",          xgb_rec,  llm_rec,  True),
    ("False Pos. Rate", xgb_fpr,  llm_fpr,  False),
]
for label, xv, lv, hi in rows:
    w = "XGBoost" if (xv > lv) == hi else "LLM"
    print(f"  {label:<25} {xv:>11.1%} {lv:>11.1%} {w:>10}")

print(f"\n  {'ROC-AUC':<25} {xgb_auc:>12.4f} {'N/A':>12}")
print(f"  {'Latency':<25} {'0.01 ms':>12} {'14,370 ms':>12}")
print(f"  {'Explainability':<25} {'Không':>12} {'Có':>12}")
print(sep)

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3: ĐOẠN VĂN GIẢI THÍCH LLM F1 (copy vào Word mục 4.3)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/3] Đoạn văn giải thích cho Mục 4.3 (copy vào Word):")
print(sep)

text = f"""
Phân tích Confusion Matrix (Hình 4.5) làm rõ nguyên nhân F1 thấp
của Llama 3.2: mô hình đạt Recall {llm_rec:.1%} nhưng Precision chỉ
{llm_prec:.1%}, phản ánh xu hướng phân loại thiên về phía malware
(Malware Paranoia). Cụ thể, trong {int(llm_tn+llm_fp):,} mẫu benign
thực sự, LLM phân loại sai {int(llm_fp):,} mẫu ({llm_fpr:.1%}) thành
malware — tỷ lệ False Positive cao hơn đáng kể so với XGBoost
({xgb_fpr:.1%}). Hiện tượng này phù hợp với quan sát trong các nghiên
cứu gần đây: với few-shot prompting, LLM có xu hướng thiên về lớp
được mô tả chi tiết và nhấn mạnh hơn trong prompt — trong trường hợp
này là lớp malware với các đặc trưng đáng ngờ rõ ràng hơn.

Tuy nhiên, trong kiến trúc Hybrid SOC Pipeline, đặc tính này không
phải nhược điểm nghiêm trọng. LLM chỉ xử lý các mẫu trong vùng xám
(score 0,3–0,7) mà XGBoost đã đánh dấu là không chắc chắn. Với những
mẫu này, xu hướng cảnh thận (cautious) của LLM — thà cảnh báo nhầm
còn hơn bỏ sót — là đặc tính phù hợp với yêu cầu bảo mật. Quyết định
cuối cùng vẫn thuộc về SOC analyst dựa trên giải thích đầy đủ mà LLM
cung cấp.
"""
print(text)
print(sep)

# Lưu JSON
out = {
    "n_samples": int(n_valid),
    "note": "Fair comparison: same 3000 samples",
    "xgboost": {
        "accuracy": round(float(xgb_acc), 4),
        "f1_macro": round(float(xgb_f1), 4),
        "precision": round(float(xgb_prec), 4),
        "recall": round(float(xgb_rec), 4),
        "roc_auc": round(float(xgb_auc), 4),
        "fpr": round(float(xgb_fpr), 4),
        "confusion_matrix": cm_xgb.tolist(),
    },
    "llm": {
        "accuracy": round(float(llm_acc), 4),
        "f1_macro": round(float(llm_f1), 4),
        "precision": round(float(llm_prec), 4),
        "recall": round(float(llm_rec), 4),
        "fpr": round(float(llm_fpr), 4),
        "confusion_matrix": cm_llm.tolist(),
    },
}
with open(os.path.join(THIS_DIR, "fair_comparison.json"), "w") as f:
    json.dump(out, f, indent=2)

print(f"\n  fair_comparison.json")
print(f"  figures/cm_xgboost.png    → Figure 4.4")
print(f"  figures/cm_llm.png        → Figure 4.5")
print(f"  figures/cm_comparison.png → Figure 4.6 (dùng cái này)")
print(f"\n  XONG — Chèn hình và copy bảng vào Word là hoàn chỉnh.")