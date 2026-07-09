"""
==============================================================================
THRESHOLD EXPERIMENT — Tìm ngưỡng tối ưu cho Hybrid Pipeline
==============================================================================
Thử các ngưỡng (low, high) khác nhau:
  - Mẫu score < low  → Benign (whitelist)
  - Mẫu score > high → Malware (alert)
  - Mẫu ở giữa      → Grey zone → gửi lên LLM

Đo với mỗi cặp ngưỡng:
  - % mẫu phải xử lý bởi LLM (chi phí)
  - FPR cuối cùng sau khi LLM xử lý grey zone
  - Recall tổng thể (không bỏ sót malware)

Chạy: py threshold_experiment.py
==============================================================================
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(THIS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load dữ liệu từ llm_results.json ─────────────────────────────────────
with open(os.path.join(THIS_DIR, "llm_results.json")) as f:
    llm_data = json.load(f)

records     = llm_data["records"]
valid       = [r for r in records if r["llm_pred"] != -1]

y_true      = np.array([r["true_label"]  for r in valid])
xgb_scores  = np.array([r["xgb_prob"]    for r in valid])
llm_preds   = np.array([r["llm_pred"]    for r in valid])
n           = len(valid)

print("=" * 66)
print("  THRESHOLD EXPERIMENT — Hybrid SOC Pipeline")
print("=" * 66)
print(f"\n  Tổng mẫu: {n:,} | Malware: {(y_true==1).sum():,} "
      f"| Benign: {(y_true==0).sum():,}")

# ── Định nghĩa các cặp ngưỡng cần thử ────────────────────────────────────
THRESHOLDS = [
    (0.1, 0.9),
    (0.2, 0.8),
    (0.3, 0.7),   # ← ngưỡng hiện tại trong báo cáo
    (0.4, 0.6),
    (0.5, 0.5),   # ← XGBoost thuần (không có grey zone)
]

results = []

sep = "=" * 76
print(f"\n{sep}")
print("  BẢNG 4.6 — SO SÁNH CÁC NGƯỠNG PHÂN VÙNG HYBRID PIPELINE")
print(sep)
print(f"\n  {'Ngưỡng':<12} {'%LLM':>6} {'FPR':>7} {'Recall':>8} "
      f"{'F1':>7} {'Đánh giá':>20}")
print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*8} {'-'*7} {'-'*20}")

for low, high in THRESHOLDS:
    # Phân loại từng mẫu
    y_pred = np.zeros(n, dtype=int)

    for i in range(n):
        score = xgb_scores[i]
        if score <= low:
            y_pred[i] = 0              # Benign — XGBoost quyết định
        elif score >= high:
            y_pred[i] = 1              # Malware — XGBoost quyết định
        else:
            # Grey zone — dùng kết quả LLM
            y_pred[i] = int(llm_preds[i])

    # Tính metrics
    grey_mask  = (xgb_scores > low) & (xgb_scores < high)
    pct_llm    = grey_mask.sum() / n * 100

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0

    # Đánh giá
    if low == 0.5 and high == 0.5:
        note = "XGBoost thuần"
    elif pct_llm < 5 and fpr < 0.05:
        note = "⭐ Tốt nhất"
    elif pct_llm < 10 and fpr < 0.08:
        note = "✅ Cân bằng tốt"
    elif fpr < 0.05:
        note = "FPR thấp"
    else:
        note = "LLM nhiều"

    is_current = (low == 0.3 and high == 0.7)
    marker = " ←current" if is_current else ""

    print(f"  ({low:.1f}, {high:.1f}){marker:<9} "
          f"{pct_llm:>5.1f}% {fpr:>6.1%} {recall:>7.1%} "
          f"{f1:>6.1%} {note:>20}")

    results.append({
        "low": low, "high": high,
        "pct_llm": round(pct_llm, 2),
        "fpr": round(float(fpr), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "n_grey": int(grey_mask.sum()),
    })

print(sep)
print("\n  %LLM = % mẫu phải xử lý bởi LLM (chi phí tính toán)")
print("  FPR  = tỷ lệ file sạch bị cảnh báo nhầm (sau Hybrid)")
print("  Recall = tỷ lệ malware được phát hiện (không bỏ sót)")

# ── Vẽ biểu đồ ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

thresholds_label = [f"({r['low']:.1f},{r['high']:.1f})" for r in results]
pct_llm_vals     = [r["pct_llm"]   for r in results]
fpr_vals         = [r["fpr"] * 100 for r in results]
recall_vals      = [r["recall"]*100 for r in results]
f1_vals          = [r["f1"]*100    for r in results]

colors = ["#e74c3c" if r["low"]==0.3 and r["high"]==0.7
          else "#3498db" for r in results]

# Plot 1: % LLM workload
ax1 = axes[0]
bars = ax1.bar(thresholds_label, pct_llm_vals, color=colors, alpha=0.85,
               edgecolor="white")
ax1.set_title("% Mẫu xử lý bởi LLM\n(Chi phí tính toán)", fontsize=11,
              fontweight="bold")
ax1.set_ylabel("%", fontsize=10)
ax1.set_xlabel("Ngưỡng (low, high)", fontsize=10)
ax1.tick_params(axis='x', rotation=25)
for bar, val in zip(bars, pct_llm_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Plot 2: FPR
ax2 = axes[1]
bars2 = ax2.bar(thresholds_label, fpr_vals, color=colors, alpha=0.85,
                edgecolor="white")
ax2.set_title("False Positive Rate\n(File sạch bị cảnh báo nhầm)",
              fontsize=11, fontweight="bold")
ax2.set_ylabel("%", fontsize=10)
ax2.set_xlabel("Ngưỡng (low, high)", fontsize=10)
ax2.tick_params(axis='x', rotation=25)
ax2.axhline(5, color="green", linestyle="--", alpha=0.7,
            label="Target FPR 5%")
ax2.legend(fontsize=9)
for bar, val in zip(bars2, fpr_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Plot 3: F1
ax3 = axes[2]
bars3 = ax3.bar(thresholds_label, f1_vals, color=colors, alpha=0.85,
                edgecolor="white")
ax3.set_title("F1-Score Tổng hợp\n(Hybrid Pipeline)",
              fontsize=11, fontweight="bold")
ax3.set_ylabel("%", fontsize=10)
ax3.set_xlabel("Ngưỡng (low, high)", fontsize=10)
ax3.tick_params(axis='x', rotation=25)
for bar, val in zip(bars3, f1_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# Legend chung
from matplotlib.patches import Patch
legend_els = [Patch(facecolor="#e74c3c", label="Ngưỡng hiện tại (0.3, 0.7)"),
              Patch(facecolor="#3498db", label="Ngưỡng khác")]
fig.legend(handles=legend_els, loc="lower center", ncol=2,
           fontsize=10, bbox_to_anchor=(0.5, -0.05))

fig.suptitle("Figure 4.7 — Phân tích Ngưỡng Phân vùng Hybrid SOC Pipeline\n"
             "Trade-off giữa Chi phí LLM, FPR và F1",
             fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
fig_path = os.path.join(FIG_DIR, "threshold_analysis.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  ✅ Figure 4.7 → {fig_path}")

# Lưu JSON
with open(os.path.join(THIS_DIR, "threshold_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"  ✅ threshold_results.json")

# Tìm ngưỡng tối ưu
best = min(results[:-1],  # bỏ (0.5, 0.5) XGBoost thuần
           key=lambda r: r["fpr"] * 0.6 + r["pct_llm"] * 0.01 - r["f1"] * 0.39)
print(f"\n  NGƯỠNG TỐI ƯU (cân bằng FPR, chi phí LLM, F1):")
print(f"  → ({best['low']:.1f}, {best['high']:.1f}) — "
      f"FPR={best['fpr']:.1%}, %LLM={best['pct_llm']:.1f}%, "
      f"F1={best['f1']:.1%}")
print(f"\n  XONG — Chèn Figure 4.7 và Bảng 4.6 vào Word.")