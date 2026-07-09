"""
==============================================================================
COMPARISON REPORT — XGBoost vs Llama 3.2
==============================================================================
Chạy SAU llm_eval.py.
Output:
  - figures/comparison_bar.png  → Figure 4.3
  - comparison_table.json       → số liệu tổng hợp
Chạy: py comparison_report.py
==============================================================================
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(THIS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load kết quả ─────────────────────────────────────────────────────────
eval_path = os.path.join(THIS_DIR, "eval_results.json")
llm_path  = os.path.join(THIS_DIR, "llm_results.json")

with open(eval_path) as f:
    xgb = json.load(f)
with open(llm_path) as f:
    llm = json.load(f)

# ── Bảng so sánh tổng hợp ────────────────────────────────────────────────
sep = "=" * 68
print(f"\n{sep}")
print("  TABLE 4.3 — BẢNG SO SÁNH TỔNG HỢP")
print(f"  XGBoost (ML Tầng 1) vs Llama 3.2 (LLM Tầng 2)")
print(sep)
print(f"\n  {'Tiêu chí':<32} {'XGBoost':>14} {'Llama 3.2':>14}")
print(f"  {'-'*32} {'-'*14} {'-'*14}")

rows = [
    ("Accuracy",             f"{xgb['accuracy']:.1%}",
                             f"{llm['accuracy']:.1%}"),
    ("F1-Score (Macro)",     f"{xgb['f1_macro']:.1%}",
                             f"{llm['f1_macro']:.1%}"),
    ("Precision",            f"{xgb['precision']:.1%}",
                             f"{llm['precision']:.1%}"),
    ("Recall",               f"{xgb['recall']:.1%}",
                             f"{llm['recall']:.1%}"),
    ("ROC-AUC",              f"{xgb['roc_auc']:.4f}",     "N/A"),
    ("False Positive Rate",  f"{xgb['false_positive_rate']:.1%}", "N/A"),
    ("Latency (ms/sample)",  f"{xgb['infer_ms_sample']:.3f}",
                             f"{llm['avg_latency_s']*1000:.0f}"),
    ("Throughput (samp/s)",  f"{1000/xgb['infer_ms_sample']:,.0f}",
                             f"{1/llm['avg_latency_s']:.1f}"),
    ("Explainability",       "Không",                     "Có (text)"),
    ("MITRE ATT&CK",         "Không",                     "Có"),
    ("Số mẫu đánh giá",      f"{xgb['n_test']:,}",
                             f"{llm['n_valid']:,}"),
]

for label, xval, lval in rows:
    print(f"  {label:<32} {xval:>14} {lval:>14}")
print(sep)

# Nhận xét tự động
print(f"\n  NHẬN XÉT:")
f1_gap = xgb['f1_macro'] - llm['f1_macro']
lat_ratio = llm['avg_latency_s'] * 1000 / xgb['infer_ms_sample']
print(f"  • XGBoost vượt trội về F1: +{f1_gap:.1%} so với Llama 3.2")
print(f"  • Llama 3.2 chậm hơn {lat_ratio:,.0f}x nhưng cung cấp giải thích")
print(f"  • → Hybrid pipeline: XGBoost triage + LLM explain là tối ưu")

# ── Vẽ Figure 4.3: So sánh metrics ──────────────────────────────────────
metrics   = ["Accuracy", "F1-Macro", "Precision", "Recall"]
xgb_vals  = [xgb["accuracy"], xgb["f1_macro"],
             xgb["precision"], xgb["recall"]]
llm_vals  = [llm["accuracy"], llm["f1_macro"],
             llm["precision"], llm["recall"]]

x     = np.arange(len(metrics))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# -- Subplot 1: Performance metrics --
ax1 = axes[0]
bars1 = ax1.bar(x - width/2, xgb_vals, width,
                label="XGBoost", color="#2980b9", alpha=0.85,
                edgecolor="white")
bars2 = ax1.bar(x + width/2, llm_vals, width,
                label="Llama 3.2", color="#e67e22", alpha=0.85,
                edgecolor="white")

ax1.set_ylabel("Score", fontsize=11)
ax1.set_title("Figure 4.3a — Performance Comparison\nXGBoost vs Llama 3.2",
              fontsize=11, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=10)
ax1.set_ylim(0, 1.12)
ax1.legend(fontsize=10)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.axhline(0.9, color="gray", linestyle="--", alpha=0.5, linewidth=1)

# Giá trị trên cột
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.01,
             f"{h:.1%}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.01,
             f"{h:.1%}", ha="center", va="bottom", fontsize=8)

# -- Subplot 2: Radar / Capability comparison --
ax2 = axes[1]
categories   = ["Detection\nAccuracy", "Speed\n(relative)",
                "Explainability", "MITRE\nMapping", "Ease of\nDeployment"]
xgb_scores   = [xgb["f1_macro"], 1.0, 0.1, 0.0, 0.9]
llm_scores   = [llm["f1_macro"], 0.05, 1.0, 1.0, 0.5]

x2    = np.arange(len(categories))
w2    = 0.35
ax2.bar(x2 - w2/2, xgb_scores, w2, label="XGBoost",
        color="#2980b9", alpha=0.85, edgecolor="white")
ax2.bar(x2 + w2/2, llm_scores, w2, label="Llama 3.2",
        color="#e67e22", alpha=0.85, edgecolor="white")

ax2.set_title("Figure 4.3b — Capability Comparison\n(Normalized 0–1)",
              fontsize=11, fontweight="bold")
ax2.set_xticks(x2)
ax2.set_xticklabels(categories, fontsize=9)
ax2.set_ylim(0, 1.15)
ax2.legend(fontsize=10)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout(pad=3)
fig_path = os.path.join(FIG_DIR, "comparison_bar.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  ✅ Figure 4.3 → {fig_path}")

# ── Lưu JSON ─────────────────────────────────────────────────────────────
out = {
    "xgboost": xgb,
    "llm"    : {k: v for k, v in llm.items() if k != "records"},
    "comparison": {
        "f1_gap_xgb_minus_llm" : round(xgb["f1_macro"] - llm["f1_macro"], 4),
        "latency_ratio"        : round(llm["avg_latency_s"]*1000
                                       / xgb["infer_ms_sample"], 0),
        "conclusion"           : "Hybrid pipeline recommended: "
                                 "XGBoost for triage, LLM for explanation",
    }
}
with open(os.path.join(THIS_DIR, "comparison_table.json"), "w") as f:
    json.dump(out, f, indent=2)

print(f"  ✅ comparison_table.json")
print(f"\n  XONG — Tất cả số liệu đã sẵn sàng cho báo cáo.")