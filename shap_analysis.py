"""
SHAP ANALYSIS — Top 10 Features & Plots
RAM-safe: chỉ dùng 2,000 mẫu (~50MB)
Chạy: py shap_analysis.py
"""

import os, json, gc
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = r"C:\new\malware_project\ember_data\ember2018"
N_FEATURES = 2351
MODEL_PATH = os.path.join(THIS_DIR, "xgboost_ember_layer1.json")
FIG_DIR    = os.path.join(THIS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 60)
print("  SHAP ANALYSIS — XGBoost + EMBER 2018")
print("=" * 60)

# ── BƯỚC 1: Lấy 2,000 mẫu từ test set ───────────────────────────────────
print("\n  [1/4] Lấy 2,000 mẫu test (1k malware + 1k benign)...")

y_path = os.path.join(DATA_DIR, "y_test.dat")
X_path = os.path.join(DATA_DIR, "X_test.dat")

y_test_all  = np.fromfile(y_path, dtype=np.float32)
labeled_idx = np.where(y_test_all != -1)[0]
y_labeled   = y_test_all[labeled_idx].astype(int)
del y_test_all; gc.collect()

mal_idx = labeled_idx[y_labeled == 1]
ben_idx = labeled_idx[y_labeled == 0]
rng     = np.random.RandomState(42)
sel_mal = rng.choice(mal_idx, size=min(1000, len(mal_idx)), replace=False)
sel_ben = rng.choice(ben_idx, size=min(1000, len(ben_idx)), replace=False)
sel_idx = np.sort(np.concatenate([sel_mal, sel_ben]))

# Đọc từng dòng — chậm nhưng RAM-safe
X_sample = np.zeros((len(sel_idx), N_FEATURES), dtype=np.float32)
for i, global_row in enumerate(sel_idx):
    offset = int(global_row) * N_FEATURES * 4
    row    = np.memmap(X_path, dtype=np.float32, mode='r',
                       offset=offset, shape=(N_FEATURES,))
    X_sample[i] = row
    del row
    if (i + 1) % 500 == 0:
        print(f"    {i+1}/{len(sel_idx)}", end="\r")

gc.collect()
print(f"  OK X_sample: {X_sample.shape}                ")

# ── BƯỚC 2: Tính SHAP ────────────────────────────────────────────────────
print("\n  [2/4] Tính SHAP values (~2-5 phut)...")
model = xgb.Booster()
model.load_model(MODEL_PATH)

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
print(f"  OK shap_values: {shap_values.shape}")

# ── BƯỚC 3: Feature names ─────────────────────────────────────────────────
try:
    import ember
    extractor     = ember.PEFeatureExtractor()
    feature_names = []
    for feat in extractor.features:
        dim   = feat.dim
        names = [f"{feat.name}_{i}" for i in range(dim)]
        feature_names.extend(names)
    if len(feature_names) != N_FEATURES:
        raise ValueError(f"{len(feature_names)} != {N_FEATURES}")
    print(f"  OK feature names: {len(feature_names)} tu ember")
except Exception as e:
    print(f"  Dung generic names ({e})")
    feature_names = [f"f{i}" for i in range(N_FEATURES)]

# ── BƯỚC 4: Top 10 + vẽ ──────────────────────────────────────────────────
print("\n  [3/4] Top 10 features...")
mean_shap = np.abs(shap_values).mean(axis=0)
top_idx   = np.argsort(mean_shap)[::-1][:10]
top_names = [feature_names[i] for i in top_idx]
top_vals  = mean_shap[top_idx]

sep = "=" * 62
print(f"\n{sep}")
print("  TABLE 4.2 — TOP 10 FEATURES (SHAP Mean |value|)")
print(sep)
print(f"  {'Rank':<5} {'Feature':<38} {'SHAP':>10}")
print(f"  {'-'*5} {'-'*38} {'-'*10}")
for r, (n, v) in enumerate(zip(top_names, top_vals), 1):
    disp = n[:36] + ".." if len(n) > 38 else n
    print(f"  {r:<5} {disp:<38} {v:>10.5f}")
print(sep)

print("\n  [4/4] Ve bieu do...")

# Bar chart
fig, ax = plt.subplots(figsize=(10, 6))
colors  = ["#c0392b" if v >= top_vals.mean() else "#2980b9"
           for v in top_vals]
ax.barh(range(10), top_vals[::-1], color=colors[::-1],
        edgecolor="white", height=0.7)
ax.set_yticks(range(10))
ax.set_yticklabels([n[:45] for n in top_names[::-1]], fontsize=9)
ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
ax.set_title("Figure 4.1 — Top 10 Features (SHAP Importance)\n"
             "XGBoost on EMBER 2018", fontsize=12, fontweight="bold")
ax.axvline(top_vals.mean(), color="gray", linestyle="--",
           alpha=0.8, label="Mean")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
bar_path = os.path.join(FIG_DIR, "shap_bar.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  OK: {bar_path}")

# Beeswarm
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_sample,
                  feature_names=feature_names,
                  max_display=10, show=False)
plt.title("Figure 4.2 — SHAP Beeswarm (Top 10)\n"
          "Red=Malware  Blue=Benign",
          fontsize=11, fontweight="bold", pad=12)
plt.tight_layout()
bee_path = os.path.join(FIG_DIR, "shap_bee.png")
plt.savefig(bee_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  OK: {bee_path}")

# JSON
json.dump({
    "model": "XGBoost", "dataset": "EMBER 2018",
    "n_shap_samples": int(len(X_sample)),
    "top_10": [{"rank": i+1, "name": n, "shap_mean_abs": round(float(v), 6)}
               for i, (n, v) in enumerate(zip(top_names, top_vals))],
}, open(os.path.join(THIS_DIR, "shap_results.json"), "w"), indent=2)

print(f"\n  shap_results.json")
print(f"  figures/shap_bar.png  (Figure 4.1)")
print(f"  figures/shap_bee.png  (Figure 4.2)")
print(f"\n  Buoc tiep theo: py llm_eval.py")