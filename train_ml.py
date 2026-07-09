"""
==============================================================================
TẦNG 1 — XGBoost Training (Chunked mode — 8GB RAM safe)
==============================================================================
Giải pháp cho OSError WinError 8:
  Không load toàn bộ X_train vào RAM cùng lúc.
  Đọc label trước (3MB) → tách train/val index →
  ghi SVM file từng chunk → train từ SVM (external memory).
==============================================================================
"""

import os, json, time, gc
import numpy as np
import xgboost as xgb

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = r"C:\new\malware_project\ember_data\ember2018"
N_FEATURES = 2351
N_TOTAL    = 800_000
MODEL_OUT  = os.path.join(THIS_DIR, "xgboost_ember_layer1.json")
CACHE_DIR  = os.path.join(THIS_DIR, "xgb_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 60)
print("  TẦNG 1 — XGBoost (Chunked Mode, 8GB RAM safe)")
print("=" * 60)

# ── BƯỚC 1: Đọc labels (chỉ 3MB) ──────────────────────────────────────────
print("\n  [1/5] Đọc y_train.dat (~3MB)...")
y_path = os.path.join(DATA_DIR, "y_train.dat")
X_path = os.path.join(DATA_DIR, "X_train.dat")

y_all       = np.fromfile(y_path, dtype=np.float32)
labeled_idx = np.where(y_all != -1)[0]
y_labeled   = y_all[labeled_idx]
n_labeled   = len(y_labeled)

print(f"  Labeled: {n_labeled:,} | Malware: {(y_labeled==1).sum():,} "
      f"| Benign: {(y_labeled==0).sum():,}")
del y_all; gc.collect()

# ── BƯỚC 2: Tách train/val 90/10 ──────────────────────────────────────────
print("\n  [2/5] Tách train/val index...")
rng        = np.random.RandomState(42)
perm       = rng.permutation(n_labeled)
n_val      = int(n_labeled * 0.1)
val_pos    = perm[:n_val]
tr_pos     = perm[n_val:]

val_global = np.sort(labeled_idx[val_pos])
tr_global  = np.sort(labeled_idx[tr_pos])
y_tr       = y_labeled[tr_pos[np.argsort(tr_pos)]]
y_val      = y_labeled[val_pos[np.argsort(val_pos)]]

print(f"  Train: {len(tr_global):,} | Val: {len(val_global):,}")

# ── BƯỚC 3: Ghi SVM file từng chunk ───────────────────────────────────────
CHUNK     = 25_000   # ~275MB/chunk — giảm xuống 15_000 nếu vẫn lỗi RAM

SVM_TRAIN = os.path.join(CACHE_DIR, "train.svm")
SVM_VAL   = os.path.join(CACHE_DIR, "val.svm")

def write_svm(global_indices, labels, out_path, desc):
    n = len(global_indices)
    print(f"  Ghi {desc} ({n:,} mẫu) → {os.path.basename(out_path)}")
    with open(out_path, "w") as fout:
        for start in range(0, n, CHUNK):
            end     = min(start + CHUNK, n)
            chunk_i = global_indices[start:end]

            # Memmap chỉ đoạn cần đọc — tiết kiệm RAM
            row_start = int(chunk_i[0])
            row_end   = int(chunk_i[-1]) + 1
            X_mm = np.memmap(X_path, dtype=np.float32, mode='r',
                             offset=row_start * N_FEATURES * 4,
                             shape=(row_end - row_start, N_FEATURES))

            local_i = chunk_i - row_start
            X_chunk = np.array(X_mm[local_i], dtype=np.float32)
            del X_mm; gc.collect()

            y_chunk = labels[start:end]
            for i in range(len(y_chunk)):
                lbl   = int(y_chunk[i])
                parts = [str(lbl)]
                for j, v in enumerate(X_chunk[i]):
                    if v != 0.0:
                        parts.append(f"{j}:{v:.6g}")
                fout.write(" ".join(parts) + "\n")

            del X_chunk; gc.collect()
            print(f"    {min(end,n):>7,}/{n:,} ({end/n*100:.0f}%)", end="\r")
    print(f"    ✅ Xong                                    ")

print("\n  [3/5] Ghi SVM files (bước này chậm ~10-20 phút)...")

if os.path.exists(SVM_TRAIN) and os.path.getsize(SVM_TRAIN) > 100_000:
    print("  ✅ SVM files đã có — bỏ qua")
else:
    write_svm(tr_global,  y_tr,  SVM_TRAIN, "train")
    write_svm(val_global, y_val, SVM_VAL,   "val")

# ── BƯỚC 4: Load DMatrix ──────────────────────────────────────────────────
print("\n  [4/5] Load DMatrix từ SVM files...")
dtrain = xgb.DMatrix(SVM_TRAIN + "?format=libsvm")
dval   = xgb.DMatrix(SVM_VAL   + "?format=libsvm")
print(f"  dtrain: {dtrain.num_row():,} | dval: {dval.num_row():,}")

# ── BƯỚC 5: Train ─────────────────────────────────────────────────────────
params = {
    "objective"        : "binary:logistic",
    "eval_metric"      : ["logloss", "auc"],
    "tree_method"      : "hist",
    "max_bin"          : 256,
    "max_depth"        : 6,
    "learning_rate"    : 0.05,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "min_child_weight" : 5,
    "gamma"            : 0.1,
    "nthread"          : 4,
    "seed"             : 42,
    "verbosity"        : 1,
}

print(f"\n  [5/5] Train XGBoost (tree_method=hist)...")
evals_result = {}
t0 = time.time()

model = xgb.train(
    params, dtrain,
    num_boost_round       = 300,
    evals                 = [(dtrain, "train"), (dval, "val")],
    early_stopping_rounds = 30,
    evals_result          = evals_result,
    verbose_eval          = 50,
)

train_time = time.time() - t0
best_auc   = max(evals_result["val"]["auc"])

print(f"\n  ✅ Train xong: {train_time:.0f}s ({train_time/60:.1f} phút)")
print(f"  Best round  : {model.best_iteration}")
print(f"  Best val AUC: {best_auc:.4f}")

model.save_model(MODEL_OUT)
json.dump({
    "n_features": N_FEATURES, "n_train": int(dtrain.num_row()),
    "best_round": int(model.best_iteration),
    "best_val_auc": round(best_auc, 4),
    "train_time_s": round(train_time, 1), "params": params,
}, open(os.path.join(THIS_DIR, "training_results.json"), "w"), indent=2)

print(f"\n  💾 Model  → {MODEL_OUT}")
print(f"  📄 Meta   → training_results.json")
print(f"\n{'='*60}")
print(f"  XONG — Bước tiếp theo: py evaluate_ml.py")
print(f"{'='*60}")