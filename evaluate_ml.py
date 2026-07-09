"""
==============================================================================
EVALUATE — Đánh giá XGBoost trên Test Set
==============================================================================
Chạy SAU train_ml.py. Output = Table 4.1 trong luận văn.
Chạy: py evaluate_ml.py
==============================================================================
"""

import os, json, gc, time
import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              confusion_matrix, classification_report)

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = r"C:\new\malware_project\ember_data\ember2018"
N_FEATURES = 2351
MODEL_PATH = os.path.join(THIS_DIR, "xgboost_ember_layer1.json")
CACHE_DIR  = os.path.join(THIS_DIR, "xgb_cache")
SVM_TEST   = os.path.join(CACHE_DIR, "test.svm")

print("=" * 60)
print("  EVALUATE — XGBoost trên EMBER Test Set")
print("=" * 60)

# ── Load model ────────────────────────────────────────────────────────────
model = xgb.Booster()
model.load_model(MODEL_PATH)
print(f"\n  ✅ Model loaded: {MODEL_PATH}")

# ── Đọc y_test (nhỏ, OK) ─────────────────────────────────────────────────
y_path = os.path.join(DATA_DIR, "y_test.dat")
X_path = os.path.join(DATA_DIR, "X_test.dat")

y_all       = np.fromfile(y_path, dtype=np.float32)
labeled_idx = np.where(y_all != -1)[0]
y_true      = y_all[labeled_idx].astype(int)
n_labeled   = len(y_true)
n_total     = len(y_all)

print(f"\n  Test set: {n_labeled:,} mẫu có nhãn")
print(f"  Malware : {(y_true==1).sum():,} | Benign: {(y_true==0).sum():,}")
del y_all; gc.collect()

# ── Ghi SVM test file nếu chưa có ────────────────────────────────────────
CHUNK = 25_000

def write_svm_test(global_indices, labels, out_path):
    n = len(global_indices)
    print(f"\n  Ghi test SVM ({n:,} mẫu)...")
    with open(out_path, "w") as fout:
        for start in range(0, n, CHUNK):
            end     = min(start + CHUNK, n)
            chunk_i = global_indices[start:end]

            row_start = int(chunk_i[0])
            row_end   = int(chunk_i[-1]) + 1
            X_mm = np.memmap(X_path, dtype=np.float32, mode='r',
                             offset=row_start * N_FEATURES * 4,
                             shape=(row_end - row_start, N_FEATURES))
            X_chunk = np.array(X_mm[chunk_i - row_start], dtype=np.float32)
            del X_mm; gc.collect()

            y_chunk = labels[start:end]
            for i in range(len(y_chunk)):
                parts = [str(int(y_chunk[i]))]
                for j, v in enumerate(X_chunk[i]):
                    if v != 0.0:
                        parts.append(f"{j}:{v:.6g}")
                fout.write(" ".join(parts) + "\n")

            del X_chunk; gc.collect()
            print(f"    {min(end,n):>6,}/{n:,} ({end/n*100:.0f}%)", end="\r")
    print(f"    ✅ Xong                          ")

os.makedirs(CACHE_DIR, exist_ok=True)
if os.path.exists(SVM_TEST) and os.path.getsize(SVM_TEST) > 100_000:
    print("\n  ✅ test.svm đã có — bỏ qua ghi")
else:
    write_svm_test(labeled_idx, y_true, SVM_TEST)

# ── Predict ───────────────────────────────────────────────────────────────
print("\n  Đang predict...")
dtest  = xgb.DMatrix(SVM_TEST + "?format=libsvm")
t0     = time.time()
y_prob = model.predict(dtest)
infer_time = (time.time() - t0) / n_labeled * 1000  # ms/sample

y_pred = (y_prob >= 0.5).astype(int)

# ── Metrics ───────────────────────────────────────────────────────────────
acc  = accuracy_score(y_true, y_pred)
f1m  = f1_score(y_true, y_pred, average="macro")
f1b  = f1_score(y_true, y_pred, average="binary")
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true, y_pred)
auc  = roc_auc_score(y_true, y_prob)
cm   = confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn)   # False Positive Rate — quan trọng cho SOC

# ── In kết quả ───────────────────────────────────────────────────────────
sep = "=" * 60
print(f"\n{sep}")
print("  TABLE 4.1 — XGBoost TRÊN EMBER 2018 TEST SET")
print(f"  (Dùng trực tiếp trong báo cáo — Section 4.1)")
print(sep)
print(f"\n  Accuracy          : {acc:.4f}  ({acc:.1%})")
print(f"  F1-Score (Macro)  : {f1m:.4f}  ({f1m:.1%})")
print(f"  F1-Score (Binary) : {f1b:.4f}  ({f1b:.1%})")
print(f"  Precision         : {prec:.4f}  ({prec:.1%})")
print(f"  Recall            : {rec:.4f}  ({rec:.1%})")
print(f"  ROC-AUC           : {auc:.4f}")
print(f"  False Positive Rate: {fpr:.4f}  ({fpr:.1%})")
print(f"  Inference speed   : {infer_time:.4f} ms/sample")
print(f"\n  Confusion Matrix:")
print(f"                   Predicted Benign  Predicted Malware")
print(f"  Actual Benign  : {tn:>15,}  {fp:>17,}")
print(f"  Actual Malware : {fn:>15,}  {tp:>17,}")
print(sep)

# ── Ý nghĩa cho SOC ──────────────────────────────────────────────────────
print(f"\n  📌 Ý NGHĨA CHO SOC ANALYST:")
print(f"  • Cứ 100 file sạch → model cảnh báo nhầm {fpr*100:.1f} file")
print(f"  • Cứ 100 malware   → model bắt được {rec*100:.1f} file")
print(f"  • Tốc độ: {1000/infer_time:.0f} mẫu/giây "
      f"({60000/infer_time:.0f} mẫu/phút)")

# ── Lưu kết quả ──────────────────────────────────────────────────────────
results = {
    "model"          : "XGBoost",
    "dataset"        : "EMBER 2018 Test Set",
    "n_test"         : int(n_labeled),
    "accuracy"       : round(float(acc),  4),
    "f1_macro"       : round(float(f1m),  4),
    "f1_binary"      : round(float(f1b),  4),
    "precision"      : round(float(prec), 4),
    "recall"         : round(float(rec),  4),
    "roc_auc"        : round(float(auc),  4),
    "false_positive_rate": round(float(fpr), 4),
    "infer_ms_sample": round(float(infer_time), 4),
    "confusion_matrix": cm.tolist(),
}

out_path = os.path.join(THIS_DIR, "eval_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\n  📄 Kết quả lưu → eval_results.json")
print(f"\n  Bước tiếp theo: py shap_analysis.py")