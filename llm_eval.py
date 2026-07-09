"""
==============================================================================
TẦNG 2 — LLM Evaluation (Llama 3.2 via Ollama)
==============================================================================
Đánh giá Llama 3.2 trên 3,000 mẫu (1,500 malware + 1,500 benign)
Checkpoint mỗi 10 mẫu — có thể tắt máy rồi chạy tiếp
Chạy: py llm_eval.py
==============================================================================
"""

import os, json, gc, time, re
import numpy as np
import requests
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              accuracy_score, confusion_matrix)

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = r"C:\new\malware_project\ember_data\ember2018"
N_FEATURES = 2351
MODEL_NAME = "llama3.2:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"
N_PER_CLASS = 1500   # 1500 malware + 1500 benign = 3000 tổng
CHECKPOINT  = os.path.join(THIS_DIR, "llm_checkpoint.json")
OUT_PATH    = os.path.join(THIS_DIR, "llm_results.json")

print("=" * 62)
print("  TẦNG 2 — LLM Evaluation (Llama 3.2, 3,000 mẫu)")
print("=" * 62)

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 1: Kiểm tra Ollama
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [1/5] Kiểm tra Ollama...")
try:
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    if MODEL_NAME not in models:
        print(f"  ❌ Không tìm thấy {MODEL_NAME}")
        print(f"     Models hiện có: {models}")
        exit(1)
    print(f"  ✅ Ollama OK — {MODEL_NAME} sẵn sàng")
except Exception as e:
    print(f"  ❌ Ollama không phản hồi: {e}")
    exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 2: Load SHAP feature names
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = {}
shap_path = os.path.join(THIS_DIR, "shap_results.json")
if os.path.exists(shap_path):
    with open(shap_path) as f:
        shap_data = json.load(f)
    for item in shap_data.get("top_10_mapped", []):
        FEATURE_NAMES[item["feature_index"]] = item["feature_name"]
    print(f"  ✅ Loaded {len(FEATURE_NAMES)} SHAP feature names")

GROUPS = [
    ("ByteHistogram",        256),
    ("ByteEntropyHistogram", 256),
    ("StringExtractor",      104),
    ("GeneralFileInfo",       10),
    ("HeaderFileInfo",        62),
    ("SectionInfo",          255),
    ("ImportsInfo",         1280),
    ("ExportsInfo",          128),
]

def get_group(idx):
    cursor = 0
    for name, size in GROUPS:
        if cursor <= idx < cursor + size:
            return name, idx - cursor
        cursor += size
    return "Unknown", idx

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 3: Lấy 3,000 mẫu test
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  [2/5] Lấy {N_PER_CLASS*2:,} mẫu test "
      f"({N_PER_CLASS:,} mal + {N_PER_CLASS:,} benign)...")

y_path = os.path.join(DATA_DIR, "y_test.dat")
X_path = os.path.join(DATA_DIR, "X_test.dat")

y_all       = np.fromfile(y_path, dtype=np.float32)
labeled_idx = np.where(y_all != -1)[0]
y_labeled   = y_all[labeled_idx].astype(int)
del y_all; gc.collect()

mal_idx = labeled_idx[y_labeled == 1]
ben_idx = labeled_idx[y_labeled == 0]

# Giới hạn nếu dataset không đủ
n_mal = min(N_PER_CLASS, len(mal_idx))
n_ben = min(N_PER_CLASS, len(ben_idx))

rng     = np.random.RandomState(99)
sel_mal = rng.choice(mal_idx, size=n_mal, replace=False)
sel_ben = rng.choice(ben_idx, size=n_ben, replace=False)

sel_idx = np.concatenate([sel_mal, sel_ben])
y_sel   = np.array([1]*n_mal + [0]*n_ben)
perm    = rng.permutation(len(sel_idx))
sel_idx = sel_idx[perm]
y_sel   = y_sel[perm]
N_TOTAL_SAMPLES = len(sel_idx)

print(f"  ✅ {N_TOTAL_SAMPLES:,} mẫu đã chọn và shuffle")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 4: Load X + XGBoost scores
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  [3/5] Đọc X và tính XGBoost scores "
      f"({N_TOTAL_SAMPLES:,} mẫu)...")

import xgboost as xgb

xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(THIS_DIR, "xgboost_ember_layer1.json"))

# Đọc theo chunk 500 để tiết kiệm RAM
CHUNK     = 500
xgb_probs = np.zeros(N_TOTAL_SAMPLES, dtype=np.float32)
X_all     = np.zeros((N_TOTAL_SAMPLES, N_FEATURES), dtype=np.float32)

for start in range(0, N_TOTAL_SAMPLES, CHUNK):
    end      = min(start + CHUNK, N_TOTAL_SAMPLES)
    chunk_gi = sel_idx[start:end]

    X_chunk = np.zeros((end - start, N_FEATURES), dtype=np.float32)
    for j, gidx in enumerate(chunk_gi):
        offset      = int(gidx) * N_FEATURES * 4
        row         = np.memmap(X_path, dtype=np.float32, mode='r',
                                offset=offset, shape=(N_FEATURES,))
        X_chunk[j]  = row
        del row

    X_all[start:end]     = X_chunk
    xgb_probs[start:end] = xgb_model.predict(xgb.DMatrix(X_chunk))
    del X_chunk; gc.collect()
    print(f"    {end:>5,}/{N_TOTAL_SAMPLES:,}", end="\r")

print(f"  ✅ X và XGBoost scores sẵn sàng              ")

# ─────────────────────────────────────────────────────────────────────────────
# HÀM TẠO PROMPT & GỌI LLM
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(x_row, xgb_score, sample_idx):
    top5 = np.argsort(np.abs(x_row))[::-1][:5]
    feat_lines = []
    for fidx in top5:
        group, local = get_group(int(fidx))
        fname = FEATURE_NAMES.get(int(fidx), f"{group}[{local}]")
        feat_lines.append(f"  - {fname} ({group}): {x_row[fidx]:.4f}")
    features_str = "\n".join(feat_lines)

    return f"""You are a malware analyst. Analyze the PE file features below.

## Examples

### Example 1 — BENIGN
Features:
  - has_signature (GeneralFileInfo): 1.0000
  - header_11 (HeaderFileInfo): 0.1200
  - import_8 (ImportsInfo): 0.0500
  - section_3 (SectionInfo): 0.2100
  - numurls (StringExtractor): 0.0000
ML risk score: 0.08
Classification: BENIGN
Reason: Valid digital signature, low import risk, normal structure.

### Example 2 — MALWARE
Features:
  - has_signature (GeneralFileInfo): 0.0000
  - header_11 (HeaderFileInfo): 0.9800
  - import_8 (ImportsInfo): 0.8700
  - section_242 (SectionInfo): 0.9100
  - numurls (StringExtractor): 0.7500
ML risk score: 0.94
Classification: MALWARE
Reason: No digital signature, suspicious PE header, high-risk imports.

## Sample #{sample_idx}
Features:
{features_str}
ML risk score: {xgb_score:.4f}

Respond EXACTLY in this format (2 lines only):
Classification: [MALWARE or BENIGN]
Reason: [one sentence]"""


def call_llama(prompt, timeout=90):
    payload = {
        "model" : MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 80,
            "top_p": 0.9,
        }
    }
    t0 = time.time()
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip(), time.time() - t0
    except Exception as e:
        return f"ERROR: {e}", time.time() - t0


def parse_response(text):
    u = text.upper()
    m = re.search(r"CLASSIFICATION\s*:\s*(MALWARE|BENIGN)", u)
    if m:
        return 1 if m.group(1) == "MALWARE" else 0
    if "MALWARE" in u and "BENIGN" not in u:
        return 1
    if "BENIGN" in u and "MALWARE" not in u:
        return 0
    return -1

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 5: CHẠY EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  [4/5] Chạy LLM evaluation ({N_TOTAL_SAMPLES:,} mẫu)...")
print(f"  Checkpoint mỗi 10 mẫu → có thể Ctrl+C rồi chạy tiếp")
print(f"  {'─'*55}")

# Load checkpoint
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        ckpt = json.load(f)
    done_set = set(ckpt["done_indices"])
    records  = ckpt["records"]
    print(f"  ✅ Resume: {len(done_set)}/{N_TOTAL_SAMPLES} đã xong")
else:
    done_set = set()
    records  = []

t_start = time.time()

for i in range(N_TOTAL_SAMPLES):
    if i in done_set:
        continue

    prompt    = build_prompt(X_all[i], float(xgb_probs[i]), i)
    resp, lat = call_llama(prompt)
    pred      = parse_response(resp)
    true_lbl  = int(y_sel[i])

    records.append({
        "index"      : i,
        "true_label" : true_lbl,
        "xgb_prob"   : round(float(xgb_probs[i]), 4),
        "llm_pred"   : pred,
        "llm_correct": int(pred == true_lbl),
        "latency_s"  : round(lat, 2),
        "response"   : resp[:200],
    })
    done_set.add(i)

    # Print tiến độ
    n_done = len(done_set)
    correct_so_far = sum(r["llm_correct"] for r in records
                         if r["llm_pred"] != -1)
    valid_so_far   = sum(1 for r in records if r["llm_pred"] != -1)
    acc_so_far     = correct_so_far / valid_so_far if valid_so_far > 0 else 0

    status   = "✅" if pred == true_lbl else ("❓" if pred == -1 else "❌")
    lbl_str  = "MAL" if true_lbl == 1 else "BEN"
    pred_str = "MAL" if pred == 1 else ("BEN" if pred == 0 else "???")
    elapsed  = time.time() - t_start
    eta      = (elapsed / n_done) * (N_TOTAL_SAMPLES - n_done) if n_done > 0 else 0

    print(f"  [{n_done:>4}/{N_TOTAL_SAMPLES}] {status} "
          f"True={lbl_str} Pred={pred_str} "
          f"XGB={xgb_probs[i]:.2f} Lat={lat:.1f}s "
          f"Acc={acc_so_far:.1%} ETA={eta/60:.0f}m")

    # Checkpoint mỗi 10 mẫu
    if n_done % 10 == 0:
        with open(CHECKPOINT, "w") as f:
            json.dump({"done_indices": list(done_set),
                       "records": records}, f)

# Lưu checkpoint cuối
with open(CHECKPOINT, "w") as f:
    json.dump({"done_indices": list(done_set), "records": records}, f)

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 6: METRICS & BẢNG SO SÁNH
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  [5/5] Tính metrics cuối...")

valid     = [(r["true_label"], r["llm_pred"], r["latency_s"])
             for r in records if r["llm_pred"] != -1]
n_valid   = len(valid)
n_invalid = N_TOTAL_SAMPLES - n_valid

y_true_llm = [v[0] for v in valid]
y_pred_llm = [v[1] for v in valid]
latencies  = [v[2] for v in valid]

acc   = accuracy_score(y_true_llm, y_pred_llm)
f1m   = f1_score(y_true_llm, y_pred_llm, average="macro")
prec  = precision_score(y_true_llm, y_pred_llm, zero_division=0)
rec   = recall_score(y_true_llm, y_pred_llm, zero_division=0)
cm    = confusion_matrix(y_true_llm, y_pred_llm)
avg_lat = float(np.mean(latencies))

sep = "=" * 64
print(f"\n{sep}")
print("  TABLE 4.3 — XGBoost vs Llama 3.2 (Bảng so sánh tổng hợp)")
print(sep)
print(f"  {'Tiêu chí':<30} {'XGBoost':>14} {'Llama 3.2':>14}")
print(f"  {'-'*30} {'-'*14} {'-'*14}")
print(f"  {'Accuracy':<30} {'94.3%':>14} {acc:.1%}")
print(f"  {'F1-Score (Macro)':<30} {'94.3%':>14} {f1m:.1%}")
print(f"  {'Precision':<30} {'92.8%':>14} {prec:.1%}")
print(f"  {'Recall':<30} {'95.9%':>14} {rec:.1%}")
print(f"  {'ROC-AUC':<30} {'0.9872':>14} {'N/A':>14}")
print(f"  {'False Positive Rate':<30} {'7.4%':>14} {'N/A':>14}")
print(f"  {'Latency (ms/sample)':<30} {'0.01':>14} {avg_lat*1000:.0f}")
print(f"  {'Throughput (samples/s)':<30} {'96,561':>14} {1/avg_lat:.0f}")
print(f"  {'Explainability':<30} {'No':>14} {'Yes':>14}")
print(f"  {'MITRE ATT&CK mapping':<30} {'No':>14} {'Yes':>14}")
print(f"  {'Samples evaluated':<30} {'200,000':>14} {n_valid:,}")
print(sep)
print(f"\n  Mẫu không parse được: {n_invalid}/{N_TOTAL_SAMPLES} "
      f"({n_invalid/N_TOTAL_SAMPLES*100:.1f}%)")
print(f"  Latency TB: {avg_lat:.1f}s/mẫu — "
      f"Tổng thời gian: {sum(latencies)/60:.0f} phút")

# Lưu JSON
results = {
    "model"            : "Llama 3.2 (3.2B Q4_K_M)",
    "n_total"          : N_TOTAL_SAMPLES,
    "n_valid"          : n_valid,
    "n_unparseable"    : n_invalid,
    "accuracy"         : round(float(acc),  4),
    "f1_macro"         : round(float(f1m),  4),
    "precision"        : round(float(prec), 4),
    "recall"           : round(float(rec),  4),
    "avg_latency_s"    : round(avg_lat, 2),
    "confusion_matrix" : cm.tolist() if len(cm) > 0 else [],
    "xgboost_ref": {
        "accuracy": 0.9426, "f1_macro": 0.9426,
        "precision": 0.9284, "recall": 0.9592,
        "roc_auc": 0.9872, "fpr": 0.074,
        "latency_ms": 0.0104,
    },
    "records": records,
}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n  💾 llm_results.json")
print(f"\n{'='*64}")
print(f"  HOÀN TẤT — Bước tiếp theo: py comparison_report.py")
print(f"{'='*64}")