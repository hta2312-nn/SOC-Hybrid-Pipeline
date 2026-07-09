"""
DEMO — Hybrid SOC Pipeline (1 mẫu end-to-end)
Chạy: py demo_pipeline.py
"""

import os, json, gc, sys
import numpy as np
import xgboost as xgb
import requests, time

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = r"C:\new\malware_project\ember_data\ember2018"
N_FEATURES = 2351
MODEL_PATH = os.path.join(THIS_DIR, "xgboost_ember_layer1.json")
SHAP_PATH  = os.path.join(THIS_DIR, "shap_results.json")
DEMO_PATH  = os.path.join(THIS_DIR, "demo_samples.json")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:latest"

# ── EMBER feature names đầy đủ ────────────────────────────────────────────
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

# Tên cụ thể cho GeneralFileInfo (10 features)
GENERAL_NAMES = [
    "file_size", "vsize", "has_debug", "exports", "imports",
    "has_relocations", "has_resources", "has_signature", "has_tls", "symbols"
]

# Tên cụ thể cho HeaderFileInfo (một số quan trọng)
HEADER_NAMES = {
    0: "machine_type", 1: "timestamp", 2: "compile_time",
    3: "num_sections", 4: "pointer_to_symbol_table",
    5: "num_symbols", 6: "size_of_optional_header",
    7: "characteristics", 8: "magic", 9: "major_linker_version",
    10: "minor_linker_version", 11: "size_of_code",
}

def get_feature_name(idx: int) -> tuple:
    """Trả về (tên_đẹp, nhóm) cho feature index."""
    cursor = 0
    for group_name, size in GROUPS:
        if cursor <= idx < cursor + size:
            local = idx - cursor
            if group_name == "ByteHistogram":
                return f"byte_freq[0x{local:02X}]", group_name
            elif group_name == "ByteEntropyHistogram":
                return f"entropy_bin[{local}]", group_name
            elif group_name == "GeneralFileInfo":
                name = GENERAL_NAMES[local] if local < len(GENERAL_NAMES) \
                       else f"general[{local}]"
                return name, group_name
            elif group_name == "HeaderFileInfo":
                name = HEADER_NAMES.get(local, f"header_field[{local}]")
                return name, group_name
            elif group_name == "SectionInfo":
                return f"section_attr[{local}]", group_name
            elif group_name == "ImportsInfo":
                return f"import_hash[{local}]", group_name
            elif group_name == "ExportsInfo":
                return f"export_hash[{local}]", group_name
            elif group_name == "StringExtractor":
                str_names = ["numstrings","avlength","numurls","numregistrykeys",
                             "numpaths","numother","has_debug_str",
                             "has_exports_str","has_mz_header"]
                name = str_names[local] if local < len(str_names) \
                       else f"string_feat[{local}]"
                return name, group_name
            else:
                return f"{group_name}[{local}]", group_name
        cursor += size
    return f"f{idx}", "Unknown"

# ── Load model ────────────────────────────────────────────────────────────
xgb_model = xgb.Booster()
xgb_model.load_model(MODEL_PATH)

y_path = os.path.join(DATA_DIR, "y_test.dat")
X_path = os.path.join(DATA_DIR, "X_test.dat")

y_all       = np.fromfile(y_path, dtype=np.float32)
labeled_idx = np.where(y_all != -1)[0]
y_labeled   = y_all[labeled_idx].astype(int)

# ── Gợi ý index từ demo_samples.json ─────────────────────────────────────
hints = {}
if os.path.exists(DEMO_PATH):
    with open(DEMO_PATH) as f:
        demo_data = json.load(f)
    rec = demo_data.get("recommended", {})
    if rec.get("demo1_malware") is not None:
        hints["malware"] = rec["demo1_malware"]
    if rec.get("demo2_benign") is not None:
        hints["benign"] = rec["demo2_benign"]
    if rec.get("demo3_borderline") is not None:
        hints["borderline"] = rec["demo3_borderline"]

print("\n" + "═"*62)
print("  HYBRID SOC PIPELINE — DEMO")
print("  XGBoost (Tầng 1) + Llama 3.2 (Tầng 2)")
print("═"*62)
print(f"\n  Test set: {len(labeled_idx):,} mẫu  (0=benign, 1=malware)\n")

if hints:
    print("  GỢI Ý INDEX ĐẸP:")
    if "malware" in hints:
        print(f"    🔴 Malware rõ ràng   : {hints['malware']}")
    if "benign" in hints:
        print(f"    🟢 Benign rõ ràng    : {hints['benign']}")
    if "borderline" in hints:
        print(f"    ⚠️  Borderline (XGB sai): {hints['borderline']}")
    print()

choice = input("  Nhập index mẫu (Enter = random): ").strip()
if choice == "":
    rng        = np.random.RandomState()
    sample_pos = rng.randint(0, len(labeled_idx))
else:
    sample_pos = int(choice) % len(labeled_idx)

global_idx = labeled_idx[sample_pos]
true_label = y_labeled[sample_pos]
label_str  = "MALWARE 🔴" if true_label == 1 else "BENIGN  🟢"

print(f"\n  Mẫu #{sample_pos}  (global index: {global_idx})")
print(f"  Nhãn thật : {label_str}")

# Đọc X
offset = int(global_idx) * N_FEATURES * 4
x_row  = np.array(
    np.memmap(X_path, dtype=np.float32, mode='r',
              offset=offset, shape=(N_FEATURES,)),
    dtype=np.float32
)

# ── TẦNG 1: XGBoost ───────────────────────────────────────────────────────
print("\n" + "─"*62)
print("  TẦNG 1 — XGBoost ML Triage")
print("─"*62)

dmat      = xgb.DMatrix(x_row.reshape(1, -1))
xgb_score = float(xgb_model.predict(dmat)[0])
xgb_pred  = "MALWARE 🔴" if xgb_score >= 0.5 else "BENIGN  🟢"

print(f"\n  Risk Score  : {xgb_score:.4f}  ({xgb_score*100:.1f}%)")
print(f"  Phân loại  : {xgb_pred}")
correct_xgb = (xgb_score >= 0.5) == (true_label == 1)
print(f"  Kết quả    : {'✅ ĐÚNG' if correct_xgb else '❌ SAI'}")
print(f"  Thời gian  : < 0.01ms")

if xgb_score < 0.3:
    zone = "SAFE ZONE   → Whitelist"
elif xgb_score > 0.7:
    zone = "DANGER ZONE → Alert ngay"
else:
    zone = "GREY ZONE   → Chuyển Tầng 2 LLM"
print(f"  Vùng       : {zone}")

# Top 5 features
top5_idx = np.argsort(np.abs(x_row))[::-1][:5]
print(f"\n  Top 5 đặc trưng nổi bật:")
top5_info = []
for rank, fidx in enumerate(top5_idx, 1):
    fname, fgroup = get_feature_name(int(fidx))
    fval = float(x_row[fidx])
    # Format giá trị đẹp hơn
    if abs(fval) > 1_000_000:
        fval_str = f"{fval:.2e}"
    else:
        fval_str = f"{fval:.4f}"
    print(f"    {rank}. {fname:<30} ({fgroup:<22}) = {fval_str}")
    top5_info.append((fname, fgroup, fval))

# ── TẦNG 2: Llama 3.2 ────────────────────────────────────────────────────
print("\n" + "─"*62)
print("  TẦNG 2 — Llama 3.2 Analysis")
print("─"*62)

try:
    r = requests.get("http://localhost:11434/api/tags", timeout=3)
    ollama_ok = True
    print("\n  ✅ Ollama sẵn sàng")
except:
    ollama_ok = False
    print("\n  ⚠️  Ollama không phản hồi")

if ollama_ok:
    feat_lines = "\n".join([
        f"  - {name} ({group}): {val:.4f}"
        for name, group, val in top5_info
    ])

    prompt = f"""You are a malware analyst at a SOC. Analyze this Windows PE file.

## Sample Features
{feat_lines}
ML Risk Score: {xgb_score:.4f} ({"HIGH RISK" if xgb_score > 0.7 else "MEDIUM" if xgb_score > 0.3 else "LOW RISK"})

## Examples
### BENIGN example:
Features: has_signature=1.0, file_size=45000, imports=12, section_attr=normal
ML Risk: 0.05
Classification: BENIGN
Reason: Valid digital signature, normal import count, standard PE structure.
MITRE ATT&CK: N/A

### MALWARE example:
Features: has_signature=0.0, size_of_code=890000, import_hash=suspicious, section_attr=high_entropy
ML Risk: 0.91
Classification: MALWARE
Reason: No digital signature, oversized code section, suspicious imports matching process injection tools.
MITRE ATT&CK: T1055 - Process Injection

## Classify this sample:
Classification: [MALWARE or BENIGN]
Risk Assessment: [1-2 sentences explaining why]
Suspicious Features: [which specific features are concerning]
MITRE ATT&CK: [technique ID and name, or N/A]
Recommended Action: [what SOC analyst should do next]"""

    print("  Đang phân tích...\n")
    t0 = time.time()
    try:
        resp    = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_predict": 250, "top_p": 0.9}
        }, timeout=120)
        llm_txt = resp.json().get("response", "").strip()
        lat     = time.time() - t0

        print("─"*62)
        print(llm_txt)
        print("─"*62)
        print(f"\n  Thời gian phân tích: {lat:.1f}s")

        # Parse kết quả LLM
        import re
        m = re.search(r"Classification\s*:\s*(MALWARE|BENIGN)", llm_txt, re.I)
        if m:
            llm_verdict = m.group(1).upper()
            llm_correct = (llm_verdict == "MALWARE") == (true_label == 1)
            print(f"  LLM phán quyết : {llm_verdict} "
                  f"{'✅ ĐÚNG' if llm_correct else '❌ SAI'}")
    except Exception as e:
        print(f"  Lỗi: {e}")
        lat = 0

# ── Tóm tắt ──────────────────────────────────────────────────────────────
print("\n" + "═"*62)
print("  TÓM TẮT KẾT QUẢ HYBRID PIPELINE")
print("═"*62)
print(f"  Nhãn thật      : {label_str}")
print(f"  XGBoost        : {xgb_pred}  (score={xgb_score:.3f})  "
      f"{'✅' if correct_xgb else '❌'}")
print(f"  Vùng phân loại : {zone}")
print(f"\n  Nhấn Enter để thoát.")
input()