"""
==============================================================================
MASTER CHECKLIST — Kiểm tra toàn bộ pipeline đã chạy đủ chưa
==============================================================================
Chạy 1 lần để biết chính xác còn thiếu file nào, cần chạy script nào tiếp.
Chạy: py check_pipeline.py
==============================================================================
"""

import os, json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r"C:\new\malware_project\ember_data\ember2018"

def fmt_size(path):
    if not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    if size > 1024**3:
        return f"{size/1024**3:.1f} GB"
    elif size > 1024**2:
        return f"{size/1024**2:.1f} MB"
    else:
        return f"{size/1024:.0f} KB"

def check(path, label, min_size_kb=0, script_to_run=""):
    exists = os.path.exists(path)
    size_str = fmt_size(path) if exists else "—"
    ok = exists and (os.path.getsize(path) >= min_size_kb * 1024 if min_size_kb else True)
    status = "✅" if ok else "❌"
    print(f"  {status} {label:<35} {size_str:>10}")
    if not ok and script_to_run:
        print(f"      → Chạy: py {script_to_run}")
    return ok

print("=" * 68)
print("  MASTER CHECKLIST — Hybrid SOC Pipeline")
print("=" * 68)

# ── NHÓM 1: Dữ liệu gốc ─────────────────────────────────────────
print("\n📦 NHÓM 1 — Dữ liệu gốc (EMBER 2018)")
g1 = [
    check(os.path.join(DATA_DIR, "X_train.dat"), "X_train.dat", 1000000, "(tải lại EMBER)"),
    check(os.path.join(DATA_DIR, "y_train.dat"), "y_train.dat", 100, "(tải lại EMBER)"),
    check(os.path.join(DATA_DIR, "X_test.dat"),  "X_test.dat",  500000, "(tải lại EMBER)"),
    check(os.path.join(DATA_DIR, "y_test.dat"),  "y_test.dat",  10, "(tải lại EMBER)"),
]

# ── NHÓM 2: Model ───────────────────────────────────────────────
print("\n🤖 NHÓM 2 — XGBoost Model")
g2 = [
    check(os.path.join(THIS_DIR, "xgboost_ember_layer1.json"), "xgboost_ember_layer1.json", 100, "train_ml.py"),
    check(os.path.join(THIS_DIR, "training_results.json"), "training_results.json", 0, "train_ml.py"),
]

# ── NHÓM 3: Evaluation ──────────────────────────────────────────
print("\n📊 NHÓM 3 — Đánh giá Test Set")
g3 = [
    check(os.path.join(THIS_DIR, "eval_results.json"), "eval_results.json", 0, "evaluate_ml.py"),
]

# ── NHÓM 4: SHAP ────────────────────────────────────────────────
print("\n🔬 NHÓM 4 — SHAP Analysis")
shap_path = os.path.join(THIS_DIR, "shap_results.json")
g4a = check(shap_path, "shap_results.json", 0, "shap_analysis.py")
g4b = check(os.path.join(THIS_DIR, "figures", "shap_bar.png"), "figures/shap_bar.png", 0, "shap_analysis.py")
g4c = check(os.path.join(THIS_DIR, "figures", "shap_bee.png"), "figures/shap_bee.png", 0, "shap_analysis.py")

# Check riêng: feature names đã map chưa
mapped_ok = False
if g4a:
    with open(shap_path) as f:
        sd = json.load(f)
    has_mapped = "top_10_mapped" in sd
    mapped_ok = has_mapped
    status = "✅" if has_mapped else "❌"
    print(f"  {status} Feature names đã map (top_10_mapped)")
    if not has_mapped:
        print(f"      → Chạy: py map_features.py")

# ── NHÓM 5: LLM ─────────────────────────────────────────────────
print("\n🦙 NHÓM 5 — LLM Evaluation (Llama 3.2)")
llm_path = os.path.join(THIS_DIR, "llm_results.json")
ckpt_path = os.path.join(THIS_DIR, "llm_checkpoint.json")

g5a = check(llm_path, "llm_results.json", 0, "llm_eval.py")
llm_complete = False
if g5a:
    with open(llm_path) as f:
        ld = json.load(f)
    n_total = ld.get("n_total", 0)
    n_valid = ld.get("n_valid", 0)
    llm_complete = n_total >= 3000
    status = "✅" if llm_complete else "⚠️ "
    print(f"  {status} Tổng số mẫu đã chạy: {n_total}/3000")
    if not llm_complete:
        print(f"      → CHƯA HOÀN TẤT — chạy lại: py llm_eval.py (sẽ tự resume)")

if os.path.exists(ckpt_path):
    with open(ckpt_path) as f:
        ck = json.load(f)
    n_done = len(ck.get("done_indices", []))
    print(f"  ℹ️  Checkpoint: {n_done}/3000 mẫu đã lưu")

# ── NHÓM 6: So sánh ─────────────────────────────────────────────
print("\n📈 NHÓM 6 — So sánh tổng hợp")
g6 = [
    check(os.path.join(THIS_DIR, "comparison_table.json"), "comparison_table.json", 0, "comparison_report.py"),
    check(os.path.join(THIS_DIR, "fair_comparison.json"), "fair_comparison.json", 0, "fair_comparison.py / improve_report.py"),
    check(os.path.join(THIS_DIR, "figures", "comparison_bar.png"), "figures/comparison_bar.png", 0, "comparison_report.py"),
    check(os.path.join(THIS_DIR, "figures", "cm_comparison.png"), "figures/cm_comparison.png", 0, "improve_report.py"),
]

# ── NHÓM 7: Threshold + Demo ─────────────────────────────────────
print("\n🎯 NHÓM 7 — Threshold Experiment + Demo Samples")
g7 = [
    check(os.path.join(THIS_DIR, "threshold_results.json"), "threshold_results.json", 0, "threshold_experiment.py"),
    check(os.path.join(THIS_DIR, "figures", "threshold_analysis.png"), "figures/threshold_analysis.png", 0, "threshold_experiment.py"),
    check(os.path.join(THIS_DIR, "demo_samples.json"), "demo_samples.json", 0, "find_demo_samples.py"),
]

# ── TỔNG KẾT ──────────────────────────────────────────────────────
print("\n" + "=" * 68)
all_groups = {
    "Nhóm 1 — Dữ liệu gốc": all(g1),
    "Nhóm 2 — XGBoost Model": all(g2),
    "Nhóm 3 — Evaluation": all(g3),
    "Nhóm 4 — SHAP": g4a and g4b and g4c and mapped_ok,
    "Nhóm 5 — LLM": g5a and llm_complete,
    "Nhóm 6 — So sánh": all(g6),
    "Nhóm 7 — Threshold/Demo": all(g7),
}

print("  TÓM TẮT TRẠNG THÁI:")
for name, ok in all_groups.items():
    print(f"    {'✅' if ok else '❌'} {name}")

all_ok = all(all_groups.values())
print("=" * 68)
if all_ok:
    print("  🎉 PIPELINE HOÀN CHỈNH — Sẵn sàng nộp báo cáo!")
else:
    missing = [n for n, ok in all_groups.items() if not ok]
    print(f"  ⚠️  CÒN THIẾU {len(missing)} NHÓM — xem chi tiết phía trên")
print("=" * 68)