"""
TÌM INDEX ĐẸP ĐỂ DEMO — v2
Chạy: py find_demo_samples.py
"""

import os, json
import numpy as np
import xgboost as xgb

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = r"C:\new\malware_project\ember_data\ember2018"
N_FEATURES = 2351
MODEL_PATH = os.path.join(THIS_DIR, "xgboost_ember_layer1.json")

print("Đang tìm index đẹp để demo...\n")

y_all       = np.fromfile(os.path.join(DATA_DIR, "y_test.dat"), dtype=np.float32)
labeled_idx = np.where(y_all != -1)[0]
y_labeled   = y_all[labeled_idx].astype(int)

model = xgb.Booster()
model.load_model(MODEL_PATH)

# Scan 10,000 mẫu để có đủ malware rõ ràng
rng      = np.random.RandomState(42)
scan_pos = rng.choice(len(labeled_idx), size=10000, replace=False)
X_path   = os.path.join(DATA_DIR, "X_test.dat")

X_scan = np.zeros((10000, N_FEATURES), dtype=np.float32)
print("Đọc 10,000 mẫu để scan...")
for i, pos in enumerate(scan_pos):
    gidx         = labeled_idx[pos]
    offset       = int(gidx) * N_FEATURES * 4
    row          = np.memmap(X_path, dtype=np.float32, mode='r',
                             offset=offset, shape=(N_FEATURES,))
    X_scan[i]    = row
    del row
    if (i+1) % 2000 == 0:
        print(f"  {i+1}/10000", end="\r")

scores = model.predict(xgb.DMatrix(X_scan))
y_scan = y_labeled[scan_pos]

# Tìm theo ngưỡng thực tế hơn
clear_mal  = np.where((y_scan == 1) & (scores > 0.80))[0]  # hạ từ 0.95
clear_ben  = np.where((y_scan == 0) & (scores < 0.05))[0]
borderline = np.where((scores > 0.40) & (scores < 0.60))[0]

# Ưu tiên score cao nhất cho malware
clear_mal_sorted = clear_mal[np.argsort(scores[clear_mal])[::-1]]

sep = "=" * 62
print(f"\n{sep}")
print("  KẾT QUẢ — INDEX ĐỂ DEMO")
print(sep)

print(f"\n  🔴 MALWARE RÕ RÀNG (score > 0.80) — {len(clear_mal)} mẫu")
print(f"  {'Index':>8}  {'Score':>7}  {'Nhãn':<12}")
print(f"  {'-'*8}  {'-'*7}  {'-'*12}")
for p in clear_mal_sorted[:5]:
    pos   = scan_pos[p]
    score = scores[p]
    print(f"  {pos:>8}  {score:.4f}  True=MALWARE ✅")

print(f"\n  🟢 BENIGN RÕ RÀNG (score < 0.05) — {len(clear_ben)} mẫu")
print(f"  {'Index':>8}  {'Score':>7}  {'Nhãn':<12}")
print(f"  {'-'*8}  {'-'*7}  {'-'*12}")
for p in clear_ben[:5]:
    pos   = scan_pos[p]
    score = scores[p]
    print(f"  {pos:>8}  {score:.4f}  True=BENIGN  ✅")

print(f"\n  ⚠️  BORDERLINE có XGBoost SAI (score 0.4-0.6, XGB đoán sai)")
border_wrong = np.where(
    (scores > 0.40) & (scores < 0.60) &
    (((scores >= 0.5) & (y_scan == 0)) | ((scores < 0.5) & (y_scan == 1)))
)[0]
print(f"  {len(border_wrong)} mẫu — dùng để demo tại sao cần LLM")
print(f"  {'Index':>8}  {'Score':>7}  {'Nhãn':<16}  {'XGB'}")
print(f"  {'-'*8}  {'-'*7}  {'-'*16}  {'-'*8}")
for p in border_wrong[:5]:
    pos      = scan_pos[p]
    score    = scores[p]
    true_lbl = "MALWARE" if y_scan[p] == 1 else "BENIGN"
    xgb_pred = "MALWARE" if score >= 0.5 else "BENIGN"
    correct  = "✅" if xgb_pred == true_lbl else "❌ SAI"
    print(f"  {pos:>8}  {score:.4f}  True={true_lbl:<12}  XGB={xgb_pred} {correct}")

print(f"\n{sep}")
print("  3 INDEX ĐƯỢC CHỌN ĐỂ DEMO:")

# Chọn 1 malware rõ nhất
best_mal = scan_pos[clear_mal_sorted[0]] if len(clear_mal_sorted) > 0 else None
best_ben = scan_pos[clear_ben[0]]        if len(clear_ben) > 0 else None
best_brd = scan_pos[border_wrong[0]]     if len(border_wrong) > 0 else None

if best_mal:
    s = scores[clear_mal_sorted[0]]
    print(f"  Demo 1 — MALWARE rõ  : index {best_mal}  (score {s:.4f})")
if best_ben:
    s = scores[clear_ben[0]]
    print(f"  Demo 2 — BENIGN rõ   : index {best_ben}  (score {s:.4f})")
if best_brd:
    s = scores[border_wrong[0]]
    print(f"  Demo 3 — BORDERLINE  : index {best_brd}  (score {s:.4f})")

print(f"{sep}")

# Lưu
best = {
    "clear_malware": [{"index": int(scan_pos[p]), "score": round(float(scores[p]),4)}
                      for p in clear_mal_sorted[:5]] if len(clear_mal_sorted) > 0 else [],
    "clear_benign" : [{"index": int(scan_pos[p]), "score": round(float(scores[p]),4)}
                      for p in clear_ben[:5]],
    "borderline_wrong": [{"index": int(scan_pos[p]), "score": round(float(scores[p]),4),
                          "true_label": int(y_scan[p])}
                         for p in border_wrong[:5]],
    "recommended": {
        "demo1_malware"   : int(best_mal) if best_mal is not None else None,
        "demo2_benign"    : int(best_ben) if best_ben is not None else None,
        "demo3_borderline": int(best_brd) if best_brd is not None else None,
    }
}
with open(os.path.join(THIS_DIR, "demo_samples.json"), "w") as f:
    json.dump(best, f, indent=2)
print("\n  demo_samples.json đã lưu")