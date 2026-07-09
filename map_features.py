"""
Map f637, f930... → tên EMBER feature thật
Chạy: py map_features.py
"""

import json, os
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ── EMBER 2018 feature layout (2351 features, lief 0.9 compatible) ────────
# Nguồn: github.com/elastic/ember/blob/master/ember/features.py
# Layout chính xác theo thứ tự concat trong PEFeatureExtractor

EMBER_FEATURE_GROUPS = [
    # (tên_nhóm, số_feature, mô_tả)
    ("ByteHistogram",        256, "Tần suất byte 0x00-0xFF trong file PE"),
    ("ByteEntropyHistogram", 256, "Entropy theo từng đoạn byte"),
    ("StringExtractor",      104, "Thống kê chuỗi: URLs, registry, MZ headers..."),
    ("GeneralFileInfo",        10, "Kích thước file, virtual size, imports, exports..."),
    ("HeaderFileInfo",         62, "PE header: timestamp, machine type, subsystem..."),
    ("SectionInfo",           255, "Thông tin sections: .text, .data, .rsrc..."),
    ("ImportsInfo",           1280, "Import functions từ DLLs (kernel32, ntdll...)"),
    ("ExportsInfo",            128, "Export functions"),
]
# Tổng: 256+256+104+10+62+255+1280+128 = 2351 ✓

def get_feature_name(idx: int) -> tuple[str, str]:
    """Trả về (tên_chi_tiết, tên_nhóm) cho feature index."""
    cursor = 0
    for group_name, size, desc in EMBER_FEATURE_GROUPS:
        if cursor <= idx < cursor + size:
            local_idx = idx - cursor
            # Tên chi tiết hơn cho một số nhóm quan trọng
            if group_name == "ByteHistogram":
                return f"byte_freq_0x{local_idx:02X}", group_name
            elif group_name == "ByteEntropyHistogram":
                return f"byte_entropy_bin{local_idx}", group_name
            elif group_name == "GeneralFileInfo":
                general_names = [
                    "file_size", "vsize", "has_debug", "exports",
                    "imports", "has_relocations", "has_resources",
                    "has_signature", "has_tls", "symbols"
                ]
                name = general_names[local_idx] if local_idx < len(general_names) \
                       else f"general_{local_idx}"
                return name, group_name
            elif group_name == "HeaderFileInfo":
                return f"header_{local_idx}", group_name
            elif group_name == "SectionInfo":
                return f"section_{local_idx}", group_name
            elif group_name == "ImportsInfo":
                return f"import_{local_idx}", group_name
            elif group_name == "ExportsInfo":
                return f"export_{local_idx}", group_name
            elif group_name == "StringExtractor":
                string_names = [
                    "numstrings", "avlength", "printabledist_0","printabledist_1",
                    "printabledist_2","printabledist_3","printabledist_4",
                    "printabledist_5","printabledist_6","printabledist_7",
                    "printabledist_8","printabledist_9","printabledist_10",
                    "printabledist_11","printabledist_12","printabledist_13",
                    "printabledist_14","printabledist_15","printabledist_16",
                    "printabledist_17","printabledist_18","printabledist_19",
                    "printabledist_20","printabledist_21","printabledist_22",
                    "printabledist_23","printabledist_24","printabledist_25",
                    "printabledist_26","printabledist_27","printabledist_28",
                    "printabledist_29","printabledist_30","printabledist_31",
                    "printabledist_32","printabledist_33","printabledist_34",
                    "printabledist_35","printabledist_36","printabledist_37",
                    "printabledist_38","printabledist_39","printabledist_40",
                    "printabledist_41","printabledist_42","printabledist_43",
                    "printabledist_44","printabledist_45","printabledist_46",
                    "printabledist_47","printabledist_48","printabledist_49",
                    "printabledist_50","printabledist_51","printabledist_52",
                    "printabledist_53","printabledist_54","printabledist_55",
                    "printabledist_56","printabledist_57","printabledist_58",
                    "printabledist_59","printabledist_60","printabledist_61",
                    "printabledist_62","printabledist_63","printabledist_64",
                    "printabledist_65","printabledist_66","printabledist_67",
                    "printabledist_68","printabledist_69","printabledist_70",
                    "printabledist_71","printabledist_72","printabledist_73",
                    "printabledist_74","printabledist_75","printabledist_76",
                    "printabledist_77","printabledist_78","printabledist_79",
                    "printabledist_80","printabledist_81","printabledist_82",
                    "printabledist_83","printabledist_84","printabledist_85",
                    "printabledist_86","printabledist_87","printabledist_88",
                    "printabledist_89","printabledist_90","printabledist_91",
                    "printabledist_92","printabledist_93","printabledist_94",
                    "printabledist_95","printabledist_96","numurls",
                    "numemails","numregistrykeys","numpaths","numother",
                    "has_debug_str","has_exports_str","has_mz_header",
                ]
                name = string_names[local_idx] if local_idx < len(string_names) \
                       else f"string_{local_idx}"
                return name, group_name
            else:
                return f"{group_name}_{local_idx}", group_name
        cursor += size
    return f"f{idx}_unknown", "Unknown"

# ── Load SHAP results và map ──────────────────────────────────────────────
shap_path = os.path.join(THIS_DIR, "shap_results.json")
with open(shap_path) as f:
    shap_data = json.load(f)

print("=" * 68)
print("  TABLE 4.2 (UPDATED) — TOP 10 FEATURES với tên EMBER thật")
print("=" * 68)
print(f"\n  {'Rank':<5} {'Feature Name':<28} {'Nhóm':<22} {'SHAP':>9}")
print(f"  {'-'*5} {'-'*28} {'-'*22} {'-'*9}")

updated_top10 = []
for item in shap_data["top_10"]:
    raw_name = item["name"]          # vd: "f637"
    idx      = int(raw_name[1:]) if raw_name.startswith("f") and \
                raw_name[1:].isdigit() else -1

    if idx >= 0:
        real_name, group = get_feature_name(idx)
    else:
        real_name, group = raw_name, "Unknown"

    val = item["shap_mean_abs"]
    print(f"  {item['rank']:<5} {real_name:<28} {group:<22} {val:>9.5f}")

    updated_top10.append({
        "rank"          : item["rank"],
        "feature_index" : idx,
        "feature_name"  : real_name,
        "feature_group" : group,
        "shap_mean_abs" : val,
    })

print("=" * 68)

# ── Phân tích theo nhóm ───────────────────────────────────────────────────
from collections import Counter
group_counts = Counter(x["feature_group"] for x in updated_top10)

print(f"\n  PHÂN BỐ TOP 10 THEO NHÓM FEATURE:")
print(f"  {'Nhóm':<25} {'Số features trong Top 10':>25}")
print(f"  {'-'*25} {'-'*25}")
for group, cnt in group_counts.most_common():
    print(f"  {group:<25} {cnt:>25}")

print(f"\n  Ý NGHĨA CHO LUẬN VĂN:")
for group, cnt in group_counts.most_common(3):
    if "Import" in group:
        print(f"  • {group} ({cnt} features): Model học từ DLL imports —")
        print(f"    malware thường import các hàm nhạy cảm (VirtualAlloc,")
        print(f"    WriteProcessMemory, CreateRemoteThread...)")
    elif "ByteHistogram" in group or "ByteEntropy" in group:
        print(f"  • {group} ({cnt} features): Model học từ phân phối byte —")
        print(f"    malware thường có entropy cao (packed/encrypted)")
    elif "Section" in group:
        print(f"  • {group} ({cnt} features): Model học từ PE sections —")
        print(f"    malware thường có sections bất thường (.text có W+X...)")
    elif "String" in group:
        print(f"  • {group} ({cnt} features): Model học từ strings —")
        print(f"    malware chứa URLs, registry keys, suspicious paths")
    elif "Header" in group:
        print(f"  • {group} ({cnt} features): Model học từ PE header —")
        print(f"    malware thường có timestamp giả hoặc subsystem bất thường")

# ── Lưu kết quả đã map ───────────────────────────────────────────────────
shap_data["top_10_mapped"] = updated_top10
shap_data["group_distribution"] = dict(group_counts)

with open(shap_path, "w", encoding="utf-8") as f:
    json.dump(shap_data, f, indent=2, ensure_ascii=False)

print(f"\n  shap_results.json đã cập nhật với tên feature thật")