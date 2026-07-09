# 🛡️ SOC Hybrid Pipeline: Phân tích Mã độc Tĩnh (Static Malware Analysis)

Hệ thống SOC Pipeline Hybrid hai tầng tự động phát hiện và phân tích mã độc tĩnh. Giải quyết bài toán Alert Fatigue trong vận hành SOC bằng cách kết hợp tốc độ của Học máy truyền thống (**XGBoost**) và khả năng giải thích chuyên sâu của Mô hình ngôn ngữ lớn (**Llama 3.2**).

Đồ án tốt nghiệp — *"Hybrid SOC Pipeline: Combining Machine Learning and LLM for Malware Detection and Analysis"* — FPT University.

## 🚀 Kiến trúc Hệ thống

- **Tầng 1 (ML Triage):** Mô hình XGBoost huấn luyện trên bộ dữ liệu EMBER 2018 (2.351 đặc trưng tĩnh), phân loại toàn bộ file PE đầu vào với tốc độ ~96.000 mẫu/giây.
  - `Risk Score < 0.3`: Whitelist (An toàn).
  - `Risk Score > 0.7`: Blacklist (Mã độc - Chặn ngay).
  - `0.3 ≤ Risk Score ≤ 0.7`: Vùng xám (~22% số mẫu, chuyển tiếp lên Tầng 2).
- **Tầng 2 (Deep Analysis):** Llama 3.2 (chạy local qua Ollama) đọc kết quả SHAP TreeExplainer, giải thích ngữ nghĩa bằng ngôn ngữ tự nhiên và ánh xạ sang kỹ thuật **MITRE ATT&CK**.

| Metric | XGBoost | Llama 3.2 |
|---|---|---|
| Accuracy | 94.3% | 57.1% |
| Recall | 95.9% | 85.5% |
| Throughput | 96,000 mẫu/giây | 0.07 mẫu/giây |
| Explainability | Không (black box) | Có (ngôn ngữ tự nhiên + MITRE ATT&CK) |

Chi tiết đầy đủ: xem `docs/BaoCao.docx`.

## 📁 Cấu trúc thư mục

```
.
├── train_ml.py                  # Huấn luyện XGBoost
├── evaluate_ml.py               # Đánh giá trên 200k mẫu test
├── shap_analysis.py             # SHAP TreeExplainer
├── map_features.py              # Ánh xạ đặc trưng → MITRE ATT&CK
├── llm_eval.py                  # Đánh giá Llama 3.2 (few-shot)
├── comparison_report.py         # So sánh XGBoost vs Llama 3.2
├── fair_comparison.py           # So sánh apple-to-apple
├── improve_report.py            # Confusion matrices
├── threshold_experiment.py      # Phân tích độ nhạy ngưỡng
├── find_demo_samples.py         # Tìm mẫu demo
├── demo_pipeline.py             # Demo CLI đầu-cuối
├── check_pipeline.py            # Kiểm tra tính toàn vẹn pipeline
├── prepare_data.py              # Tiền xử lý dữ liệu EMBER
├── test.py                      # Unit test
├── test_real_malware.py         # Test trên mẫu mã độc thực tế
├── app.py                       # Giao diện Streamlit (upload PE file, phân tích real-time)
├── results/                     # eval_results.json, shap_results.json, llm_results.json, comparison_table.json, fair_comparison.json, threshold_results.json, training_results.json, demo_samples.json, llm_checkpoint.json, xgboost_ember_layer1.json
├── figures/                     # cm_xgboost.png, cm_llm.png, cm_comparison.png, comparison_bar.png, shap_bar.png, shap_bee.png, threshold_analysis.png
├── docs/                        # BaoCao.docx, slide thuyết trình
├── requirements.txt
└── .gitignore
```

> Lưu ý: thư mục `ember/`, `build/`, `malconv/`, `resources/`, `licenses/`, `scripts/`, `xgb_cache/` tồn tại trên máy chạy dự án nhưng **không** thuộc repo này — đó là source code + cache của thư viện `ember` (cài qua `pip install ember`, xem mục Cài đặt bên dưới), đã được loại trừ qua `.gitignore`.

## 🛠️ Yêu cầu Hệ thống

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8 cores |
| Storage | 60 GB | 100 GB |
| OS | Windows 10/11 | Windows 10/11 |
| Python | 3.10+ | - |

## ⚙️ Cài đặt

**1. Clone repo:**
```bash
git clone https://github.com/hta2312-nn/SOC-Hybrid-Pipeline.git
cd SOC-Hybrid-Pipeline
```

**2. Tạo và kích hoạt môi trường ảo:**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Cài đặt thư viện:**
```bash
pip install -r requirements.txt
```

**4. Tải mô hình Llama 3.2 qua Ollama (~2GB, một lần):**
```bash
ollama pull llama3.2
```

**5. Chuẩn bị dữ liệu EMBER 2018:**
Tải bộ dữ liệu EMBER 2018 và đặt vào thư mục `ember_data/ember2018/` (xem hướng dẫn tại [EMBER repo](https://github.com/elastic/ember)).

## ▶️ Chạy Pipeline

Chạy tuần tự theo thứ tự sau (tổng thời gian ~10–14 giờ, có checkpoint để chạy lại từ giữa):

| Bước | Script | Mục đích | Thời gian |
|---|---|---|---|
| 1 | `test_py.py` | Xác minh dataset | < 1 phút |
| 2 | `train_ml.py` | Train XGBoost | 8–10 phút |
| 3 | `evaluate_ml.py` | Đánh giá 200k mẫu | 5–8 phút |
| 4 | `shap_analysis.py` | SHAP analysis | 5–10 phút |
| 5 | `map_features.py` | Ánh xạ MITRE | < 1 phút |
| 6 | `llm_eval.py` | Đánh giá LLM | 10–14 giờ |
| 7 | `comparison_report.py` | Biểu đồ so sánh | < 1 phút |
| 8 | `fair_comparison.py` | So sánh apple-to-apple | < 1 phút |
| 9 | `improve_report.py` | Confusion matrices | 1–2 phút |
| 10 | `threshold_experiment.py` | Phân tích ngưỡng | 1–2 phút |
| 11 | `find_demo_samples.py` | Tìm mẫu demo | 3–5 phút |
| 12 | `demo_pipeline.py` | Demo CLI | ~15s/mẫu |

**Chạy giao diện web (Streamlit):**
```bash
streamlit run app.py
```

## 🔍 Phát hiện Khoa học Chính

- **Clever Hans Effect:** Dataset API call sequence (Fellicious et al.) cho thấy dấu hiệu mô hình học sai lệch log của Cuckoo Sandbox thay vì hành vi API thực — dẫn đến quyết định chuyển sang đặc trưng tĩnh EMBER 2018.
- **Concept Drift:** Mô hình huấn luyện trên EMBER 2018 giảm độ chính xác khi kiểm thử trên mẫu mã độc mới (MalwareBazaar 2026), cho thấy nhu cầu validation định kỳ trong triển khai thực tế.
- **Kiến trúc Hybrid tối ưu:** Ngưỡng (0.3, 0.7) giảm ~22% khối lượng xử lý xuống Tầng 2 LLM trong khi vẫn giữ FPR và F1 tổng thể ở mức tốt.

## 📄 License

MIT License (hoặc điều chỉnh theo yêu cầu của trường/khoa).

## 👥 Tác giả

Tạ Minh Hoàng - SE194567
Lê Đức Hải - SE194586

Đồ án tốt nghiệp — Chuyên ngành An toàn Thông tin, FPT University, 2026.