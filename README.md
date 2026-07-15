# SOC Hybrid Pipeline: Static Malware Analysis

A two-tier hybrid SOC pipeline for automated malware detection and analysis. Addresses the Alert Fatigue problem in SOC operations by combining the speed of traditional Machine Learning (**XGBoost**) with the deep explainability of a Large Language Model (**Llama 3.2**).

Graduation thesis project — *"Hybrid SOC Pipeline: Combining Machine Learning and LLM for Malware Detection and Analysis"* — FPT University.

## Architecture

- **Tier 1 (ML Triage):** An XGBoost model trained on the EMBER 2018 dataset (2,351 static features), classifying all incoming PE files at a throughput of ~96,000 samples/second.
  - `Risk Score < 0.3`: Whitelist (benign).
  - `Risk Score > 0.7`: Blacklist (malware — block immediately).
  - `0.3 ≤ Risk Score ≤ 0.7`: Grey zone (~22% of samples, escalated to Tier 2).
- **Tier 2 (Deep Analysis):** Llama 3.2 (run locally via Ollama) reads the SHAP TreeExplainer output, produces a natural-language explanation, and maps the finding to **MITRE ATT&CK** techniques.

| Metric | XGBoost | Llama 3.2 |
|---|---|---|
| Accuracy | 94.3% | 57.1% |
| Recall | 95.9% | 85.5% |
| Throughput | 96,000 samples/sec | 0.07 samples/sec |
| Explainability | No (black box) | Yes (natural language + MITRE ATT&CK) |

Full details: see `docs/BaoCao.docx`.

## Project Structure

```
.
├── train_ml.py                  # Train XGBoost
├── evaluate_ml.py               # Evaluate on the 200k-sample test set
├── shap_analysis.py             # SHAP TreeExplainer
├── map_features.py              # Map features → MITRE ATT&CK
├── llm_eval.py                  # Evaluate Llama 3.2 (few-shot)
├── comparison_report.py         # XGBoost vs Llama 3.2 comparison
├── fair_comparison.py           # Apples-to-apples comparison
├── improve_report.py            # Confusion matrices
├── threshold_experiment.py      # Threshold sensitivity analysis
├── find_demo_samples.py         # Find demo samples
├── demo_pipeline.py             # End-to-end CLI demo
├── check_pipeline.py            # Pipeline integrity check
├── prepare_data.py              # EMBER data preprocessing
├── test.py                      # Unit tests
├── test_real_malware.py         # Test on real-world malware samples
├── app.py                       # Streamlit web UI (upload a PE file, real-time analysis)
├── results/                     # eval_results.json, shap_results.json, llm_results.json, comparison_table.json, fair_comparison.json, threshold_results.json, training_results.json, demo_samples.json, llm_checkpoint.json, xgboost_ember_layer1.json
├── figures/                     # cm_xgboost.png, cm_llm.png, cm_comparison.png, comparison_bar.png, shap_bar.png, shap_bee.png, threshold_analysis.png
├── docs/                        # BaoCao.docx, presentation slides
├── requirements.txt
└── .gitignore
```

> Note: the `ember/`, `build/`, `malconv/`, `resources/`, `licenses/`, `scripts/`, and `xgb_cache/` folders exist on the machine running the project but are **not** part of this repo — they are the source code and cache of the `ember` library (installed via `pip install ember`, see Installation below) and are excluded via `.gitignore`.

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8 cores |
| Storage | 60 GB | 100 GB |
| OS | Windows 10/11 | Windows 10/11 |
| Python | 3.10+ | - |

## Installation

**1. Clone the repo:**
```bash
git clone https://github.com/hta2312-nn/SOC-Hybrid-Pipeline.git
cd SOC-Hybrid-Pipeline
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Pull the Llama 3.2 model via Ollama (~2GB, one-time):**
```bash
ollama pull llama3.2
```

**5. Prepare the EMBER 2018 dataset:**
Download the EMBER 2018 dataset and place it under `ember_data/ember2018/` (see instructions in the [EMBER repo](https://github.com/elastic/ember)).

## Running the Pipeline

Run the following steps in order (~10–14 hours total, with checkpointing so you can resume mid-run):

| Step | Script | Purpose | Time |
|---|---|---|---|
| 0 | `prepare_data.py` | Preprocess EMBER data (run before step 1 if the cache doesn't exist yet) | depends on storage |
| 1 | `test.py` | Verify the dataset | < 1 min |
| 2 | `train_ml.py` | Train XGBoost | 8–10 min |
| 3 | `evaluate_ml.py` | Evaluate on 200k samples | 5–8 min |
| 4 | `shap_analysis.py` | SHAP analysis | 5–10 min |
| 5 | `map_features.py` | Map to MITRE ATT&CK | < 1 min |
| 6 | `llm_eval.py` | Evaluate the LLM | 10–14 hours |
| 7 | `comparison_report.py` | Comparison charts | < 1 min |
| 8 | `fair_comparison.py` | Apples-to-apples comparison | < 1 min |
| 9 | `improve_report.py` | Confusion matrices | 1–2 min |
| 10 | `threshold_experiment.py` | Threshold analysis | 1–2 min |
| 11 | `find_demo_samples.py` | Find demo samples | 3–5 min |
| 12 | `demo_pipeline.py` | CLI demo | ~15s/sample |

Utility scripts (run as needed, not part of the main flow): `check_pipeline.py` (checks integrity between steps), `test_real_malware.py` (tests on real-world malware samples outside the EMBER set).

**Run the web interface (Streamlit):**
```bash
streamlit run app.py
```

## Key Scientific Findings

- **Clever Hans Effect:** The API call sequence dataset (Fellicious et al.) showed signs that the model was learning spurious correlations from Cuckoo Sandbox logging artifacts rather than real API behavior — leading to the decision to switch to the static EMBER 2018 features instead.
- **Concept Drift:** The model trained on EMBER 2018 shows reduced accuracy when tested on newer malware samples (MalwareBazaar 2026), indicating a need for periodic revalidation in real-world deployment.
- **Optimal Hybrid Architecture:** The (0.3, 0.7) threshold reduces the workload escalated to the Tier 2 LLM by ~22%, while keeping overall FPR and F1 at a good level.

## License

MIT License (or adjust as required by your school/department).

## Authors

Ta Minh Hoang -  
Le Duc Hai -
Huynh Ngoc Tuan - 
Do Tran Vy -
Nguyen Duc Hoang Anh


