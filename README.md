# Clinical Trial Eligibility Classifier

An end-to-end ML/NLP pipeline for predicting oncology clinical trial eligibility using a fine-tuned DistilBERT model and a rule-based reasoning engine. Served via FastAPI and deployed on Render.

> **Note:** This project will be extended into a fullstack application.

---

## Overview

Automates patient screening for clinical trials by combining structured data (demographics, labs) with unstructured clinical notes. Uses a two-tier decision pipeline: a rule engine handles hard exclusion criteria, and a fine-tuned DistilBERT model evaluates text-based criteria.

**Stack:** Python · DistilBERT (HuggingFace Transformers) · FastAPI · scikit-learn · Render · Hugging Face Hub

---

## Architecture

```
Rule Engine (hard criteria: age, stage, ECOG, labs)
    │
    ├─ FAIL → ineligible  (decision_source: "rule_engine")
    │
    └─ PASS → DistilBERT model (text-based criteria in clinical notes)
                  │
                  └─ eligible / ineligible  (decision_source: "model")
```

The rule engine cannot override the model for text-based criteria, and the model cannot override the rule engine for hard exclusions. Every response includes a `decision_source` field.

---

## Dataset

500 synthetic oncology patient records with:

- Demographics: age, cancer type, stage
- Lab values: hemoglobin, creatinine, neutrophils, platelets
- Performance status: ECOG score
- Biomarkers: HER2, ER, PD-L1, EGFR
- Clinical notes: unstructured text

**Eligibility criteria** (based on real oncology trials):

| Criterion | Threshold |
|-----------|-----------|
| Age | 18–75 years |
| Stage | I–III (Stage IV excluded) |
| ECOG | 0–2 |
| Hemoglobin | ≥ 9.0 g/dL |
| Creatinine | ≤ 2.0 mg/dL |
| Neutrophils | ≥ 1.5 × 10⁹/L |

Three additional exclusion criteria (`prior_platinum_therapy`, `active_cardiac_disease`, `organ_transplant`) exist **only** in clinical notes — invisible to the rule engine. This is the model's genuine learning task.

---

## Pipeline

Run scripts in order:

```bash
python synthetic_data.py       # Generate clinical_trial_data.csv (500 patients)
python data_preprocess.py      # Produce train/val/test_data.csv + label_encoders.json
python model.py                # Fine-tune DistilBERT → ./clinical_trial_model/
python model_evaluate.py       # Evaluate on test set
python llm_reasonings.py       # Demo rule-based reasoning on test_data.csv
python app.py                  # Start FastAPI server on :8000
python api_testing.py          # Hit the running API with test payloads
```

---

## Setup

```bash
git clone <repo-url>
cd clinical-eligibility-classifier

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

```
HF_REPO=hellomelo/clinical_eligibility_classifier
```

---

## API

Start the server:

```bash
python app.py
# or (production)
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Interactive docs at `http://localhost:8000/docs`.

### Endpoints

**`GET /health`**

**`POST /predict`**
```json
{
  "patient_id": "PT001",
  "age": 55,
  "cancer_type": "Breast",
  "stage": "II",
  "biomarker": "HER2+",
  "ecog_score": 1,
  "hemoglobin": 12.5,
  "creatinine": 1.1,
  "neutrophil_count": 3.8,
  "platelet_count": 245.0,
  "clinical_notes": "Patient with stage II breast cancer..."
}
```

**`POST /predict/batch`**
```json
{ "patients": [...] }
```

---

## Model Performance

Approximate metrics on the test set:

| Metric | Score |
|--------|-------|
| Accuracy | ~85–90% |
| F1 Score | ~0.85–0.90 |
| AUC-ROC | ~0.90–0.95 |

*Varies based on synthetic data generation seed.*

---

## Project Structure

```
├── synthetic_data.py        # Synthetic patient data generation
├── data_preprocess.py       # Feature engineering & train/val/test splits
├── model.py                 # DistilBERT fine-tuning
├── model_evaluate.py        # Evaluation & inference
├── llm_reasonings.py        # Rule-based reasoning engine
├── app.py                   # FastAPI server
├── hf_upload.py             # Upload model to Hugging Face Hub
├── api_testing.py           # API test payloads
├── requirements.txt
├── render.yaml              # Render deployment config
└── clinical_trial_model/    # Trained model (also on HF Hub)
```

---

## Deployment

Deployed on **Render** (see `render.yaml`). At startup, `app.py` downloads the model from Hugging Face Hub via `snapshot_download`.

Set `HF_REPO` as an environment variable in the Render dashboard.

---

## Potential Extensions

- Replace the rule-based reasoning engine with a real LLM (GPT-4 / Claude)
- Multi-trial matching: rank patients across multiple open trials
- PDF parsing for actual trial protocol documents
- Active learning: flag low-confidence cases for human review
- SHAP/LIME explainability on top of the model predictions

---

> **Disclaimer:** This is a demonstration project using synthetic data. Real clinical trial matching requires IRB approval, HIPAA compliance, and EHR integration.
