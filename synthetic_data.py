import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

np.random.seed(42)

cancer_types = ['Breast', 'Lung', 'Colon', 'Prostate', 'Melanoma']
stages = ['I', 'II', 'III', 'IV']
biomarkers = ['HER2+', 'HER2-', 'ER+', 'ER-', 'PD-L1+', 'PD-L1-', 'EGFR+', 'EGFR-']
ecog_scores = [0, 1, 2, 3, 4]

def generate_clinical_note(age, cancer_type, stage, biomarker, ecog,
                           prior_platinum=False, cardiac_disease=False,
                           organ_transplant=False):
    notes = [
        f"Patient is a {age}-year-old with {stage} {cancer_type} cancer.",
        f"Biomarker profile shows {biomarker} expression.",
        f"ECOG performance status: {ecog}.",
        f"{'Metastatic disease present' if stage == 'IV' else 'No evidence of distant metastasis'}.",
    ]

    if prior_platinum:
        notes.append("Patient has received prior platinum-based chemotherapy.")
    else:
        notes.append("No prior platinum-based chemotherapy on record.")

    if cardiac_disease:
        notes.append("Active cardiac disease noted; currently managed with medication.")
    else:
        notes.append("No active cardiac conditions reported.")

    if organ_transplant:
        notes.append("History of solid organ transplant; on chronic immunosuppressive therapy.")

    return " ".join(notes)

def generate_eligibility_label(row):
    eligible = True
    reasons = []

    if row['age'] < 18 or row['age'] > 75:
        eligible = False
        reasons.append("Age outside range")

    if row['stage'] == 'IV':
        eligible = False
        reasons.append("Stage IV excluded")

    if row['ecog_score'] > 2:
        eligible = False
        reasons.append("ECOG score too high")

    if row['hemoglobin'] < 9.0:
        eligible = False
        reasons.append("Hemoglobin too low")

    if row['creatinine'] > 2.0:
        eligible = False
        reasons.append("Creatinine elevated")

    if row['neutrophil_count'] < 1.5:
        eligible = False
        reasons.append("Neutrophil count too low")

    # Text-only exclusion criteria — not derivable from structured fields
    if row.get('prior_platinum_therapy'):
        eligible = False
        reasons.append("Prior platinum-based therapy")

    if row.get('active_cardiac_disease'):
        eligible = False
        reasons.append("Active cardiac disease")

    if row.get('organ_transplant'):
        eligible = False
        reasons.append("Organ transplant history")

    return 1 if eligible else 0, reasons

def generate_dataset(n_samples=500):
    data = []

    for i in range(n_samples):
        age = np.random.randint(25, 85)
        cancer_type = np.random.choice(cancer_types)
        stage = np.random.choice(stages, p=[0.25, 0.30, 0.25, 0.20])
        biomarker = np.random.choice(biomarkers)
        ecog = np.random.choice(ecog_scores, p=[0.30, 0.35, 0.20, 0.10, 0.05])

        hemoglobin = np.random.normal(12.5, 2.0)
        creatinine = np.random.gamma(2, 0.4)
        neutrophil_count = np.random.normal(4.0, 1.5)
        platelet_count = np.random.normal(250, 50)

        prior_platinum = np.random.random() < 0.15
        cardiac_disease = np.random.random() < 0.10
        organ_transplant = np.random.random() < 0.05

        clinical_note = generate_clinical_note(
            age, cancer_type, stage, biomarker, ecog,
            prior_platinum=prior_platinum,
            cardiac_disease=cardiac_disease,
            organ_transplant=organ_transplant
        )

        row = {
            'patient_id': f'PT{i:04d}',
            'age': age,
            'cancer_type': cancer_type,
            'stage': stage,
            'biomarker': biomarker,
            'ecog_score': ecog,
            'hemoglobin': round(hemoglobin, 1),
            'creatinine': round(creatinine, 2),
            'neutrophil_count': round(neutrophil_count, 2),
            'platelet_count': round(platelet_count, 1),
            'prior_platinum_therapy': prior_platinum,
            'active_cardiac_disease': cardiac_disease,
            'organ_transplant': organ_transplant,
            'clinical_notes': clinical_note,
        }

        eligible, reasons = generate_eligibility_label(row)
        row['eligible'] = eligible
        row['exclusion_reasons'] = '; '.join(reasons) if reasons else 'None'

        data.append(row)
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating synthetic clinical trial dataset...")
    df = generate_dataset(3000)
    
    # Save to CSV
    df.to_csv('clinical_trial_data.csv', index=False)
    
    print(f" Dataset generated: {len(df)} samples")
    print(f"   Eligible: {df['eligible'].sum()}")
    print(f"   Ineligible: {len(df) - df['eligible'].sum()}")
    print(f"   Eligibility rate: {df['eligible'].mean():.2%}")
    print("\nSample records:")
    print(df.head(3))
    print("\nDataset saved to 'clinical_trial_data.csv'")