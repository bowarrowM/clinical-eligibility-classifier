import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

np.random.seed(42)

cancer_types = ['Breast', 'Lung', 'Colon', 'Prostate', 'Melanoma']
stages = ['I', 'II', 'III', 'IV']
biomarkers = ['HER2+', 'HER2-', 'ER+', 'ER-', 'PD-L1+', 'PD-L1-', 'EGFR+', 'EGFR-']
ecog_scores = [0, 1, 2, 3, 4]

def generate_clinical_note(age, cancer_type, stage, biomarker, ecog):
    notes = [
        f"Patient is a {age}-year-old with {stage} {cancer_type} cancer.",
        f"Biomarker profile shows {biomarker} expression.",
        f"ECOG performance status: {ecog}.",
        f"Patient {'has' if np.random.random() > 0.7 else 'has not'} received prior chemotherapy.",
        f"{'Metastatic disease present' if stage == 'IV' else 'No evidence of distant metastasis'}.",
    ]
    return " ".join(notes)

def generate_eligibility_label(row):
    """
    Logic eg:
    row : description
    Age: 18-75...
    row : description ...
    """
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
        
        clinical_note = generate_clinical_note(age, cancer_type, stage, biomarker, ecog)
        
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
            'clinical_notes': clinical_note
        }
        
        eligible, reasons = generate_eligibility_label(row)
        row['eligible'] = eligible
        row['exclusion_reasons'] = '; '.join(reasons) if reasons else 'None'
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating synthetic clinical trial dataset...")
    df = generate_dataset(500)
    
    # Save to CSV
    df.to_csv('clinical_trial_data.csv', index=False)
    
    print(f" Dataset generated: {len(df)} samples")
    print(f"   Eligible: {df['eligible'].sum()}")
    print(f"   Ineligible: {len(df) - df['eligible'].sum()}")
    print(f"   Eligibility rate: {df['eligible'].mean():.2%}")
    print("\nSample records:")
    print(df.head(3))
    print("\nDataset saved to 'clinical_trial_data.csv'")