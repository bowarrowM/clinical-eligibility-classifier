import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

def preprocess_data(df):
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['cancer_type', 'stage', 'biomarker']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # combined text field for transformer
    df['combined_text'] = df.apply(lambda row: 
        f"Age: {row['age']} years. Cancer: {row['cancer_type']} Stage {row['stage']}. "
        f"Biomarker: {row['biomarker']}. ECOG: {row['ecog_score']}. "
        f"Labs: Hgb {row['hemoglobin']}, Cr {row['creatinine']}, "
        f"Neut {row['neutrophil_count']}, Plt {row['platelet_count']}. "
        f"Notes: {row['clinical_notes']}", axis=1)
    
    encoders_dict = {col: le.classes_.tolist() for col, le in label_encoders.items()}
    with open('label_encoders.json', 'w') as f:
        json.dump(encoders_dict, f)
    
    return df, label_encoders

def create_train_test_split(df, test_size=0.2, val_size=0.1):
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['eligible']
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_size_adjusted, random_state=42, 
        stratify=train_val['eligible']
    )
    
    print(f"Train set: {len(train)} samples ({train['eligible'].mean():.2%} eligible)")
    print(f"Val set: {len(val)} samples ({val['eligible'].mean():.2%} eligible)")
    print(f"Test set: {len(test)} samples ({test['eligible'].mean():.2%} eligible)")
    
    return train, val, test

if __name__ == "__main__":
    print("Preprocessing data...")
    df = pd.read_csv('clinical_trial_data.csv')
    
    df, label_encoders = preprocess_data(df)
    
    train_df, val_df, test_df = create_train_test_split(df)
    # save splits
    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('val_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    
    print("\n Data preprocessing complete")
    print("Files saved: train_data.csv, val_data.csv, test_data.csv")
    
    # sample showcase
    print("\nSample combined text:")
    print(train_df['combined_text'].iloc[0][:200] + "...")