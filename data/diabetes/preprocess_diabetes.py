import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

def categorize_icd9(code):
    try:
        s = str(code).strip()
        if s.startswith('250'):
            return 3  # Diabetes complications
        val = float(s)
        if (390 <= val <= 459) or val == 785:
            return 0  # Circulatory
        elif (460 <= val <= 519) or val == 786:
            return 1  # Respiratory
        elif (520 <= val <= 579) or val == 787:
            return 2  # Digestive
        else:
            return -1  # Other / excluded
    except:
        return -1

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    out_dir = os.path.join(base_dir, 'data', 'diabetes')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'diabetes_preprocessed.pkl')
    
    print("Fetching Diabetes130US dataset from OpenML (data_id=4541)...")
    data = fetch_openml(data_id=4541, as_frame=True, parser='auto')
    df = data.frame
    
    # Subsample 20,000 encounters for computational tractability
    # Use fixed seed 42 to guarantee deterministic sampling
    rng = np.random.RandomState(42)
    sample_idx = rng.permutation(len(df))[:20000]
    df_20k = df.iloc[sample_idx].copy().reset_index(drop=True)
    
    # Binary proxy task: Hospital Readmission
    # y = 1 if readmitted (<30 or >30), y = 0 if not readmitted (NO)
    full_y = (df_20k['readmitted'] != 'NO').astype(int).values
    full_sample_ids = df_20k['encounter_id'].values.tolist()
    
    # Age midpoint mapping
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    df_20k['age_numeric'] = df_20k['age'].map(age_map).astype(float)
    df_20k['gender_numeric'] = df_20k['gender'].map({'Male': 0.0, 'Female': 1.0}).fillna(0.0)
    
    # Medication level mappings: No -> 0, Down -> 1, Steady -> 2, Up -> 3
    med_map = {'No': 0.0, 'Down': 1.0, 'Steady': 2.0, 'Up': 3.0}
    med_cols = ['insulin', 'metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone']
    for med in med_cols:
        df_20k[f'{med}_numeric'] = df_20k[med].map(med_map).fillna(0.0)
        
    # One-hot encode race
    race_dummies = pd.get_dummies(df_20k['race'], prefix='race', dtype=float)
    expected_race = ['race_?', 'race_AfricanAmerican', 'race_Asian', 'race_Caucasian', 'race_Hispanic', 'race_Other']
    for col in expected_race:
        if col not in race_dummies.columns:
            race_dummies[col] = 0.0
    race_dummies = race_dummies[expected_race]
    
    # Continuous hospital features
    num_cols = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
        'num_medications', 'number_outpatient', 'number_emergency', 
        'number_inpatient', 'number_diagnoses'
    ]
    for col in num_cols:
        df_20k[col] = pd.to_numeric(df_20k[col], errors='coerce').fillna(0.0)
        
    feature_cols = ['age_numeric', 'gender_numeric'] + expected_race + num_cols + [f'{m}_numeric' for m in med_cols]
    
    full_X_df = pd.concat([df_20k[['age_numeric', 'gender_numeric']], race_dummies, df_20k[num_cols + [f'{m}_numeric' for m in med_cols]]], axis=1)
    full_X_raw = full_X_df[feature_cols].values.astype(float)
    
    # Target cohort: Readmitted patients (y == 1) with primary diagnosis in the 4 ICD-9 classes
    df_20k['subtypes'] = df_20k['diag_1'].apply(categorize_icd9)
    target_mask = (full_y == 1) & (df_20k['subtypes'] >= 0)
    
    target_df = df_20k[target_mask].copy()
    target_sample_ids = target_df['encounter_id'].values.tolist()
    subtypes = target_df['subtypes'].values.astype(int)
    
    # Standardize features for the target cohort
    target_X_raw = full_X_raw[target_mask]
    scaler = StandardScaler()
    X = scaler.fit_transform(target_X_raw)
    
    output_dict = {
        'X': X,
        'X_raw': target_X_raw,
        'y': np.ones(len(subtypes)),
        'subtypes': subtypes,
        'sample_ids': target_sample_ids,
        'feature_names': feature_cols,
        'full_X_raw': full_X_raw,
        'full_y': full_y,
        'full_sample_ids': full_sample_ids
    }
    
    with open(out_path, 'wb') as f:
        pickle.dump(output_dict, f)
        
    print(f"Successfully processed Diabetes dataset:")
    print(f"  Full cohort: N={len(full_sample_ids)}, Features={full_X_raw.shape[1]}")
    print(f"  Target cohort: N={len(target_sample_ids)}")
    print(f"  Subtypes distribution (0: Circ, 1: Resp, 2: Dig, 3: Diab): {np.bincount(subtypes)}")
    print(f"  Saved to: {out_path}")

if __name__ == '__main__':
    main()
