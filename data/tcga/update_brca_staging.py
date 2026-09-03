import os
import json
import pickle
import requests
import numpy as np
from collections import Counter

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tcga'))
    pkl_path = os.path.join(data_dir, 'brca_preprocessed.pkl')
    
    print("Loading BRCA preprocessed data...")
    with open(pkl_path, 'rb') as f:
        ds = pickle.load(f)
        
    print("Fetching staging data from GDC API...")
    url = 'https://api.gdc.cancer.gov/cases'
    filt = json.dumps({
        "op": "=",
        "content": {
            "field": "project.project_id",
            "value": "TCGA-BRCA"
        }
    })
    params = {
        'filters': filt,
        'fields': 'submitter_id,diagnoses.ajcc_pathologic_stage',
        'size': '1200',
        'format': 'json'
    }
    r = requests.get(url, params=params)
    data = r.json()
    
    stage_map = {}
    for hit in data['data']['hits']:
        sid = hit.get('submitter_id', '')
        diag = hit.get('diagnoses', [])
        if diag:
            stage = diag[0].get('ajcc_pathologic_stage', None)
            if stage:
                stage_map[sid] = stage
                
    # Map to broad stages for the target cohort (where y == 1)
    # The existing target cohort in ds is already subsetted to y == 1 (and non-NaN subtypes)
    # So we just match on ds['sample_ids']
    
    matched_stages = []
    for sid in ds['sample_ids']:
        patient_id = '-'.join(sid.split('-')[:3])
        s = stage_map.get(patient_id, None)
        
        if s is None:
            matched_stages.append(None)
        elif 'Stage I' in s and 'II' not in s and 'IV' not in s:
            matched_stages.append('Stage_I')
        elif 'Stage II' in s:
            matched_stages.append('Stage_II')
        elif 'Stage III' in s:
            matched_stages.append('Stage_III')
        elif 'Stage IV' in s:
            matched_stages.append('Stage_IV')
        else:
            matched_stages.append(None)
            
    matched_stages = np.array(matched_stages)
    print("Broad stage distribution in target cohort:")
    counts = Counter(matched_stages)
    for k, v in sorted(counts.items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v}")
        
    # We only want to keep samples that are Stage I, II, or III
    valid_mask = np.isin(matched_stages, ['Stage_I', 'Stage_II', 'Stage_III'])
    
    ds['staging_subtypes'] = matched_stages[valid_mask]
    ds['staging_X'] = ds['X'][valid_mask]
    ds['staging_sample_ids'] = [ds['sample_ids'][i] for i in range(len(ds['sample_ids'])) if valid_mask[i]]
    
    print(f"Final target cohort for staging task: {len(ds['staging_sample_ids'])} samples")
    
    out_path = os.path.join(data_dir, 'brca_preprocessed_with_staging.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(ds, f)
        
    print(f"Saved updated dataset to {out_path}")

if __name__ == '__main__':
    main()
