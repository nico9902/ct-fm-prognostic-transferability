"""Recover imaging features without asserting unverified clinical round linkage."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib
sys.path.insert(0,str(Path.cwd()/'scripts'))
import pandas as pd
import numpy as np
from run_colipri_decodability import extract,REVISION
root=Path('results/nlst_patient_recovery/215070')
original=pd.read_csv('data/nlst/processed/nlst/pilot_1000_balanced/sliding_npy_manifest.csv',dtype=str)
t0=original[original.patient_id.eq('215070') & original.timepoint.eq('T0')].copy()
t0['timepoint']='T1'
candidate=pd.read_csv(root/'axial_candidate_result.csv',dtype=str)
candidate['timepoint']='T2'
cols=['patient_id','study_uid','series_uid','study_date','timepoint','npy_path','dicom_dir_resolved']
prepared=pd.concat([t0[cols],candidate[cols]],ignore_index=True)
prepared['clinical_round_linkage']='verified_dicom_date_trial_tag_description'
prepared.to_csv(root/'recovered_imaging_manifest.csv',index=False)
checkpoint=Path('external/colipri/model.safetensors')
a=SimpleNamespace(checkpoint=checkpoint,prepared_manifest=root/'recovered_imaging_manifest.csv',output_dir=root/'colipri',device='cuda',no_dense_cache=False,smoke_only=False,workers=0)
a.output_dir.mkdir(parents=True,exist_ok=True)
extract(a,prepared)
shapes={}
for pool in ['mean','max','attention']:
 z=np.load(a.output_dir/pool/'cache/embeddings.npy')
 assert np.isfinite(z).all()
 shapes[pool]=list(z.shape)
shapes['dense_per_scan']=list(np.load(a.output_dir/'cache/dense/00000.npy',mmap_mode='r').shape)
meta={'patient_id':'215070','scans':2,'clinical_round_linkage_verified':True,'patient_reinstated':False,'encoder_frozen':True,'checkpoint_sha256':hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'revision':REVISION,'shapes':shapes}
(root/'feature_recovery_validation.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
