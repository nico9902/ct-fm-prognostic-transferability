import numpy as np
import pandas as pd

from run_colipri_token_alignment_audit import (COORDS, analyze_similarity,
                                                random_derangements)


def test_identity_alignment_metrics():
    similarity=np.eye(64,dtype=np.float32)
    result=analyze_similarity(similarity)
    assert result['same_token_top1_fraction']==1
    assert result['hungarian_fixed_fraction']==1
    assert result['top1_displacement_l1']==0
    assert result['translation_is_zero']==1
    assert abs(result['translation_eval_gain'])<1e-8


def test_known_translation_selected_on_heldout_tokens():
    similarity=np.full((64,64),-1.,dtype=np.float32);shift=np.array([1,0,0])
    target=COORDS+shift;valid=np.all((target>=0)&(target<4),axis=1)
    mapped=np.ravel_multi_index(target[valid].T,(4,4,4))
    similarity[np.flatnonzero(valid),mapped]=1
    np.fill_diagonal(similarity,.1)
    result=analyze_similarity(similarity)
    assert (result['translation_d0'],result['translation_d1'],result['translation_d2'])==(1,0,0)
    assert result['translation_eval_gain']>.5


def test_random_controls_are_within_group_derangements():
    pairs=pd.DataFrame({'split':['train']*4+['test']*4,'pair_type':['T0_T1']*2+['T1_T2']*2+['T0_T1']*2+['T1_T2']*2})
    for mapping in random_derangements(pairs,5,3):
        assert np.all(mapping!=np.arange(len(pairs)))
        assert np.all(pairs.split.to_numpy()[mapping]==pairs.split.to_numpy())
        assert np.all(pairs.pair_type.to_numpy()[mapping]==pairs.pair_type.to_numpy())
