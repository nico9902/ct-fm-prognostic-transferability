#!/usr/bin/env python3
"""CPU checks for subset isolation, metric reference and frozen-file integrity."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import run_lung1_e2_encoder_bf16_lightning as training
import run_lung1_e5_lora as external
from run_lung1_e3_frozen_linear import make_subsets


class ProtocolTests(unittest.TestCase):
    def test_e3_subsets_match_frozen_and_exclude_test(self):
        data = training.cohort(training.COHORT)
        folds = training.locked_folds(data, training.FOLDS)
        subsets, table = make_subsets(data, folds)
        for directory in ('lung1_e3_frozen_linear', 'lung1_e3_frozen_linear_pca16'):
            saved = pd.read_csv(training.ROOT / f'outputs/{directory}/subsets.csv')
            pd.testing.assert_frame_equal(saved, table)
        self.assertEqual(len(subsets), 155)
        for (fold, draw, budget), indices in subsets.items():
            test = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), 'patient_id'])
            self.assertFalse(set(data.iloc[indices].patient_id) & test)

    def test_censoring_reference_does_not_change_breslow_training(self):
        time, event = np.array([100., 200.]), np.array([1, 0])
        module = training.FoldModel(torch.nn.Linear(1, 1), 'lora', [], time, event,
                                    [], time, event, [], time, event)
        reference = (np.array([100., 200., 300.]), np.array([1, 0, 1]))
        module.metric_reference = reference
        with patch.object(training, 'risk_all', return_value=np.array([0., 1.])), \
             patch.object(training, 'breslow_survival', return_value=np.ones((2, 12))) as baseline, \
             patch.object(training, 'fold_metrics', return_value={'uno_c_36m': .5}) as metrics, \
             patch.object(training, 'cox_loss', return_value=torch.tensor(1.)), \
             patch.object(module, 'log'):
            module.test_step(None, 0)
        self.assertIs(baseline.call_args.args[0], time)
        self.assertIs(baseline.call_args.args[1], event)
        self.assertIs(metrics.call_args.args[0], reference[0])
        self.assertIs(metrics.call_args.args[1], reference[1])

    def test_frozen_hash_rejects_mutated_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'adapter.pt').write_bytes(b'original')
            manifest = {'protocol': {'test': True},
                        'files': {'adapter.pt': training.sha256(output / 'adapter.pt')}}
            (output / 'FROZEN.json').write_text(json.dumps(manifest))
            with patch.object(external, 'protocol', return_value={'test': True}):
                external.checked_frozen(output, 'colipri')
                (output / 'adapter.pt').write_bytes(b'modified')
                with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
                    external.checked_frozen(output, 'colipri')

    def test_external_pairing_and_zero_difference(self):
        # Use the existing cohort and frozen scores as identical stand-in predictions;
        # exercise evaluation only, without claiming a new trained model result.
        frozen = pd.read_csv(training.ROOT / 'outputs/phase4_external/predictions.csv')
        predictions = frozen.loc[(frozen.model == 'colipri') & (frozen.population == 'secondary'),
                                 ['patient_id', 'risk']]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            predictions.to_csv(output / 'predictions.csv', index=False)
            (output / 'FROZEN.json').write_text('{}')
            training.write_json(output / 'PREDICTED.json', {
                'predictions_sha256': training.sha256(output / 'predictions.csv'),
                'frozen_sha256': training.sha256(output / 'FROZEN.json'),
                'cohort_sha256': training.sha256(external.EXTERNAL)})
            with patch.object(external, 'checked_frozen', return_value={}):
                external.evaluate(output, 'colipri')
            result = json.loads((output / 'summary.json').read_text())['populations']
            for population, n in [('primary', 143), ('secondary', 199)]:
                self.assertEqual(result[population]['patients'], n)
                self.assertAlmostEqual(result[population]['difference'], 0.)
                self.assertEqual(result[population]['difference_ci95'], [0., 0.])


if __name__ == '__main__':
    unittest.main()
