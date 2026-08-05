from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MciTransferRunnerTest(unittest.TestCase):
    def test_runner_covers_both_targets_with_validation_only_selection(self):
        source = (ROOT / 'scripts' / 'run_mci_transfer_finetune.sh').read_text(encoding='utf-8')
        for token in (
            'levir_cc_mci_init_seed',
            'second_cc_mci_init_seed',
            'MCI_INIT_CHECKPOINT',
            'val_baseline_pareto',
            '--require_audited_validation_baseline',
            'Do not use test metrics to choose a run.',
        ):
            self.assertIn(token, source)
        self.assertNotIn('test_specific_snapshot_sgc_card.sh', source)


if __name__ == '__main__':
    unittest.main()
