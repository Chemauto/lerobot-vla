"""Artifact safety and metric denominators; no training or GPU required."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'model'))
from artifacts import collect_checkpoints, summarize, write_batch_report, write_json, parse_losses
from run_experiment import validated_dataset, dataset_path


class ExperimentArtifactsTest(unittest.TestCase):
    def test_loss_parser_uses_exact_step_not_rounded_logger_step(self):
        text = ('Training: 9%|bar| 2800/30000 [00:10]INFO step:3K loss:0.491 l1_loss:0.066 kld_loss:0.042\n'
                'INFO step 3000: eval_loss=0.1081\n'
                'Training: 10%|bar| 3000/30000 [00:11]INFO step:3K loss:0.425\n'
                'INFO step:3100 loss:0.40\n')
        parsed = parse_losses(text)
        self.assertEqual([p['step'] for p in parsed['train']], [2800,3000,3100])
        self.assertEqual(parsed['train'][0]['l1_loss'], .066)
        self.assertEqual(parsed['validation'], [dict(step=3000,loss=.1081)])

    def test_loss_parser_resumed_steps_replace_earlier_values(self):
        parsed = parse_losses('INFO step:100 loss:0.5\nINFO step:100 loss:0.4\n'
                              'INFO step 100: eval_loss=0.3\nINFO step 100: eval_loss=0.2\n'
                              'INFO step:200 loss:nan\n')
        self.assertEqual(parsed['train'], [dict(step=100,loss=.4)])
        self.assertEqual(parsed['validation'], [dict(step=100,loss=.2)])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'act20261028/custom_dataset'
        self.staging = self.run / 'metadata/trainer/checkpoints'
        self.staging.mkdir(parents=True)
        (self.run / 'checkpoints').mkdir()
        (self.run / 'metadata/train.log').write_text('')

    def save_official(self, step, loss):
        path = self.staging / f'{step:06d}'
        (path / 'pretrained_model').mkdir(parents=True)
        (path / 'pretrained_model/model.safetensors').write_bytes(str(step).encode())
        (path / 'training_state').mkdir()
        write_json(path / 'training_state/training_step.json', {'step': step})
        if loss is not None:
            with (self.run / 'metadata/train.log').open('a') as f:
                f.write(f'INFO step {step}: eval_loss={loss:.4f}\n')
        last = self.staging / 'last'
        if last.is_symlink():
            last.unlink()
        last.symlink_to(path.name)
        return path

    def test_latest_best_tie_and_unpublished_save(self):
        self.save_official(100, .2)
        collect_checkpoints(self.run)
        self.save_official(200, .1)
        collect_checkpoints(self.run)
        self.save_official(300, .1)
        # A newer save being written must never be copied or removed.
        incomplete = self.staging / '000400'
        incomplete.mkdir()
        (incomplete / 'still_writing').write_text('partial')
        collect_checkpoints(self.run)
        best = self.run / 'checkpoints/best'
        latest = self.run / 'checkpoints/latest'
        self.assertEqual((best / 'pretrained_model/model.safetensors').read_bytes(), b'200')
        self.assertEqual((latest / 'pretrained_model/model.safetensors').read_bytes(), b'300')
        self.assertEqual(json.loads((latest / 'training_state/training_step.json').read_text())['step'], 300)
        self.assertTrue((self.staging / 'last').exists())
        self.assertTrue(incomplete.exists())
        self.assertFalse((self.staging / '000100').exists())
        self.assertFalse((self.staging / '000200').exists())
        self.assertEqual(sorted(p.name for p in (self.run / 'checkpoints').iterdir()), ['best', 'latest'])
        # Idempotent observation must not duplicate checkpoint history.
        collect_checkpoints(self.run)
        self.assertEqual(len(json.loads((self.run / 'metadata/checkpoint_history.json').read_text())), 3)

    def test_missing_loss_does_not_delete_or_invent_best(self):
        source = self.save_official(100, None)
        with self.assertRaises(ValueError):
            collect_checkpoints(self.run)
        self.assertTrue(source.exists())
        self.assertFalse((self.run / 'checkpoints/best').exists())

    def test_metrics_use_all_cubes_and_separate_raw_success(self):
        result = {'target_cube_count': 4, 'episodes': [
            dict(grasped_cubes=4, cubes_in_box=4, success=True, official_success_seen=True, steps=100),
            dict(grasped_cubes=2, cubes_in_box=2, success=False, official_success_seen=True, steps=200),
        ]}
        summary = summarize(result)
        self.assertEqual(summary['grasp_rate'], .75)
        self.assertEqual(summary['placement_rate'], .75)
        self.assertEqual(summary['success_rate'], .5)
        self.assertEqual(summary['official_rate'], 1)

    def test_planned_report_does_not_fabricate_scores(self):
        state = dict(status='planned', model='act', dataset='data/custom_dataset',
                     steps=15000, eval_episodes=10, eval_seed=10000, max_steps=3600)
        write_json(self.run / 'metadata/run.json', state)
        write_batch_report(self.run.parent)
        text = (self.run.parent / 'README.md').read_text()
        self.assertIn('待训练', text)
        self.assertNotIn('0.0%', text)
        self.assertIn('暂无成功率', (self.run / 'README.md').read_text())

    def test_validation_accepts_sizes_other_than_200(self):
        dataset = self.run/'example_data'
        dataset.mkdir()
        write_json(dataset/'collection_manifest.json', dict(episodes=[{}, {}, {}], task_variant={'cube_count': 1}))
        write_json(dataset/'validation_report.json', dict(episodes=3, full_video_decode=True, physics_replay={'episodes': 3}))
        self.assertEqual(len(validated_dataset(dataset)['episodes']), 3)
        write_json(dataset/'validation_report.json', dict(episodes=3, full_video_decode=True, physics_replay={'episodes': 2}))
        with self.assertRaisesRegex(ValueError, 'complete physics replay'):
            validated_dataset(dataset)

    def test_paths_and_reports_accept_arbitrary_dataset_names(self):
        self.assertEqual(dataset_path('/tmp/my_dataset'), Path('/tmp/my_dataset'))
        self.assertEqual(dataset_path('custom_dataset').name, 'custom_dataset')
        state = dict(status='planned', model='act', dataset='data/custom_dataset',
                     steps=300, eval_episodes=3, eval_seed=10000, max_steps=20)
        write_json(self.run/'metadata/run.json', state)
        write_batch_report(self.run.parent)
        self.assertIn('custom_dataset', (self.run.parent/'README.md').read_text())


if __name__ == '__main__':
    unittest.main()
