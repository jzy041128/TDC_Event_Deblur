import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from tools import benchmark_reblur_fusion_memory as bench


CONFIGS = {
    "gate4ca": "train_tdc_reblur_rgb_gated4ca_channel_3gpu_scratch.yml",
    "mean_v": "train_tdc_reblur_mean_v_2ca_allscales_3gpu_scratch.yml",
    "temporal": "train_tdc_reblur_cond_temporal_2ca_3gpu_scratch.yml",
    "local_st": "train_tdc_reblur_local_st_2ca_allscales_3gpu_scratch.yml",
    "dual_expert": "train_tdc_reblur_identity_preserving_dual_expert_allscales_3gpu_scratch.yml",
    "bi_gru": "train_tdc_reblur_conditioned_bi_gru_allscales_3gpu_scratch.yml",
    "dynamic_filter": "train_tdc_reblur_event_controlled_dynamic_filter_3gpu_scratch.yml",
    "direct2ca": "train_tdc_reblur_direct2ca_scratch.yml",
}


def archive(root):
    entries = yaml.safe_load(bench.DEFAULT_MANIFEST.read_text())['experiments']
    for entry in entries:
        folder = Path(root) / entry['directory']
        folder.mkdir()
        (folder / 'train_config.yml').write_bytes((bench.ROOT / 'configs' / CONFIGS[entry['id']]).read_bytes())
        (folder / 'train_log.txt').write_text(f"Device: cuda:0 | DDP world_size: {entry['original_world_size']} | per_gpu_batch: 1\n")
    return entries


def fake_metrics(config_path, warmup=3, measured=3, samples=3):
    import hashlib
    raw = Path(config_path).read_bytes()
    config = yaml.safe_load(raw)
    return {'schema': 'attention_memory.v1', 'status': 'pass', 'world_size': 3, 'per_gpu_batch': 1,
            'precision': 'float32', 'config_sha256': hashlib.sha256(raw).hexdigest(),
            'model_config': config['model'], 'parameters': 3000000,
            'warmup_steps': warmup, 'measured_steps': measured, 'train_samples': 486,
            'train': {'allocated_gib': 4.0, 'reserved_gib': 4.5, 'step_seconds_max': 0.7,
                      'batch_shapes': {'event': [1, 6, 256, 256]}},
            'validation': {'allocated_gib': 1.2, 'reserved_gib': 4.5, 'patch_size': None,
                           'mode': 'Complete-image validation', 'samples_checked': samples, 'dataset_samples': 903}}


class FusionMemoryBenchmarkTests(unittest.TestCase):
    def test_manifest_has_eight_unique_archives_and_preserves_original_configs(self):
        with tempfile.TemporaryDirectory() as root:
            archive(root)
            plan = bench.build_plan(bench.DEFAULT_MANIFEST, root)
            self.assertEqual(len(plan), 8)
            self.assertEqual({row['id'] for row in plan}, bench.IDS)
            for row in plan:
                self.assertEqual(row['raw_config'], Path(row['source_config']).read_bytes())
                self.assertEqual(row['original_world_size'], 5 if row['id'] == 'direct2ca' else 3)
            temporal = next(row for row in plan if row['id'] == 'temporal')
            self.assertEqual(temporal['model_config']['fusion_options']['local_st_scales'], [2])

    def test_refuses_missing_archives_and_coarse_gru_or_wrong_training_history(self):
        with tempfile.TemporaryDirectory() as root:
            entries = archive(root)
            gru = next(entry for entry in entries if entry['id'] == 'bi_gru')
            source = Path(root) / gru['directory'] / 'train_config.yml'
            original = source.read_bytes()
            config = yaml.safe_load(original)
            config['model']['fusion_options']['gru_scales'] = [2]
            source.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, 'coarse-only GRU'):
                bench.build_plan(bench.DEFAULT_MANIFEST, root)
            source.write_bytes(original)
            log = source.parent / 'train_log.txt'
            log.write_text('DDP world_size: 5 | per_gpu_batch: 1')
            with self.assertRaisesRegex(ValueError, 'GPU/batch history'):
                bench.build_plan(bench.DEFAULT_MANIFEST, root)
            source.unlink()
            with self.assertRaises(FileNotFoundError):
                bench.build_plan(bench.DEFAULT_MANIFEST, root)

    def test_worker_command_uses_three_ranks_and_structured_output(self):
        command = bench.worker_command(Path('archived.yml'), Path('metrics.json'), 3, 3, 3)
        self.assertIn('--nproc_per_node=3', command)
        self.assertIn('torch.distributed.run', command)
        self.assertEqual(command[command.index('--config') + 1], 'archived.yml')
        self.assertEqual(command[command.index('--result-json') + 1], 'metrics.json')

    def test_dry_run_creates_no_outputs_or_cuda_workers(self):
        with tempfile.TemporaryDirectory() as root:
            archive(root)
            output = Path(root) / 'not_written'
            with patch('sys.argv', ['benchmark', '--experiments-root', root, '--output-dir', str(output), '--dry-run']), \
                    patch.object(bench, 'run_worker') as worker, patch('builtins.print'):
                self.assertEqual(bench.main(), 0)
            self.assertFalse(output.exists())
            worker.assert_not_called()

    def test_eight_sequential_workers_write_json_csv_and_exact_config_snapshots(self):
        with tempfile.TemporaryDirectory() as root:
            archive(root)
            output = Path(root) / 'results'
            calls = []
            def worker(command, environment, log, timeout):
                calls.append(command)
                self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '0,2,3')
                source = command[command.index('--config') + 1]
                target = command[command.index('--result-json') + 1]
                Path(target).write_text(json.dumps(fake_metrics(source)))
                Path(log).write_text('mock measurement only')
                return 0
            with patch('sys.argv', ['benchmark', '--experiments-root', root, '--output-dir', str(output)]), \
                    patch.object(bench, 'run_worker', side_effect=worker), patch('builtins.print'):
                self.assertEqual(bench.main(), 0)
            self.assertEqual(len(calls), 8)
            rows = json.loads((output / 'summary.json').read_text())['results']
            self.assertTrue(all(row['status'] == 'pass' for row in rows))
            self.assertEqual(rows[-1]['original_training_gpus'], 5)
            self.assertEqual(rows[-1]['benchmark_gpus'], 3)
            for row in rows:
                self.assertEqual((output / 'configs' / (row['id'] + '.yml')).read_bytes(), Path(row['source_config']).read_bytes())
            with (output / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 8)
            self.assertEqual(csv_rows[0]['train_allocated_gib'], '4.0')

    def test_failure_stops_remaining_workers_and_keeps_partial_results(self):
        with tempfile.TemporaryDirectory() as root:
            archive(root)
            output = Path(root) / 'failed_results'
            with patch('sys.argv', ['benchmark', '--experiments-root', root, '--output-dir', str(output)]), \
                    patch.object(bench, 'run_worker', return_value=1) as worker, patch('builtins.print'):
                self.assertEqual(bench.main(), 1)
            self.assertEqual(worker.call_count, 1)
            row = json.loads((output / 'summary.json').read_text())['results'][0]
            self.assertEqual(row['status'], 'failed')
            self.assertNotIn('train_allocated_gib', row)

    def test_rejects_failed_stale_mismatched_or_nonfinite_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            archive(root)
            entry = bench.build_plan(bench.DEFAULT_MANIFEST, root)[0]
            result = fake_metrics(entry['source_config'])
            target = Path(root) / 'metrics.json'
            for key, value in (('status', 'failed'), ('world_size', 5), ('config_sha256', 'wrong')):
                data = copy.deepcopy(result)
                data[key] = value
                target.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    bench.checked_metrics(target, 0, entry, 3, 3, 3)
            result['train']['allocated_gib'] = float('nan')
            target.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'Invalid memory'):
                bench.checked_metrics(target, 0, entry, 3, 3, 3)
            with self.assertRaises(RuntimeError):
                bench.checked_metrics(target, 1, entry, 3, 3, 3)


if __name__ == '__main__':
    unittest.main()
