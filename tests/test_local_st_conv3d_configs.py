import copy
import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from evaluate_ddp import tiled_forward
from models.tdc_deblur_net import build_deblur_model
from models.temporal_event_fusion import ConditionalTemporalTwoCAFusion, LocalSTRead
from train_ddp import make_experiment_dir
from utils.model_checkpoint import checkpoint_metadata, load_pretrained_weights


ROOT = Path(__file__).resolve().parents[1]
TRAIN = 'train_tdc_evrb_16bin_local_st_2ca_conv3d5_allscales_5gpu_scratch.yml'
EVAL = 'eval_tdc_evrb_16bin_local_st_2ca_conv3d5_allscales_full.yml'
BASELINE = 'train_tdc_evrb_16bin_local_st_2ca_allscales_5gpu_scratch.yml'
GROUP = 'LocalST2CA_Conv3d5_AllScales'
REBLUR = 'train_tdc_reblur_6bin_local_st_2ca_conv3d5_allscales_3gpu_scratch.yml'
GOPRO = 'train_tdc_gopro_6bin_local_st_2ca_conv3d5_allscales_5gpu_scratch.yml'
REVD = 'train_tdc_revd_6bin_local_st_2ca_conv3d5_allscales_5gpu_finetune.yml'


def config(filename):
    return yaml.safe_load((ROOT / 'configs' / filename).read_text(encoding='utf-8'))


class LocalSTConv3dConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_only_convolution_name_and_output_group_differ_from_tdc_control(self):
        plain, baseline = config(TRAIN), config(BASELINE)
        self.assertIn('Conv3d5_AllScales_5GPU', plain['name'])
        self.assertEqual(plain['model']['event3d_conv_type'], 'conv3d')
        self.assertEqual(plain['path']['experiments_root'], '/data1/jzy/experiments/' + GROUP)
        plain['name'] = baseline['name']
        plain['model']['event3d_conv_type'] = 'tdc'
        plain['path']['experiments_root'] = baseline['path']['experiments_root']
        self.assertEqual(plain, baseline)
        self.assertIsNone(plain['path']['pretrain_model'])
        self.assertIsNone(plain['path']['resume_state'])

    def test_evaluation_preserves_exact_model_metadata_and_only_removes_crop(self):
        train, evaluation = config(TRAIN), config(EVAL)
        self.assertEqual(evaluation['model'], train['model'])
        val = copy.deepcopy(train['datasets']['val'])
        val['patch_size'] = None
        self.assertEqual(val, evaluation['datasets']['val'])
        self.assertEqual(evaluation['train'], {'seed': 42, 'deterministic': True})

    def test_reblur_preserves_the_existing_three_gpu_tdc_protocol(self):
        plain = config(REBLUR)
        tdc = config('train_tdc_reblur_local_st_2ca_allscales_3gpu_scratch.yml')
        self.assertIn('3GPU_FromScratch_200ep', plain['name'])
        self.assertEqual(plain['model']['event3d_conv_type'], 'conv3d')
        self.assertIsNone(plain['datasets']['val']['patch_size'])
        self.assertEqual(plain['path']['experiments_root'], '/data1/jzy/experiments/' + GROUP)
        plain['name'] = tdc['name']
        plain['model']['event3d_conv_type'] = 'tdc'
        plain['path']['experiments_root'] = tdc['path']['experiments_root']
        self.assertEqual(plain, tdc)

    def test_followup_models_match_and_keep_dataset_specific_training_settings(self):
        expected_model = config(TRAIN)['model']
        expected_model['event_in'] = 6
        # The archived REBlur configuration relies on the default rgb_in=3.
        self.assertEqual(expected_model.pop('rgb_in'), 3)
        for filename in (REBLUR, GOPRO, REVD):
            cfg = config(filename)
            self.assertEqual(cfg['model'], expected_model, filename)
            self.assertEqual(cfg['path']['experiments_root'], '/data1/jzy/experiments/' + GROUP)
            self.assertIsNone(cfg['path']['resume_state'])
        gopro = config(GOPRO)
        revd = config(REVD)
        for candidate, reference in (
            (gopro, config('train_tdc_gopro_rgb_gated4ca_conv3d_scratch.yml')),
            (revd, config('train_tdc_revd_finetune.yml')),
        ):
            for field in ('datasets', 'loss', 'train'):
                self.assertEqual(candidate[field], reference[field], field)
        self.assertIsNone(gopro['path']['pretrain_model'])
        self.assertEqual(revd['path']['pretrain_model'], '/data1/jzy/experiments/' + GROUP
                         + '/REPLACE_WITH_GOPRO_L_CONV3D5_RUN/best.pth')

    def test_gopro_and_revd_full_image_configs_keep_exact_models(self):
        for train_name, dataset in ((GOPRO, 'gopro'), (REVD, 'revd')):
            train = config(train_name)
            evaluation = config(f'eval_tdc_{dataset}_6bin_local_st_2ca_conv3d5_allscales_full.yml')
            self.assertEqual(evaluation['model'], train['model'])
            val = copy.deepcopy(train['datasets']['val'])
            val['patch_size'] = None
            self.assertEqual(evaluation['datasets'], {'val': val})
            self.assertEqual(evaluation['train'], {'seed': 42, 'deterministic': True})

    def test_revd_accepts_the_matching_gopro_checkpoint_strictly(self):
        gopro = config(GOPRO)['model']
        revd = config(REVD)['model']
        gopro['base_dim'] = revd['base_dim'] = 8
        source = build_deblur_model(**gopro)
        checkpoint = {'model_state_dict': source.state_dict(), 'event3d_conv_type': 'conv3d',
                      **checkpoint_metadata(source, gopro)}
        target = build_deblur_model(**revd)
        load_pretrained_weights(target, checkpoint)
        for name, tensor in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], tensor, rtol=0, atol=0)

    def test_original_trainer_creates_group_and_keeps_resume_in_original_run(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config(TRAIN)
            cfg['path']['experiments_root'] = str(Path(root) / GROUP)
            moment = datetime.datetime(2026, 11, 4, 12, 30, 0)
            with patch('train_ddp.rank0', return_value=True), \
                 patch('train_ddp.dist.broadcast_object_list'), \
                 patch('train_ddp.dist.barrier'), \
                 patch('train_ddp.datetime.datetime') as clock:
                clock.now.return_value = moment
                directory, resume = make_experiment_dir(cfg, None)
                expected = Path(root) / GROUP / (cfg['name'] + '_20261104_123000')
                self.assertEqual(Path(directory), expected)
                self.assertTrue(expected.is_dir())
                self.assertFalse(resume)
                checkpoint = expected / 'latest.pth'
                checkpoint.touch()
                cfg['path']['experiments_root'] = str(Path(root) / 'different_group')
                resumed, resume = make_experiment_dir(cfg, str(checkpoint))
                self.assertTrue(resume)
                self.assertEqual(Path(resumed), expected)
                self.assertFalse((Path(root) / 'different_group').exists())

    def test_all_scales_keep_full_time_local_ca_and_plain_conv_with_finite_updates(self):
        for bins in (6, 16):
            with self.subTest(bins=bins):
                cfg = config(TRAIN)['model']
                cfg.update(base_dim=8, event_in=bins)
                model = build_deblur_model(**cfg)
                calls, ca_calls, handles = [], [], []
                for i, block in enumerate((model.event3d_stem, *model.event3d_down)):
                    self.assertEqual(block.tdc.conv_type, 'conv3d')
                    self.assertEqual(block.tdc.conv.kernel_size, (5, 3, 3))
                    self.assertEqual(block.tdc.conv.padding, (2, 1, 1))
                    self.assertEqual(block.tdc.conv.stride, (1, 1, 1) if i == 0 else (1, 2, 2))
                    self.assertEqual(block.spatial.kernel_size, (1, 3, 3))
                    self.assertIsNone(block.tdc.conv.bias)
                for i, stage in enumerate(model.fusions):
                    fusion = stage.temporal_fusion
                    self.assertIsInstance(fusion, ConditionalTemporalTwoCAFusion)
                    self.assertIsInstance(fusion.temporal_read, LocalSTRead)
                    self.assertEqual(fusion.temporal_read.num_heads, 1)
                    self.assertEqual(fusion.temporal_read.kernel_size, 3)
                    self.assertTrue(fusion.temporal_read.checkpoint_chunks)
                    self.assertEqual(fusion.temporal_read.query_chunk, 512)
                    self.assertIsNone(fusion.temporal_read.spatial_bias)
                    def record_time(_, args, index=i):
                        calls.append((index, args[1].ndim, args[1].shape[2]))
                    handles.append(fusion.temporal_read.register_forward_pre_hook(record_time))
                    handles.append(fusion.ca_e3.register_forward_hook(lambda _, args, out, index=i: ca_calls.append(index)))
                blur, event = torch.rand(1, 3, 17, 19), torch.randn(1, bins, 17, 19)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    output = model(blur, event)
                    self.assertEqual(output.shape, blur.shape)
                    output.square().mean().backward()
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    optimizer.step()
                self.assertEqual(calls, [(i, 5, bins) for i in range(3)] * 2)
                self.assertEqual(ca_calls, [0, 1, 2] * 2)
                for handle in handles:
                    handle.remove()

    def test_exact_models_preserve_parameter_count_and_seeded_initial_weights(self):
        torch.manual_seed(42)
        tdc = build_deblur_model(**config(BASELINE)['model'])
        torch.manual_seed(42)
        plain = build_deblur_model(**config(TRAIN)['model'])
        self.assertEqual(sum(p.numel() for p in plain.parameters()), 2980846)
        self.assertEqual(set(plain.state_dict()), set(tdc.state_dict()))
        for name, parameter in tdc.state_dict().items():
            self.assertTrue(torch.equal(parameter, plain.state_dict()[name]), name)

    def test_conv3d_checkpoint_roundtrip_and_tiled_eval_reject_tdc_checkpoint(self):
        cfg = config(TRAIN)['model']
        cfg['base_dim'] = 8
        source = build_deblur_model(**cfg).eval()
        checkpoint = {'model_state_dict': source.state_dict(), 'event3d_conv_type': 'conv3d',
                      **checkpoint_metadata(source, cfg)}
        target_cfg = config(EVAL)['model']
        target_cfg['base_dim'] = 8
        target = build_deblur_model(**target_cfg).eval()
        load_pretrained_weights(target, checkpoint)
        blur, event = torch.rand(1, 3, 17, 19), torch.randn(1, 16, 17, 19)
        with torch.no_grad():
            torch.testing.assert_close(source(blur, event), target(blur, event), rtol=0, atol=0)
            output = tiled_forward(target, blur, event, 12, 4)
            self.assertEqual(output.shape, blur.shape)
            self.assertTrue(torch.isfinite(output).all())
        wrong = dict(checkpoint, event3d_conv_type='tdc')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            load_pretrained_weights(target, wrong)


if __name__ == '__main__':
    unittest.main()
