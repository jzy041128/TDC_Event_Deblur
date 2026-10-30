import copy
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data.distributed import DistributedSampler
import yaml

from data.dataset import build_dataset
from evaluate_ddp import tiled_forward
from models.tdc_deblur_net import build_deblur_model
from models.temporal_event_fusion import ConditionalTemporalTwoCAFusion, LocalSTRead
from utils.model_checkpoint import checkpoint_metadata, load_pretrained_weights


ROOT = Path(__file__).resolve().parents[1]


def load_config(kind):
    filename = (f"train_tdc_evrb_16bin_local_st_2ca_allscales_{kind}gpu_scratch.yml"
                if kind in (3, 5) else "eval_tdc_evrb_16bin_local_st_2ca_allscales_full.yml")
    return yaml.safe_load((ROOT / "configs" / filename).read_text(encoding="utf-8"))


class EVRBLocalSTConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_train_configs_differ_only_in_name_and_keep_existing_evrb_protocol(self):
        three, five = load_config(3), load_config(5)
        self.assertIn("3GPU", three["name"])
        self.assertIn("5GPU", five["name"])
        three["name"] = five["name"]
        self.assertEqual(three, five)
        old = yaml.safe_load((ROOT / "configs/train_tdc_evrb_rgb_gated4ca_scratch.yml").read_text())
        for key in ("datasets", "loss", "train", "path"):
            self.assertEqual(five[key], old[key], key)
        for key, value in old["model"].items():
            if key != "fusion_mode":
                self.assertEqual(five["model"][key], value, key)
        for split in ("train", "val"):
            self.assertEqual(five["datasets"][split]["event_bins"], 16)
            self.assertNotIn("event_cache_root", five["datasets"][split])
        self.assertIsNone(five["path"]["pretrain_model"])
        self.assertIsNone(five["path"]["resume_state"])

    def test_eval_keeps_identical_model_and_only_removes_validation_crop(self):
        evaluation = load_config("eval")
        for count in (3, 5):
            train = load_config(count)
            self.assertEqual(evaluation["model"], train["model"])
            val = copy.deepcopy(train["datasets"]["val"])
            val["patch_size"] = None
            self.assertEqual(evaluation["datasets"]["val"], val)
            for key, value in evaluation["train"].items():
                self.assertEqual(train["train"][key], value)

    def test_three_vs_five_gpus_have_different_steps_not_equal_global_batches(self):
        for count, expected in ((3, 547), (5, 328)):
            per_rank = len(DistributedSampler(range(1639), num_replicas=count, rank=0, drop_last=False))
            self.assertEqual(per_rank, expected)
            self.assertEqual(load_config(count)["datasets"]["train"]["batch_size"], 1)

    def test_all_stages_use_full_sixteen_bin_local_read_then_ca_and_backpropagate(self):
        config = load_config(5)["model"]
        config["base_dim"] = 8
        model = build_deblur_model(**config)
        reader_calls, ca_calls, handles = [], [], []
        for i, stage in enumerate(model.fusions):
            fusion = stage.temporal_fusion
            self.assertIsInstance(fusion, ConditionalTemporalTwoCAFusion)
            self.assertIsInstance(fusion.temporal_read, LocalSTRead)
            reader = fusion.temporal_read
            self.assertEqual(reader.num_heads, 1)
            self.assertEqual(reader.kernel_size, 3)
            self.assertIsNone(reader.spatial_bias)
            self.assertEqual(reader.query_chunk, 512)
            self.assertTrue(reader.checkpoint_chunks)
            def record_reader(_, args, i=i):
                reader_calls.append((i, args[1].ndim, args[1].shape[2]))
            handles.append(reader.register_forward_pre_hook(record_reader))
            handles.append(fusion.ca_e3.register_forward_hook(lambda _, args, out, i=i: ca_calls.append(i)))
        blur, event = torch.rand(1, 3, 17, 19), torch.randn(1, 16, 17, 19)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = model(blur, event)
            self.assertEqual(result.shape, blur.shape)
            result.square().mean().backward()
            for key, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, key)
                self.assertTrue(torch.isfinite(parameter.grad).all(), key)
            optimizer.step()
        self.assertEqual(reader_calls, [(i, 5, 16) for i in range(3)] * 2)
        self.assertEqual(ca_calls, [0, 1, 2] * 2)
        for handle in handles:
            handle.remove()
        checkpoint = {"model_state_dict": model.state_dict(), "event3d_conv_type": "tdc",
                      **checkpoint_metadata(model, config)}
        for kind in (3, "eval"):
            other = load_config(kind)["model"]
            other["base_dim"] = 8
            load_pretrained_weights(build_deblur_model(**other), checkpoint)

    def test_official_voxels_work_with_crop_and_full_image_configs(self):
        with tempfile.TemporaryDirectory() as root:
            sequence = Path(root) / "00000"
            for folder in ("blur_processed", "gt_processed", "event_voxel"):
                (sequence / folder).mkdir(parents=True)
            height, width = 260, 268
            pixels = np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3)
            pixels = (pixels % 256).astype(np.uint8)
            for folder in ("blur_processed", "gt_processed"):
                Image.fromarray(pixels).save(sequence / folder / "00000.png")
            voxel = np.linspace(-2, 2, 16 * height * width, dtype=np.float32).reshape(16, height, width)
            np.savez(sequence / "event_voxel" / "00000.npz", data=voxel)
            full = load_config("eval")["datasets"]["val"]
            full.update(dataroot=root, random_crop=False)
            full_sample = build_dataset(full)[0]
            cropped = load_config(5)["datasets"]["val"]
            cropped.update(dataroot=root, random_crop=False)
            crop_sample = build_dataset(cropped)[0]
            top, left = (height - 256) // 2, (width - 256) // 2
            self.assertEqual(tuple(full_sample["event"].shape), (16, height, width))
            for key in ("blur", "gt", "event"):
                torch.testing.assert_close(crop_sample[key], full_sample[key][:, top:top + 256, left:left + 256])

    @unittest.skipUnless(os.environ.get("TDC_EVRB_LOCAL_ST_FULL_SIZE_TESTS") == "1", "Opt-in full-size CPU synthetic check")
    def test_full_size_training_and_tiled_inference(self):
        config = load_config(5)
        torch.manual_seed(config["train"]["seed"])
        model = build_deblur_model(**config["model"])
        parameters = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"])
        blur, event = torch.rand(1, 3, 256, 256), torch.randn(1, 16, 256, 256)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = model(blur, event)
            (result - blur).abs().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        model.eval()
        with torch.no_grad():
            full_blur, full_event = torch.rand(1, 3, 260, 320), torch.randn(1, 16, 260, 320)
            output = tiled_forward(model, full_blur, full_event, 256, 32)
            self.assertEqual(output.shape, full_blur.shape)
            self.assertTrue(torch.isfinite(output).all())
        print(f"EVRB L CPU PASS: parameters={parameters:,}, two 256x256 AdamW steps, tiled 260x320 inference; not a GPU memory measurement", flush=True)


if __name__ == "__main__":
    unittest.main()
