import copy
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import yaml

from tools import check_attention_memory as memory


class AttentionMemoryShutdownTests(unittest.TestCase):
    def test_report_waits_for_cuda_collective_on_every_rank(self):
        for rank in (0, 1, 2):
            events = []
            peaks = torch.tensor([3.989, 4.539])
            with patch.object(memory.dist, "is_initialized", return_value=True), \
                    patch.object(memory.dist, "get_rank", return_value=rank), \
                    patch.object(memory.dist, "all_reduce", side_effect=lambda *a, **kw: events.append("reduce")), \
                    patch.object(memory.torch, "tensor", return_value=peaks), \
                    patch.object(memory.torch.cuda, "synchronize", side_effect=lambda *a: events.append("sync")), \
                    patch.object(memory.torch.cuda, "max_memory_allocated", return_value=4 * 2**30), \
                    patch.object(memory.torch.cuda, "max_memory_reserved", return_value=5 * 2**30), \
                    patch("builtins.print"):
                memory.report_peak("test", torch.device("cuda", rank))
            self.assertEqual(events, ["sync", "reduce", "sync"])

    def test_pass_marker_follows_barrier_sync_and_group_destruction(self):
        events = []
        with patch.object(memory.dist, "is_initialized", return_value=True), \
                patch.object(memory.dist, "get_rank", return_value=0), \
                patch.object(memory.dist, "barrier", side_effect=lambda **kw: events.append(("barrier", kw["device_ids"]))), \
                patch.object(memory.dist, "destroy_process_group", side_effect=lambda: events.append("destroy")), \
                patch.object(memory.torch.cuda, "synchronize", side_effect=lambda *a: events.append("sync")), \
                patch("builtins.print", side_effect=lambda *a, **kw: events.append("pass")):
            memory.finish_preflight(torch.device("cuda", 2), 3)
        self.assertEqual(events, ["sync", ("barrier", [2]), "sync", "destroy", "pass"])

    def test_single_gpu_also_synchronizes_before_pass(self):
        events = []
        with patch.object(memory.dist, "is_initialized", return_value=False), \
                patch.object(memory.dist, "destroy_process_group") as destroy, \
                patch.object(memory.torch.cuda, "synchronize", side_effect=lambda *a: events.append("sync")), \
                patch("builtins.print", side_effect=lambda *a, **kw: events.append("pass")):
            memory.finish_preflight(torch.device("cuda", 0), 1)
        self.assertEqual(events, ["sync", "pass"])
        destroy.assert_not_called()

    def test_failure_destroys_initialized_group_without_barrier_or_pass(self):
        with patch.object(memory, "main", side_effect=SystemExit(2)), \
                patch.object(memory.dist, "is_initialized", return_value=True), \
                patch.object(memory.dist, "destroy_process_group") as destroy, \
                patch.object(memory.dist, "barrier") as barrier, \
                patch("builtins.print") as output:
            with self.assertRaises(SystemExit) as error:
                memory.run_preflight()
        self.assertEqual(error.exception.code, 2)
        destroy.assert_called_once_with()
        barrier.assert_not_called()
        output.assert_not_called()

    def test_finished_or_uninitialized_group_is_not_destroyed_again(self):
        for error in (None, RuntimeError("CUDA check failed")):
            with patch.object(memory, "main", side_effect=error), \
                    patch.object(memory.dist, "is_initialized", return_value=False), \
                    patch.object(memory.dist, "destroy_process_group") as destroy:
                if error is None:
                    memory.run_preflight()
                else:
                    with self.assertRaisesRegex(RuntimeError, "CUDA check failed"):
                        memory.run_preflight()
            destroy.assert_not_called()


class AttentionMemoryValidationTests(unittest.TestCase):
    def test_validation_keeps_configured_evrb_crop_and_reblur_complete_images(self):
        root = Path(__file__).resolve().parents[1]
        cases = (
            ("train_tdc_evrb_16bin_local_st_2ca_allscales_5gpu_scratch.yml", 256, "Center-crop validation"),
            ("train_tdc_evrb_16bin_local_st_2ca_allscales_3gpu_scratch.yml", 256, "Center-crop validation"),
            ("train_tdc_reblur_local_st_2ca_allscales_3gpu_scratch.yml", None, "Complete-image validation"),
        )
        for filename, size, label in cases:
            with self.subTest(config=filename):
                config = yaml.safe_load((root / "configs" / filename).read_text(encoding="utf-8"))
                original = copy.deepcopy(config)
                shape = (size, size) if size is not None else (260, 320)
                event_bins = config["model"]["event_in"]
                sample = {"blur": torch.zeros(3, *shape), "event": torch.zeros(event_bins, *shape)}
                calls = []

                class RecordModel(torch.nn.Module):
                    def forward(self, blur, event):
                        calls.append((tuple(blur.shape), tuple(event.shape), self.training, torch.is_grad_enabled()))
                        return blur

                model = RecordModel()
                with patch.object(memory, "build_dataset", return_value=[sample, sample]) as build, \
                        patch.object(memory.torch.cuda, "reset_peak_memory_stats") as reset, \
                        patch.object(memory, "report_peak") as report, \
                        patch("builtins.print"):
                    memory.check_validation(config, model, torch.device("cpu"), 3)
                expected = dict(config["datasets"]["val"], split="val", random_crop=False)
                build.assert_called_once_with(expected)
                reset.assert_called_once_with(torch.device("cpu"))
                report.assert_called_once_with(label, torch.device("cpu"))
                self.assertEqual(calls, [((1, 3, *shape), (1, event_bins, *shape), False, False)] * 2)
                self.assertEqual(config, original)

    def test_validation_sample_limit_and_default_crop_label(self):
        config = {"datasets": {"val": {"dataroot": "unused"}}}
        sample = {"blur": torch.zeros(3, 256, 256), "event": torch.zeros(16, 256, 256)}
        model = Mock(return_value=sample["blur"].unsqueeze(0))
        with patch.object(memory, "build_dataset", return_value=[sample] * 5), \
                patch.object(memory.torch.cuda, "reset_peak_memory_stats"), \
                patch.object(memory, "report_peak") as report, \
                patch("builtins.print"):
            memory.check_validation(config, model, torch.device("cpu"), 2)
        self.assertEqual(model.call_count, 2)
        report.assert_called_once_with("Center-crop validation", torch.device("cpu"))

    def test_empty_validation_fails_without_claiming_memory_success(self):
        with patch.object(memory, "build_dataset", return_value=[]), \
                patch.object(memory.torch.cuda, "reset_peak_memory_stats") as reset, \
                patch.object(memory, "report_peak") as report, \
                patch("builtins.print"):
            with self.assertRaisesRegex(ValueError, "Validation dataset is empty"):
                memory.check_validation({"datasets": {"val": {"patch_size": 256}}}, torch.nn.Identity(), torch.device("cpu"), 3)
        reset.assert_not_called()
        report.assert_not_called()


if __name__ == "__main__":
    unittest.main()
