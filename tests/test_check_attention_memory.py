import unittest
from unittest.mock import patch

import torch

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


if __name__ == "__main__":
    unittest.main()
