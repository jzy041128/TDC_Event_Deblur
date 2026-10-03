import unittest
from unittest.mock import patch

import torch

from evaluate_ddp import parse_args, scale_event_exchange
from models.tdc_deblur_net import build_deblur_model


def make_model(mode="bidirectional_event_then_rgb_ca"):
    return build_deblur_model(base_dim=8, event_in=16, fusion_mode=mode).eval()


class EventExchangeDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_default_cli_leaves_exchange_unspecified(self):
        with patch("sys.argv", ["evaluate_ddp.py", "--config", "eval.yml", "--checkpoint", "best.pth"]):
            self.assertIsNone(parse_args().event_exchange_scale)

    def test_scale_one_preserves_weights_and_prediction(self):
        model = make_model()
        original = {name: value.clone() for name, value in model.state_dict().items()}
        blur = torch.randn(1, 3, 16, 16)
        event = torch.randn(1, 16, 16, 16)
        with torch.inference_mode():
            expected = model(blur, event)
        changes = scale_event_exchange(model, 1.0)
        self.assertEqual(len(changes), 3)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]), name)
        with torch.inference_mode():
            torch.testing.assert_close(model(blur, event), expected, rtol=0, atol=0)

    def test_half_scale_changes_only_six_event_exchange_gammas(self):
        model = make_model()
        original = {name: value.clone() for name, value in model.state_dict().items()}
        scale_event_exchange(model, 0.5)
        count = 0
        for name, value in model.state_dict().items():
            if name.endswith((".gamma_e2", ".gamma_e3")):
                self.assertTrue(torch.equal(value, original[name] * 0.5), name)
                count += 1
            else:
                self.assertTrue(torch.equal(value, original[name]), name)
        self.assertEqual(count, 6)

    def test_zero_scale_matches_direct_two_ca_with_same_retained_weights(self):
        model = make_model()
        direct = make_model("direct_event_to_rgb_2ca")
        source_state = model.state_dict()
        direct.load_state_dict({name: source_state[name] for name in direct.state_dict()}, strict=True)
        scale_event_exchange(model, 0.0)
        blur = torch.randn(1, 3, 16, 16)
        event = torch.randn(1, 16, 16, 16)
        with torch.inference_mode():
            torch.testing.assert_close(model(blur, event), direct(blur, event), rtol=0, atol=0)

    def test_invalid_scale_and_incompatible_fusion_are_rejected(self):
        model = make_model()
        for scale in (-1.0, float("nan"), float("inf")):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    scale_event_exchange(model, scale)
        for mode in ("direct_event_to_rgb_2ca", "independent_event_then_rgb_4ca"):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    scale_event_exchange(make_model(mode), 0.5)


if __name__ == "__main__":
    unittest.main()
