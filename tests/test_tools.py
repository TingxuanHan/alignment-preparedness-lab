import copy
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cpu_checks import authorize, dpo_loss, masked_nll, wilson
from validate_plan import validate


class ProtocolToolsTests(unittest.TestCase):
    def test_mask_and_empty_targets(self):
        self.assertEqual(masked_nll([-100, -2, -4], [False, True, True]), 3)
        with self.assertRaises(ValueError):
            masked_nll([-1], [False])
        with self.assertRaises(ValueError):
            masked_nll([-1], [])

    def test_preference_direction_and_extremes(self):
        self.assertAlmostEqual(dpo_loss(-3, -4, -3, -4), math.log(2))
        self.assertAlmostEqual(dpo_loss(-2, -4, -3, -3), 0.5981388693815918)
        self.assertLess(dpo_loss(-2, -4, -3, -3), dpo_loss(-4, -2, -3, -3))
        self.assertTrue(math.isfinite(dpo_loss(-10000, 0, 0, 0)))

    def test_zero_risk_not_proven_by_zero_observations(self):
        low, high = wilson(0, 100)
        self.assertAlmostEqual(low, 0)
        self.assertAlmostEqual(high, 0.03699349820698568)
        with self.assertRaises(ValueError):
            wilson(0, 0)

    def test_authorization_requires_trusted_grant_and_recipient(self):
        allowed = {"allowed@example.invalid"}
        action = {"tool": "send_message", "recipient": "allowed@example.invalid"}
        self.assertFalse(authorize(action, allowed))
        self.assertTrue(authorize(action, allowed, True))
        self.assertFalse(authorize({**action, "recipient": "other@example.invalid"}, allowed, True))
        self.assertFalse(authorize({"tool": "unknown"}, allowed, True))

    def test_draft_stays_explicitly_unready_and_rejects_unfair_initialization(self):
        plan = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
        self.assertEqual(len(validate(plan)), 6)
        changed = copy.deepcopy(plan)
        changed["arms"][2]["init"] = "sft_checkpoint"
        with self.assertRaises(ValueError):
            validate(changed)


if __name__ == "__main__":
    unittest.main()
