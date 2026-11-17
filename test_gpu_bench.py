from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gpu_bench import completed_sample


class CompletedSampleTests(unittest.TestCase):
    def test_times_each_token_after_copy_to_host(self):
        events = []

        def now():
            events.append("clock")
            return next(times)

        def copy(token_id):
            events.append(token_id)
            return token_id

        times = iter([0, 9_000_000, 11_000_000, 15_000_000])
        tokens = [SimpleNamespace(item=lambda token_id=i: copy(token_id)) for i in (4, 5, 6)]
        with patch("gpu_bench.perf_counter_ns", side_effect=now):
            result = completed_sample(lambda: iter(tokens), new_tokens=3)
        self.assertEqual(events, ["clock", 4, "clock", 5, "clock", 6, "clock"])
        self.assertEqual(result["ttft_ms"], 9)
        self.assertEqual(result["tpot_ms"], 3)
        self.assertEqual(result["itl_ms"], [2, 4])
        self.assertEqual(result["e2e_ms"], 15)

    def test_rejects_short_generation(self):
        with self.assertRaisesRegex(ValueError, "expected 3 tokens"):
            completed_sample(lambda: iter([SimpleNamespace(item=lambda: 4)]), new_tokens=3)
