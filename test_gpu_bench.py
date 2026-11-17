import unittest
from unittest.mock import patch

from gpu_bench import stream_sample


def chunk(token_ids, completion_tokens, cached_tokens=0):
    return {
        "output_ids": token_ids,
        "meta_info": {
            "prompt_tokens": 2,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
        },
    }


class StreamSampleTests(unittest.TestCase):
    def test_measures_first_token_and_each_subsequent_gap(self):
        chunks = [chunk([4], 1), chunk([5], 2), chunk([6], 3)]
        with patch("gpu_bench.perf_counter_ns", side_effect=[0, 9_000_000, 11_000_000, 15_000_000]):
            result = stream_sample(lambda: iter(chunks), prompt_tokens=2, new_tokens=3)
        self.assertEqual(result["ttft_ms"], 9)
        self.assertEqual(result["tpot_ms"], 3)
        self.assertEqual(result["itl_ms"], [2, 4])
        self.assertEqual(result["e2e_ms"], 15)

    def test_rejects_grouped_tokens_early_exit_and_prefix_cache_hits(self):
        cases = [
            ([chunk([4, 5], 2), chunk([6], 3)], "one token"),
            ([chunk([4], 1)], "expected 3 tokens"),
            ([chunk([4], 1, cached_tokens=1)], "prefix cache"),
        ]
        for chunks, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                stream_sample(lambda: iter(chunks), prompt_tokens=2, new_tokens=3)
