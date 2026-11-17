import unittest

from speculative import NgramDraft, verified_tokens


class SpeculativeTests(unittest.TestCase):
    def test_rejection_emits_correction_and_discards_the_suffix(self):
        self.assertEqual(verified_tokens([2, 3, 4], [2, 8, 4, 5]), [2, 8])
        self.assertEqual(verified_tokens([2, 3, 4], [8, 3, 4, 5]), [8])

    def test_acceptance_emits_bonus_token(self):
        self.assertEqual(verified_tokens([2, 3, 4], [2, 3, 4, 5]), [2, 3, 4, 5])

    def test_ngram_uses_longest_available_match_and_tracks_new_tokens(self):
        draft = NgramDraft()
        history = [1, 2, 3, 4, 5, 2, 6, 1, 2]
        self.assertEqual(draft.propose(history, 3), [3, 4, 5])
        history.extend([3, 9, 1, 2, 3])
        self.assertEqual(draft.propose(history, 2), [9, 1])
        self.assertEqual(NgramDraft().propose([1, 2, 3], 3), [])
