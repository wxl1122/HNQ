"""The physical GPU budget remains fixed when CPU metric stages overlap."""

import unittest

from tools.evaluate_seed_queue import seed_lanes


class SeedQueueTests(unittest.TestCase):
    def test_legacy_layout_and_seed_groups_are_preserved(self):
        lanes = seed_lanes([111, 222, 333, 444, 555], [0, 2])
        self.assertEqual([lane['key'] for lane in lanes], ['0', '2'])
        self.assertEqual(lanes[0]['seeds'], [111, 333, 555])
        self.assertEqual(lanes[1]['seeds'], [222, 444])

    def test_overlapping_jobs_cover_each_seed_once_on_only_two_cards(self):
        seeds = [111, 222, 333, 444, 555]
        lanes = seed_lanes(seeds, [0, 2], 2)
        self.assertEqual({lane['gpu'] for lane in lanes}, {0, 2})
        self.assertEqual(len({lane['key'] for lane in lanes}), len(lanes))
        self.assertEqual(sorted(seed for lane in lanes for seed in lane['seeds']), seeds)
        self.assertTrue(all(sum(lane['gpu'] == gpu for lane in lanes) <= 2 for gpu in [0, 2]))

    def test_resource_limit_and_duplicate_inputs_are_rejected(self):
        for seeds, gpus, slots in [([111], [0, 1, 2], 1), ([111], [0, 0], 1),
                                  ([111, 111], [0, 2], 1), ([111], [0, 2], 3)]:
            with self.assertRaises(ValueError):
                seed_lanes(seeds, gpus, slots)


if __name__ == '__main__':
    unittest.main()
