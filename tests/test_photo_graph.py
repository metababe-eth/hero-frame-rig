"""Versioned still graph policy and offline execution boundary."""
import copy
import unittest

import rig_workflow as rig
from replay_desktop import inspect_graph


class PhotoGraph(unittest.TestCase):
    def graph(self, blur=12, photo=True, guide=0):
        return rig.develop('still.png', 'raw.png', blur, .4, 'digital',
                           round_masks={'main': [f'w{i}.png' for i in range(5)]},
                           width=1200, height=1600, photo_edges=photo,
                           guide=guide, guide_masks={'main': 'guide.png'} if guide else None)

    def test_photo_zero_and_blurred_graphs_replay_with_and_without_guide(self):
        for blur in (0, 12):
            for guide in (0, 35):
                with self.subTest(blur=blur, guide=guide):
                    graph = self.graph(blur=blur, guide=guide)
                    report = inspect_graph(graph)
                    self.assertTrue(report['ok'], report['errors'])
                    self.assertEqual(graph['30']['inputs']['highlight_recovery'], 0)
                    self.assertIn(rig.round_lens.PHOTO_VERSION, graph['32']['_meta']['title'])

    def test_legacy_graph_remains_expanded_and_replayable(self):
        for blur in (0, 12):
            graph = self.graph(blur=blur, photo=False)
            self.assertEqual(graph['30']['inputs']['highlight_recovery'], 1)
            self.assertEqual(graph['32']['_meta']['title'], 'Save hero frame / ' + rig.round_lens.VERSION)
            self.assertTrue(inspect_graph(graph)['ok'])

    def test_photo_version_cannot_disguise_expanded_or_extra_linear_inputs(self):
        for key, value in [('highlight_recovery', 1), ('source_gamma', 1), ('extra', True)]:
            graph = self.graph()
            graph['30']['inputs'][key] = value
            self.assertFalse(inspect_graph(graph)['ok'])
        graph = self.graph()
        graph['32']['_meta']['title'] += '-unknown'
        self.assertFalse(inspect_graph(graph)['ok'])

    def test_unverified_layered_path_is_rejected(self):
        with self.assertRaises(ValueError):
            rig.develop('still.png', 'raw.png', 12, .4, photo_edges=True)
        with self.assertRaises(ValueError):
            rig.develop('still.png', 'raw.png', 12, .4, photo_edges=True,
                        layers=('plate.png', 'pd.png', 'alpha.png'), round_masks={})


if __name__ == '__main__':
    unittest.main()
