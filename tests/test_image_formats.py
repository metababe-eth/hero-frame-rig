"""Verify image geometry reaches each real graph without network or inference."""
import unittest
from unittest.mock import patch

import still_engines as engines


class ImageFormats(unittest.TestCase):
    def test_omitted_format_keeps_existing_shoot_defaults(self):
        self.assertEqual(engines.shoot_canvas('flux3'), (2768, 1520))
        self.assertEqual(engines.shoot_canvas('moss', '1K'), (1376, 752))
        self.assertEqual(engines.shoot_canvas('klein4b'), (1344, 768))

    def test_all_engine_graphs_receive_the_selected_format_and_size(self):
        expected = {
            ('1K', 'Landscape'): (1376, 752),
            ('1K', 'Square'): (1024, 1024),
            ('1K', 'Portrait'): (752, 1376),
            ('2K', 'Landscape'): (2768, 1520),
            ('2K', 'Square'): (2048, 2048),
            ('2K', 'Portrait'): (1520, 2768),
        }
        for engine, spec in engines.ENGINES.items():
            files = {role: spec[role][0] for role in ('unet', 'clip', 'vae')}
            for (size, shape), dimensions in expected.items():
                with self.subTest(engine=engine, size=size, shape=shape):
                    width, height = engines.shoot_canvas(engine, size, shape)
                    self.assertEqual((width, height), dimensions)
                    self.assertEqual(width % 16, 0)
                    self.assertEqual(height % 16, 0)
                    self.assertLessEqual(width * height, 4_300_000)
                    with patch.object(engines, 'resolve', return_value=(engine, files)):
                        graph, _ = engines.graph('unchanged scene', 701, width, height,
                                                 engine, {}, 'Local ComfyUI',
                                                 analysis=True, expand_prompt=False)
                    self.assertEqual(graph['10']['inputs']['width'], width)
                    self.assertEqual(graph['10']['inputs']['height'], height)
                    self.assertEqual(graph['4']['inputs']['text'], 'unchanged scene')
                    self.assertEqual(graph['15']['inputs']['image'], ['12', 0])
                    self.assertEqual(graph['23']['inputs']['image'], ['12', 0])
                    if engine == 'klein4b':
                        self.assertEqual(graph['9']['inputs']['width'], width)
                        self.assertEqual(graph['9']['inputs']['height'], height)

    def test_invalid_new_format_fails_before_graph_construction(self):
        for engine, size, shape in [('moss', '10MP', 'Square'),
                                    ('flux3', '2K', 'Panorama'),
                                    ('moss-hd', '2K', 'Portrait')]:
            with self.subTest(engine=engine, size=size, shape=shape):
                with self.assertRaises(ValueError):
                    engines.shoot_canvas(engine, size, shape)


if __name__ == '__main__':
    unittest.main()
