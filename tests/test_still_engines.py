import unittest
from unittest.mock import patch
import still_engines as engines


def cloud_info():
    info = {n: {} for n in ('CLIPTextEncode', 'VAEDecode', 'EmptyFlux2LatentImage',
                            'SaveImage', 'KSampler', 'TextGenerate', 'PreviewAny')}
    info.update({
        'UNETLoader': {'input': {'required': {'unet_name': [[]]}}},
        'CLIPLoader': {'input': {'required': {'clip_name': [[]], 'type': [['flux3']]}}},
        'VAELoader': {'input': {'required': {'vae_name': [['flux2-vae.safetensors']]}}},
        '_hfr_cloud_catalogue': [
            {'path': engines.ENGINES['flux3']['unet'][0], 'tags': ['diffusion_models']},
            {'path': engines.ENGINES['flux3']['clip'][0], 'tags': ['text_encoders']},
            {'path': engines.PUP_FILE, 'tags': ['text_encoders']},
        ]})
    return info


class StillEngines(unittest.TestCase):
    def test_cloud_catalogue_fills_missing_enum_without_fabricating_files(self):
        info = cloud_info()
        engine, files = engines.resolve('auto', info, 'Comfy Cloud')
        self.assertEqual(engine, 'flux3')
        self.assertEqual(files['unet'], 'flux3-image-klein-int8-convrot.safetensors')
        info.pop('_hfr_cloud_catalogue')
        with self.assertRaises(ValueError): engines.resolve('flux3', info, 'Comfy Cloud')

    def test_catalogue_does_not_hide_missing_core_support(self):
        info = cloud_info(); info['CLIPLoader']['input']['required']['type'] = [['flux2']]
        with self.assertRaises(ValueError): engines.resolve('flux3', info, 'Comfy Cloud')

    def test_caption_routes_to_conditioning_and_keeps_user_request(self):
        with patch.object(engines, 'prompt_system', return_value='BFL instructions plus rig rules'):
            graph, _ = engines.graph('far cat, near glass', 701, 2768, 1520, 'auto', cloud_info(), 'Comfy Cloud')
        self.assertEqual(graph['102']['inputs']['prompt'], 'far cat, near glass')
        self.assertEqual(graph['4']['inputs']['text'], ['103', 0])
        self.assertEqual(graph['103']['inputs']['source'], ['102', 0])
        self.assertEqual(graph['11']['inputs']['seed'], 701)
        self.assertEqual(graph['11']['inputs']['steps'], 16)

    def test_verbatim_mode_needs_neither_caption_model_nor_network(self):
        info = cloud_info(); info['_hfr_cloud_catalogue'].pop(); info.pop('TextGenerate')
        with patch.object(engines, 'prompt_system', side_effect=AssertionError('network not expected')):
            graph, _ = engines.graph('exact caption', 701, 1280, 704, 'flux3', info, 'Comfy Cloud', expand_prompt=False)
        self.assertEqual(graph['4']['inputs']['text'], 'exact caption')
        self.assertNotIn('102', graph)

    def test_canvas_and_analysis_use_same_generated_image(self):
        self.assertEqual(engines.canvas('2K'), (2768, 1520))
        self.assertEqual(engines.canvas('1K'), (1376, 752))
        with patch.object(engines, 'prompt_system', return_value='instructions'):
            graph, _ = engines.graph('scene', 701, 2768, 1520, 'flux3', cloud_info(), 'Comfy Cloud', analysis=True)
        self.assertEqual(graph['15']['inputs']['image'], ['12', 0])
        self.assertEqual(graph['23']['inputs']['image'], ['12', 0])


if __name__ == '__main__': unittest.main()
