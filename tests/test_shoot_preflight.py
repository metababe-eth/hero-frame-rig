import unittest
from unittest.mock import Mock, patch
import gradio as gr
import agent_api


class ShootPreflight(unittest.TestCase):
    def test_missing_key_is_actionable_and_never_creates_a_client(self):
        client = Mock()
        api = agent_api.AgentAPI(client)
        with patch.object(agent_api.backend, 'object_info', side_effect=ValueError('Comfy Cloud needs your own Comfy API key.')):
            with self.assertRaisesRegex(gr.Error, 'own Comfy API key'):
                api.shoot_still('a scene')
        client.assert_not_called()


if __name__ == '__main__': unittest.main()
