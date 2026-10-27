"""Hosted callers cannot inherit an owner credential, even on private Spaces."""
import os
import unittest
from unittest.mock import patch
import backend


class HostedByok(unittest.TestCase):
    def test_hosted_without_session_key_refuses_before_network(self):
        with patch.dict(os.environ, {'SPACE_ID': 'test/private-release', 'COMFY_API_KEY': 'comfyui-owner-fixture'}, clear=True), patch.object(backend.httpx, 'get') as network:
            with self.assertRaisesRegex(ValueError, 'your own Comfy API key'):
                backend.cloud_key({'key': '', 'loopback': True})
            network.assert_not_called()

    def test_hosted_uses_only_the_supplied_session_key(self):
        with patch.dict(os.environ, {'SPACE_ID': 'test/private-release', 'COMFY_API_KEY': 'comfyui-owner-fixture'}, clear=True):
            self.assertEqual(backend.cloud_key({'key': 'comfyui-session-fixture'}), 'comfyui-session-fixture')

    def test_desktop_environment_key_requires_loopback(self):
        with patch.dict(os.environ, {'COMFY_API_KEY': 'comfyui-local-fixture'}, clear=True):
            self.assertEqual(backend.cloud_key({'key': '', 'loopback': True}), 'comfyui-local-fixture')
            with self.assertRaisesRegex(ValueError, 'your own Comfy API key'):
                backend.cloud_key({'key': '', 'loopback': False})
