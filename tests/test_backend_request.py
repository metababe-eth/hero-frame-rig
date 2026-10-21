"""Exercise Gradio's real special-argument injection, not a manual wrapper call."""
import unittest
from types import SimpleNamespace
import gradio as gr
from gradio.helpers import special_args
import backend

class RequestInjection(unittest.TestCase):
    def request(self, session='test', key='test-only-key'):
        return gr.Request(headers={'x-comfy-api-key':key,'host':'example.test'},session_hash=session,client=SimpleNamespace(host='203.0.113.1'))

    def test_bound_method_receives_header_without_exposing_input(self):
        class API:
            @backend.bind
            def call(self, value: str='value'):
                return value,backend.config()['key']
        call=API().call
        args,*_=special_args(call,['hello'],request=self.request())
        self.assertEqual(call(*args),('hello','test-only-key'))
        self.assertNotEqual(backend.config()['key'],'test-only-key')

    def test_generator_gets_session_and_context_is_reset(self):
        backend._sessions['test']=({'kind':backend.LOCAL,'url':'http://example.test:8188','key':''},0)
        @backend.bind
        def stream(value='hello'):
            yield value,backend.config()['kind']
            yield backend.config()['key']
        try:
            args,*_=special_args(stream,['input'],request=self.request())
            values=list(stream(*args))
            self.assertEqual(values,[('input',backend.LOCAL),'test-only-key'])
            self.assertNotEqual(backend.config()['key'],'test-only-key')
        finally:backend._sessions.pop('test',None)

    def test_variadic_ui_handler_keeps_all_inputs(self):
        @backend.bind
        def call(*values):return values,backend.config()['key']
        args,*_=special_args(call,['a','b'],request=self.request())
        self.assertEqual(call(*args),(('a','b'),'test-only-key'))

if __name__=='__main__':unittest.main()
