"""Exercise the live Gradio queue and the actual local HTTP provider path."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import unittest

from gradio_client import Client

from app import build_ui
from agent_app.config import DEFAULT_MODEL, Settings
from tests.fake_model import FakeModel


@contextmanager
def fixture_server():
    fake = FakeModel()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, status, body):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/v1/models":
                self.send_json(200, {"data": [{"id": DEFAULT_MODEL}]})
            else:
                self.send_json(404, {"error": "unexpected test URL"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": "unexpected test URL"})
                return
            status, response = fake.respond(body)
            self.send_json(status, response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", fake
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


class GradioIntegrationTests(unittest.TestCase):
    def test_chat_tool_follow_up_and_separate_browser_session(self):
        with fixture_server() as (base_url, fake):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            ui = build_ui(Settings(base_url=base_url, gradio_port=port, timeout_seconds=15))
            try:
                _, local_url, _ = ui.launch(
                    server_name="127.0.0.1", server_port=port, share=False,
                    prevent_thread_lock=True, quiet=True, show_error=True,
                )
                client = Client(local_url, verbose=False,
                                httpx_kwargs={"timeout": 20.0, "trust_env": False})
                status = client.predict(api_name="/check_server")
                self.assertIn("configured model found", str(status))
                first = client.predict("CLOCK", [], True, api_name="/chat")
                self.assertIn("local_system_clock", json.dumps(first))
                self.assertIn("tool_result", json.dumps(first))
                second = client.predict("HISTORY", first[0], True, api_name="/chat")
                self.assertIn("Earlier user messages: CLOCK", json.dumps(second))
                other_client = Client(local_url, verbose=False,
                                      httpx_kwargs={"timeout": 20.0, "trust_env": False})
                separate = other_client.predict("HISTORY", [], True, api_name="/chat")
                self.assertNotIn("CLOCK", json.dumps(separate))
                self.assertTrue(any(m["role"] == "tool" for m in fake.requests[2]["messages"]))
            finally:
                ui.close()


if __name__ == "__main__":
    unittest.main()
