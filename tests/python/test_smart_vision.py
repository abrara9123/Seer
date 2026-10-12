import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import smart_vision
import chatgpt_auth


class SmartVisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image = Path(self.temp.name) / "snapshot.jpg"
        self.image.write_bytes(b"test-image-bytes")

    def analyze(self, response=None, error=None):
        with patch("smart_vision.urllib.request.urlopen") as request:
            if error:
                request.side_effect = error
            else:
                request.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            result = smart_vision.analyze_image_with_openai(str(self.image), "test-key", "gpt-5.6-luna")
            return result, request

    def test_image_request_and_spoken_response(self):
        result, request = self.analyze({
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "content": [
                    {"type": "output_text", "text": "**Extra gum**.\n"},
                    {"type": "output_text", "text": "It is green."}
                ]}
            ]
        })
        self.assertEqual(result, "Extra gum. It is green.")
        req = request.call_args.args[0]
        self.assertEqual(req.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.get_header("Authorization"), "Bearer test-key")
        self.assertNotIn("test-key", req.full_url)
        payload = json.loads(req.data)
        self.assertEqual(payload["model"], "gpt-5.6-luna")
        self.assertFalse(payload["store"])
        self.assertEqual(payload["reasoning"], {"effort": "none"})
        self.assertEqual(payload["max_output_tokens"], 160)
        image = payload["input"][0]["content"][0]
        self.assertEqual(image["type"], "input_image")
        self.assertEqual(image["detail"], "high")
        self.assertEqual(image["image_url"], "data:image/jpeg;base64," + base64.b64encode(self.image.read_bytes()).decode())

    def test_config_key_and_model(self):
        config = Path(self.temp.name) / "config.txt"
        config.write_text("# settings\nopenai_api_key = config-key\nopenai_model = gpt-5.5-2026-04-23\n")
        with patch.dict(os.environ, {}, clear=True), patch.object(smart_vision, "PRIVATE_CONFIG_PATH", config):
            settings = smart_vision.load_settings_from_config()
        self.assertEqual(settings, {"api_key": "config-key", "model": "gpt-5.5-2026-04-23", "auth_mode": "chatgpt", "chatgpt_model": ""})

    def test_environment_key_takes_priority(self):
        config = Path(self.temp.name) / "config.txt"
        config.write_text("openai_api_key=config-key\nopenai_model=\n")
        with patch.dict(os.environ, {"OPENAI_API_KEY": " env-key "}, clear=True), patch.object(smart_vision, "PRIVATE_CONFIG_PATH", config):
            settings = smart_vision.load_settings_from_config()
        self.assertEqual(settings, {"api_key": "env-key", "model": "gpt-5.6-luna", "auth_mode": "chatgpt", "chatgpt_model": ""})

    def test_private_key_file(self):
        config = Path(self.temp.name) / "config.txt"
        config.write_text("openai_api_key_file=.openai_api_key\n")
        (config.parent / ".openai_api_key").write_text(" file-key\n")
        with patch.dict(os.environ, {}, clear=True), patch.object(smart_vision, "PRIVATE_CONFIG_PATH", config):
            settings = smart_vision.load_settings_from_config()
        self.assertEqual(settings["api_key"], "file-key")

    def test_environment_key_takes_priority_over_file(self):
        config = Path(self.temp.name) / "config.txt"
        config.write_text("openai_api_key_file=.openai_api_key\n")
        (config.parent / ".openai_api_key").write_text("file-key")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True), patch.object(smart_vision, "PRIVATE_CONFIG_PATH", config):
            settings = smart_vision.load_settings_from_config()
        self.assertEqual(settings["api_key"], "env-key")

    def test_missing_key_file(self):
        config = Path(self.temp.name) / "config.txt"
        config.write_text("openai_api_key_file=missing-key-file\n")
        with patch.dict(os.environ, {}, clear=True), patch.object(smart_vision, "PRIVATE_CONFIG_PATH", config):
            settings = smart_vision.load_settings_from_config()
        self.assertEqual(settings["api_key"], "")

    def test_public_config_cannot_supply_credentials(self):
        root = Path(self.temp.name)
        (root / "config").mkdir()
        (root / "config/default.txt").write_text("openai_api_key=public-file-key\n")
        with patch.dict(os.environ, {}, clear=True), patch.object(smart_vision, "PRIVATE_CONFIG_PATH", root / ".local/openai.conf"):
            settings = smart_vision.load_settings_from_config()
        self.assertEqual(settings["api_key"], "")

    def test_relative_image_is_resolved_from_project_root(self):
        settings = {"api_key": "test-key", "model": smart_vision.DEFAULT_MODEL, "auth_mode": "api_key"}
        with patch.object(smart_vision, "PROJECT_ROOT", Path(self.temp.name)), patch.object(smart_vision, "load_settings_from_config", return_value=settings), patch.object(smart_vision.sys, "argv", ["smart_vision.py", "snapshot.jpg"]), patch.object(smart_vision, "analyze_image_with_openai", return_value="A chair.") as analyze, patch("sys.stdout", new_callable=io.StringIO):
            smart_vision.main()
        self.assertEqual(analyze.call_args.args[0], str(self.image))

    def test_missing_key_does_not_send_request(self):
        with patch.object(smart_vision, "load_settings_from_config", return_value={"api_key": "", "model": "gpt-5.6-luna"}), patch.object(smart_vision.sys, "argv", ["smart_vision.py", str(self.image)]), patch("smart_vision.urllib.request.urlopen") as request, patch("sys.stdout", new_callable=io.StringIO) as stdout:
            with self.assertRaises(SystemExit) as exit_result:
                smart_vision.main()
            self.assertEqual(exit_result.exception.code, 0)
            self.assertIn("OPENAI_API_KEY", stdout.getvalue())
            request.assert_not_called()

    def test_api_errors_are_spoken_without_exposing_credentials(self):
        for status, expected in [(401, "key is invalid"), (403, "permission"), (404, "gpt-5.6-luna is unavailable"), (429, "billing"), (500, "HTTP 500")]:
            with self.subTest(status=status):
                error = urllib.error.HTTPError(smart_vision.OPENAI_RESPONSES_URL, status, "Error", {}, io.BytesIO(b'test-key'))
                result, request = self.analyze(error=error)
                self.assertIn(expected, result)
                self.assertNotIn("test-key", result)
                self.assertEqual(request.call_count, 1)

    def test_connection_errors(self):
        for error in [TimeoutError(), urllib.error.URLError("offline"), ConnectionResetError()]:
            with self.subTest(error=type(error).__name__):
                result, _ = self.analyze(error=error)
                self.assertIn("Check your connection", result)

    def test_unusable_responses(self):
        for response, expected in [
            ({"status": "incomplete", "output": [{"type": "message", "content": [{"type": "output_text", "text": "Partial"}]}]}, "did not finish"),
            ({"status": "failed", "error": {"message": "failure"}}, "could not complete"),
            ({"output": []}, "no description"),
            ({"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "Cannot help"}]}]}, "could not describe"),
            (None, "invalid response")
        ]:
            with self.subTest(response=response):
                result, _ = self.analyze(response)
                self.assertIn(expected, result)

    def test_invalid_json(self):
        with patch("smart_vision.urllib.request.urlopen") as request:
            request.return_value.__enter__.return_value.read.return_value = b"not json"
            result = smart_vision.analyze_image_with_openai(str(self.image), "test-key", "gpt-5.6-luna")
        self.assertIn("invalid response", result)

    def test_missing_image_does_not_send_request(self):
        with patch("smart_vision.urllib.request.urlopen") as request:
            result = smart_vision.analyze_image_with_openai(str(self.image.parent / "missing.jpg"), "test-key", "gpt-5.6-luna")
        self.assertIn("could not be read", result)
        request.assert_not_called()

    @staticmethod
    def stream(*events):
        return io.BytesIO(b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events))

    def test_chatgpt_stream_success_and_request_contract(self):
        response = self.stream(
            {"type": "response.output_text.delta", "delta": "A **green** "},
            {"type": "response.output_text.delta", "delta": "bottle."},
            {"type": "response.completed", "response": {"status": "completed"}}
        )
        with patch("smart_vision.urllib.request.urlopen", return_value=response) as request:
            result = smart_vision.analyze_image_with_openai(str(self.image), "plan-token", "gpt-5.6-luna", use_chatgpt_plan=True)
        self.assertEqual(result, "A green bottle.")
        payload = json.loads(request.call_args.args[0].data)
        self.assertTrue(payload["stream"])
        self.assertFalse(payload["store"])
        self.assertNotIn("max_output_tokens", payload)
        self.assertEqual(payload["reasoning"], {"effort": "none"})

    def test_chatgpt_stream_requires_completion(self):
        partial = {"type": "response.output_text.delta", "delta": "Partial description"}
        cases = [
            ([], "ended before"),
            ([{"type": "response.failed", "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded"}}}], "usage limit"),
            ([{"type": "response.incomplete"}], "did not finish"),
            ([{"type": "response.refusal.delta", "delta": "refused"}], "could not describe"),
            ([{"type": "response.completed", "response": {"status": "failed"}}], "did not complete")
        ]
        for events, expected in cases:
            with self.subTest(events=events):
                result = smart_vision.stream_description(self.stream(partial, *events))
                self.assertIn(expected, result)
                self.assertNotIn("Partial description", result)

    def test_chatgpt_mode_never_falls_back_to_api_key(self):
        settings = {"auth_mode": "chatgpt", "api_key": "billable-api-key", "model": "gpt-5.6-luna"}
        with patch.object(smart_vision, "load_settings_from_config", return_value=settings), patch.object(smart_vision.sys, "argv", ["smart_vision.py", str(self.image)]), patch.object(chatgpt_auth, "access_token", side_effect=chatgpt_auth.AuthError("Press L to sign in")), patch.object(smart_vision, "analyze_image_with_openai") as analyze, patch("sys.stdout", new_callable=io.StringIO) as stdout:
            smart_vision.main()
        self.assertIn("Press L", stdout.getvalue())
        analyze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
