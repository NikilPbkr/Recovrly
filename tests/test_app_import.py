"""app.py should import, and the UI should load, with no OpenAI API key.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

import app

ROOT = Path(__file__).resolve().parents[1]


def _run_isolated(script: str) -> subprocess.CompletedProcess:
    """Import app in a fresh interpreter with credential env vars removed."""
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("OPENAI_ADMIN_KEY", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class AppImportTests(unittest.TestCase):
    def test_module_imports_without_an_api_key(self):
        result = _run_isolated(
            """
            import os
            assert "OPENAI_API_KEY" not in os.environ
            assert "OPENAI_ADMIN_KEY" not in os.environ
            import app
            assert app._openai_client is None
            print("import-ok")
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("import-ok", result.stdout)

    def test_client_is_created_only_when_an_api_call_needs_it(self):
        result = _run_isolated(
            """
            import os
            os.environ["OPENAI_API_KEY"] = "test-key-not-used"
            import app
            from openai import OpenAI
            assert app._openai_client is None
            client = app.get_openai_client()
            assert isinstance(client, OpenAI)
            assert client.api_key == "test-key-not-used"
            assert app.get_openai_client() is client
            print("lazy-ok")
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("lazy-ok", result.stdout)

    def test_missing_key_shows_a_message_instead_of_crashing(self):
        saved_key = os.environ.pop("OPENAI_API_KEY", None)
        saved_admin = os.environ.pop("OPENAI_ADMIN_KEY", None)
        try:
            at = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(at.button[0].label, "Process")
            self.assertIn(app.MISSING_API_KEY_MESSAGE, [w.value for w in at.warning])

            at.button[0].click().run()
            self.assertEqual(len(at.exception), 0)
            self.assertIn(app.MISSING_API_KEY_MESSAGE, [e.value for e in at.error])
        finally:
            _restore("OPENAI_API_KEY", saved_key)
            _restore("OPENAI_ADMIN_KEY", saved_admin)

    def test_process_with_a_key_still_requires_a_csv(self):
        saved_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key-not-used"
        try:
            at = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
            self.assertEqual(len(at.exception), 0)
            self.assertNotIn(app.MISSING_API_KEY_MESSAGE, [w.value for w in at.warning])

            at.button[0].click().run()
            self.assertEqual(len(at.exception), 0)
            self.assertIn("Please upload a CSV file.", [e.value for e in at.error])
            self.assertNotIn(app.MISSING_API_KEY_MESSAGE, [e.value for e in at.error])
        finally:
            _restore("OPENAI_API_KEY", saved_key)


def _restore(name: str, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
