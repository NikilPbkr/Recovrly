"""A failed OpenAI call should explain what happened, not dump the raw error.

The model call still falls back, and a successful call is unchanged.
No test talks to the network.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import json
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)
from streamlit.testing.v1 import AppTest

import app

ROOT = Path(__file__).resolve().parents[1]
SECRET = "sk-recovrly-test-secret"

RANKING = (
    "Evidence ranking could not be completed, so a fallback was used. "
)
ARGUMENTS = (
    "Argument generation could not be completed, so a fallback was used. "
)
PROCESSING = (
    "Dispute processing could not be completed, so a fallback was used. "
)

CSV_ONE = (
    "transaction_id,amount,reason_code\n"
    "T1,100.00,Fraud\n"
).encode()
CSV_FOUR = (
    "transaction_id,amount,reason_code\n"
    "T1,100.00,Fraud\n"
    "T2,20.00,duplicate\n"
    "T3,15.00,Product Not Received\n"
    "T4,5.00,canceled\n"
).encode()


def _openai_error(cls, text, status_code=None):
    """Build an OpenAI exception without a real HTTP request."""
    err = cls.__new__(cls)
    BaseException.__init__(err, text)
    if status_code is not None:
        err.status_code = status_code
    return err


def _page_text(at) -> str:
    parts = []
    for name in (
        "warning",
        "error",
        "exception",
        "info",
        "success",
        "caption",
        "markdown",
        "text",
    ):
        for element in getattr(at, name):
            parts.append(str(getattr(element, "value", element)))
    for element in at.dataframe:
        value = element.value
        parts.append(value.to_csv(index=False) if hasattr(value, "to_csv") else str(value))
    return "\n".join(parts)


def _preview(at):
    return next(
        element.value
        for element in at.dataframe
        if "Recommendation" in getattr(element.value, "columns", [])
    )


class ApiFailureMessageTests(unittest.TestCase):
    def test_timeout_names_the_timeout_and_hides_the_exception(self):
        exc = _openai_error(APITimeoutError, f"Request timed out. {SECRET}")
        message = app.api_failure_message(exc, "Evidence ranking")
        self.assertEqual(
            message,
            RANKING + "The OpenAI request timed out. Try again, or process fewer disputes.",
        )
        self.assertNotIn(SECRET, message)
        self.assertNotIn("Traceback", message)

    def test_connection_error_is_not_described_as_a_timeout(self):
        exc = _openai_error(APIConnectionError, f"Connection error. {SECRET}")
        message = app.api_failure_message(exc, "Evidence ranking")
        self.assertEqual(
            message,
            RANKING + "The app could not reach OpenAI. Check your network connection and try again.",
        )
        self.assertNotIn("timed out", message)
        self.assertNotIn(SECRET, message)

    def test_rejected_key_tells_the_user_where_to_look(self):
        for cls in (AuthenticationError, PermissionDeniedError):
            with self.subTest(cls=cls.__name__):
                exc = _openai_error(cls, f"Incorrect API key provided: {SECRET}")
                message = app.api_failure_message(exc, "Argument generation")
                self.assertEqual(
                    message,
                    ARGUMENTS
                    + "OpenAI rejected the API key. "
                    "Check OPENAI_API_KEY in your .env file or shell, then try again.",
                )
                self.assertNotIn(SECRET, message)

    def test_rate_limit_asks_the_user_to_wait(self):
        exc = _openai_error(RateLimitError, f"Rate limit reached for {SECRET}")
        message = app.api_failure_message(exc, "Evidence ranking")
        self.assertEqual(
            message,
            RANKING + "OpenAI rate-limited the request. Wait a moment and try again.",
        )
        self.assertNotIn(SECRET, message)

    def test_unreadable_model_output_does_not_include_the_parser_error(self):
        cases = [
            json.JSONDecodeError(f"leaked {SECRET}", "doc", 0),
            KeyError(SECRET),
            TypeError(SECRET),
            IndexError(SECRET),
            AttributeError(SECRET),
        ]
        for exc in cases:
            with self.subTest(exc=type(exc).__name__):
                message = app.api_failure_message(exc, "Evidence ranking")
                self.assertEqual(
                    message,
                    RANKING + "OpenAI returned a response this app could not read. Try again.",
                )
                self.assertNotIn(SECRET, message)
                self.assertNotIn("Expecting value", message)

    def test_http_status_includes_the_code_and_not_the_body(self):
        exc = _openai_error(
            InternalServerError,
            f"upstream exploded {SECRET}",
            status_code=503,
        )
        message = app.api_failure_message(exc, "Evidence ranking")
        self.assertEqual(
            message,
            RANKING + "OpenAI returned an error (HTTP 503). Try again in a moment.",
        )
        self.assertNotIn(SECRET, message)

        plain = _openai_error(APIStatusError, f"no code {SECRET}")
        self.assertEqual(
            app.api_failure_message(plain, "Evidence ranking"),
            RANKING + "OpenAI returned an error. Try again in a moment.",
        )
        self.assertNotIn(SECRET, app.api_failure_message(plain, "Evidence ranking"))

    def test_unknown_failure_stays_generic(self):
        exc = RuntimeError(f"Traceback (most recent call last):\n{SECRET}")
        message = app.api_failure_message(exc, "Dispute processing")
        self.assertEqual(
            message,
            PROCESSING + "Something went wrong while calling OpenAI. Try again.",
        )
        self.assertNotIn("Traceback", message)
        self.assertNotIn(SECRET, message)


class _BaseClient:
    def __init__(self, *args, **kwargs):
        pass

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self


class _FailingClient(_BaseClient):
    """Every model call raises. Nothing is sent over the network."""

    error = None

    def create(self, *args, **kwargs):
        raise self.error


class _TimeoutClient(_FailingClient):
    error = _openai_error(APITimeoutError, f"Request timed out. {SECRET}")


class _AuthClient(_FailingClient):
    error = _openai_error(
        AuthenticationError,
        f"Incorrect API key provided: {SECRET}. You can find your API key at https://platform.openai.com/account/api-keys.",
    )


class _BadJsonClient(_BaseClient):
    def create(self, *args, **kwargs):
        return _completion("{not-json")


class _SuccessClient(_BaseClient):
    def create(self, *args, **kwargs):
        content = kwargs["messages"][-1]["content"]
        if "evidence_ranking" in content:
            payload = {
                "rankings": [
                    {
                        "dispute_id": 1,
                        "evidence_ranking": ["Receipt", "Email", "Shipping", "Photo"],
                    }
                ]
            }
            return _completion(json.dumps(payload), tokens=7)
        payload = {
            "arguments": [
                {"dispute_id": 1, "argument": "The customer received the item on time."}
            ]
        }
        return _completion(json.dumps(payload), tokens=11)


class _RankingOnlyClient(_BaseClient):
    """Ranking succeeds. Writing the argument times out."""

    def create(self, *args, **kwargs):
        content = kwargs["messages"][-1]["content"]
        if "evidence_ranking" not in content:
            raise _openai_error(APITimeoutError, f"Request timed out. {SECRET}")
        payload = {
            "rankings": [
                {
                    "dispute_id": 1,
                    "evidence_ranking": ["Receipt", "Email", "Shipping", "Photo"],
                }
            ]
        }
        return _completion(json.dumps(payload), tokens=4)


def _completion(content, tokens=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(total_tokens=tokens),
    )


class ApiFailureAppTests(unittest.TestCase):
    def setUp(self):
        self._saved_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key-not-used"

    def tearDown(self):
        if self._saved_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self._saved_key

    def test_timeout_warns_once_and_still_shows_fallback_results(self):
        at = self._process(CSV_FOUR, 4, _TimeoutClient)
        self.assertEqual(len(at.exception), 0, list(at.exception))
        self.assertEqual(
            [warning.value for warning in at.warning],
            [
                RANKING + "The OpenAI request timed out. Try again, or process fewer disputes.",
                ARGUMENTS + "The OpenAI request timed out. Try again, or process fewer disputes.",
            ],
        )
        self.assertNotIn(SECRET, _page_text(at))

        preview = _preview(at)
        self.assertEqual(list(preview["Top_Evidence"]), ["Shipping, Receipt"] * 4)
        self.assertEqual(
            list(preview["AI_Argument"]),
            ["Dispute argument generation failed."] * 4,
        )
        self.assertEqual(len(preview), 4)
        self.assertIn("API Usage", at.info[0].value)

        # A second click in the same session warns again.
        with mock.patch("openai.OpenAI", _TimeoutClient), mock.patch(
            "time.sleep", return_value=None
        ):
            at.button[0].click().run()
        self.assertEqual(len(at.exception), 0, list(at.exception))
        self.assertEqual(len(at.warning), 2)
        self.assertNotIn(SECRET, _page_text(at))

    def test_rejected_key_does_not_show_the_key_or_the_response_body(self):
        at = self._process(CSV_ONE, 1, _AuthClient)
        self.assertEqual(len(at.exception), 0, list(at.exception))
        self.assertEqual(
            [warning.value for warning in at.warning],
            [
                RANKING
                + "OpenAI rejected the API key. "
                "Check OPENAI_API_KEY in your .env file or shell, then try again.",
                ARGUMENTS
                + "OpenAI rejected the API key. "
                "Check OPENAI_API_KEY in your .env file or shell, then try again.",
            ],
        )
        page = _page_text(at)
        self.assertNotIn(SECRET, page)
        self.assertNotIn("platform.openai.com", page)
        self.assertNotIn("Incorrect API key", page)
        self.assertEqual(_preview(at).loc[0, "AI_Argument"], "Dispute argument generation failed.")

    def test_unreadable_response_names_the_problem(self):
        at = self._process(CSV_ONE, 1, _BadJsonClient)
        self.assertEqual(len(at.exception), 0, list(at.exception))
        self.assertEqual(
            [warning.value for warning in at.warning],
            [
                RANKING + "OpenAI returned a response this app could not read. Try again.",
                ARGUMENTS + "OpenAI returned a response this app could not read. Try again.",
            ],
        )
        self.assertNotIn("Expecting", _page_text(at))
        self.assertNotIn("{not-json", _page_text(at))

    def test_successful_call_is_unchanged(self):
        at = self._process(CSV_ONE, 1, _SuccessClient)
        self.assertEqual(len(at.exception), 0, list(at.exception))
        self.assertEqual(list(at.warning), [])
        preview = _preview(at)
        self.assertEqual(preview.loc[0, "Transaction_ID"], "T1")
        self.assertEqual(preview.loc[0, "Top_Evidence"], "Receipt, Email")
        self.assertEqual(
            preview.loc[0, "AI_Argument"],
            "The customer received the item on time.",
        )
        self.assertIn(preview.loc[0, "Recommendation"], ("Fight", "Don't Fight"))
        self.assertIn("18 tokens", at.info[0].value)
        self.assertNotIn("fallback", _page_text(at).lower())

    def test_ranking_is_kept_when_only_the_argument_call_fails(self):
        at = self._process(CSV_ONE, 1, _RankingOnlyClient)
        self.assertEqual(len(at.exception), 0, list(at.exception))
        self.assertEqual(
            [warning.value for warning in at.warning],
            [
                ARGUMENTS + "The OpenAI request timed out. Try again, or process fewer disputes.",
            ],
        )
        preview = _preview(at)
        self.assertEqual(preview.loc[0, "Top_Evidence"], "Receipt, Email")
        self.assertEqual(preview.loc[0, "AI_Argument"], "Dispute argument generation failed.")
        self.assertNotIn(SECRET, _page_text(at))

    def test_failure_outside_the_model_call_stays_plain(self):
        # The app pauses 0.5s between the two model calls. Streamlit's own
        # test runner also sleeps, so only that pause should fail.
        real_sleep = time.sleep

        def explode(seconds):
            if seconds == 0.5:
                raise _openai_error(APITimeoutError, f"Request timed out. {SECRET}")
            return real_sleep(seconds)

        at = self._process(CSV_ONE, 1, _SuccessClient, sleep=explode)
        self.assertEqual(len(at.exception), 0, list(at.exception))
        self.assertEqual(
            [warning.value for warning in at.warning],
            [
                PROCESSING + "The OpenAI request timed out. Try again, or process fewer disputes.",
            ],
        )
        preview = _preview(at)
        self.assertEqual(preview.loc[0, "Top_Evidence"], "Shipping, Receipt")
        self.assertEqual(
            preview.loc[0, "AI_Argument"],
            "Dispute argument could not be generated due to processing error.",
        )
        self.assertNotIn(SECRET, _page_text(at))
        self.assertNotIn("The customer received the item on time.", _page_text(at))

    def _process(self, csv_bytes, rows, client, sleep=None):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0, list(at.exception))
        at.number_input[0].set_value(rows)
        at.file_uploader[0].upload("disputes.csv", csv_bytes, "text/csv")
        if sleep is None:
            def sleep(_seconds):
                return None
        with mock.patch("openai.OpenAI", client), mock.patch("time.sleep", sleep):
            at.button[0].click().run()
        return at
