"""Model rankings should become a Fight / Don't Fight decision.

The model does not choose Fight or Don't Fight. It returns an evidence
ranking; the first two labels are what the win-rate formula scores.
These tests stub the OpenAI client, so they need no API key and no network.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

import app


FALLBACK_EVIDENCE = "Shipping, Receipt"
# Unauthorized at $100, scored with the two fallback labels:
# 0.30 + 0.16 - 0.01 = 0.45, recovery 45.00, which is Fight.
FALLBACK_WIN_RATE = 0.45
FALLBACK_RECOVERY = 45.0


class ModelResponseDecisionTests(unittest.TestCase):
    def setUp(self):
        app.st.session_state.ai_cache = {}
        app.st.session_state.api_failure_notices = []
        app.st.session_state.total_tokens = 0

    def test_top_two_labels_set_the_recommendation(self):
        # Three disputes, two per API batch. Extra labels must not raise the
        # win rate, and an empty list must not be replaced with the fallback.
        frame = _frame([
            ("T-one", "Alpha", 100, "unauthorized"),
            ("T-four", "Beta", 100, "Fraud"),
            ("T-none", "Gamma", 100, "duplicate"),
        ])
        rankings = {
            "Alpha": ["Email"],
            "Beta": ["Photo", "Receipt", "Shipping", "Email"],
            "Gamma": [],
        }
        result, client = _process(frame, _by_product(rankings), batch_size=2)

        self.assertEqual(client.ranking_calls, 2)
        self.assertEqual(client.argument_calls, 2)
        _assert_decision(self, result, 0, "T-one", "Email", 0.37, 37.0, "Don't Fight")
        # Four labels would be 0.66. Only Photo and Receipt count: 0.50.
        _assert_decision(self, result, 1, "T-four", "Photo, Receipt", 0.50, 50.0, "Fight")
        _assert_decision(self, result, 2, "T-none", "", 0.69, 69.0, "Fight")
        self.assertEqual(
            list(result["AI_Argument"]),
            ["Argument for Alpha", "Argument for Beta", "Argument for Gamma"],
        )
        self.assertEqual(app.st.session_state.api_failure_notices, [])

        # The parsed lists are cached. A second run does not call the model,
        # and it still scores two labels rather than the original four.
        again, client = _process(frame, _by_product(rankings), batch_size=2)
        self.assertEqual(client.ranking_calls, 0)
        self.assertEqual(client.argument_calls, 0)
        self.assertEqual(list(again["Simulated_Win_Rate"]), [0.37, 0.50, 0.69])
        self.assertEqual(
            list(again["Recommendation"]),
            ["Don't Fight", "Fight", "Fight"],
        )

    def test_rankings_follow_response_order_and_extras_are_ignored(self):
        # dispute_id is not used to reorder. The third ranking is not a row.
        frame = _frame([
            ("T-first", "Alpha", 100, "unauthorized"),
            ("T-second", "Beta", 100, "duplicate"),
        ])
        payload = {
            "rankings": [
                {"dispute_id": 2, "evidence_ranking": ["Email"]},
                {"dispute_id": 1, "evidence_ranking": ["Photo", "Receipt", "Shipping"]},
                {"dispute_id": 3, "evidence_ranking": ["Ghost"]},
            ]
        }
        result, _client = _process(frame, _fixed(payload))

        _assert_decision(self, result, 0, "T-first", "Email", 0.37, 37.0, "Don't Fight")
        _assert_decision(self, result, 1, "T-second", "Photo, Receipt", 0.85, 85.0, "Fight")
        self.assertNotIn("Ghost", result.to_csv(index=False))
        self.assertEqual(len(result), 2)

    def test_short_ranking_list_is_padded_with_the_default_pair(self):
        frame = _frame([
            ("T-real", "Alpha", 100, "unauthorized"),
            ("T-missing", "Beta", 100, "unauthorized"),
        ])
        payload = {
            "rankings": [
                {"dispute_id": 1, "evidence_ranking": ["Photo"]},
            ]
        }
        result, _client = _process(frame, _fixed(payload))

        _assert_decision(self, result, 0, "T-real", "Photo", 0.37, 37.0, "Don't Fight")
        _assert_decision(
            self, result, 1, "T-missing", FALLBACK_EVIDENCE,
            FALLBACK_WIN_RATE, FALLBACK_RECOVERY, "Fight",
        )
        self.assertEqual(app.st.session_state.api_failure_notices, [])

    def test_string_ranking_is_not_scored_character_by_character(self):
        # A string used to be sliced. "S" counted as one piece of evidence
        # (Don't Fight, 0.37). "No" and "Shipping" counted as two characters
        # (Fight, 0.45) and the preview showed "N, o" or "S, h".
        frame = _frame([("T1", "Alpha", 100, "unauthorized")])
        for raw in ("S", "No", "Shipping", ""):
            with self.subTest(raw=raw):
                self.setUp()
                result, client = _process(
                    frame,
                    _fixed(
                        {"rankings": [{"dispute_id": 1, "evidence_ranking": raw}]},
                        arguments=["Kept the argument"],
                    ),
                )
                _assert_decision(
                    self, result, 0, "T1", FALLBACK_EVIDENCE,
                    FALLBACK_WIN_RATE, FALLBACK_RECOVERY, "Fight",
                )
                self.assertEqual(result.loc[0, "AI_Argument"], "Kept the argument")
                self.assertNotIn("S, h", result.loc[0, "Top_Evidence"])
                self.assertNotIn("N, o", result.loc[0, "Top_Evidence"])
                self.assertEqual(client.ranking_calls, 1)
                self.assertEqual(client.argument_calls, 1)
                notice = app.st.session_state.api_failure_notices
                self.assertEqual(len(notice), 1)
                self.assertIn("could not read", notice[0])
                self.assertIn("Evidence ranking", notice[0])
                self.assertNotIn("evidence ranking must be a list", notice[0])

        # One bad ranking makes the whole batch unreadable. The valid sibling
        # is not kept, and neither row is scored from sliced characters.
        pair = _frame([
            ("T-good", "Alpha", 100, "unauthorized"),
            ("T-bad", "Beta", 100, "unauthorized"),
        ])
        self.setUp()
        result, _client = _process(
            pair,
            _fixed(
                {"rankings": [
                    {"dispute_id": 1, "evidence_ranking": ["Email"]},
                    {"dispute_id": 2, "evidence_ranking": "Shipping"},
                ]},
                arguments=["Arg A", "Arg B"],
            ),
        )
        for index, txn in ((0, "T-good"), (1, "T-bad")):
            _assert_decision(
                self, result, index, txn, FALLBACK_EVIDENCE,
                FALLBACK_WIN_RATE, FALLBACK_RECOVERY, "Fight",
            )
        self.assertEqual(list(result["AI_Argument"]), ["Arg A", "Arg B"])
        self.assertNotIn("Email", result.to_csv(index=False))
        self.assertNotIn("S, h", result.to_csv(index=False))

    def test_unreadable_ranking_uses_the_fallback_pair(self):
        frame = _frame([("T1", "Alpha", 100, "unauthorized")])
        cases = {
            "invalid json": "{not-json",
            "missing rankings": {"results": []},
            "rankings object": {"rankings": {"dispute_id": 1, "evidence_ranking": ["Email"]}},
            "missing labels": {"rankings": [{"dispute_id": 1}]},
            "null ranking": {"rankings": [{"dispute_id": 1, "evidence_ranking": None}]},
        }
        for name, body in cases.items():
            with self.subTest(name=name):
                self.setUp()
                result, client = _process(
                    frame,
                    _fixed(body, arguments=["Still written"]),
                )
                _assert_decision(
                    self, result, 0, "T1", FALLBACK_EVIDENCE,
                    FALLBACK_WIN_RATE, FALLBACK_RECOVERY, "Fight",
                )
                self.assertEqual(result.loc[0, "AI_Argument"], "Still written")
                self.assertEqual(client.argument_calls, 1)
                self.assertIn("could not read", app.st.session_state.api_failure_notices[0])

    def test_malformed_arguments_keep_the_ranking_decision(self):
        # One label is Don't Fight. A bad argument payload must not replace
        # that ranking with the two-label fallback, which would flip to Fight.
        frame = _frame([
            ("T1", "Alpha", 100, "unauthorized"),
            ("T2", "Beta", 100, "unauthorized"),
        ])
        rankings = {
            "rankings": [
                {"dispute_id": 1, "evidence_ranking": ["Email"]},
                {"dispute_id": 2, "evidence_ranking": ["Email"]},
            ]
        }

        missing_key, _client = _process(
            frame,
            _fixed(rankings, argument_body={"arguments": [{"dispute_id": 1}]}),
        )
        _assert_decision(self, missing_key, 0, "T1", "Email", 0.37, 37.0, "Don't Fight")
        _assert_decision(self, missing_key, 1, "T2", "Email", 0.37, 37.0, "Don't Fight")
        self.assertEqual(
            list(missing_key["AI_Argument"]),
            ["Dispute argument generation failed.", "Dispute argument generation failed."],
        )

        self.setUp()
        short, _client = _process(
            frame,
            _fixed(rankings, arguments=["Only the first row"]),
        )
        _assert_decision(self, short, 0, "T1", "Email", 0.37, 37.0, "Don't Fight")
        _assert_decision(self, short, 1, "T2", "Email", 0.37, 37.0, "Don't Fight")
        self.assertEqual(
            list(short["AI_Argument"]),
            ["Only the first row", "Argument generation incomplete."],
        )

        self.setUp()
        extra, _client = _process(
            frame,
            _fixed(rankings, arguments=["First", "Second", "Not a dispute"]),
        )
        self.assertEqual(list(extra["AI_Argument"]), ["First", "Second"])
        self.assertNotIn("Not a dispute", extra.to_csv(index=False))
        self.assertEqual(list(extra["Recommendation"]), ["Don't Fight", "Don't Fight"])


def _assert_decision(test, result, index, txn, evidence, win_rate, recovery, recommendation):
    row = result.loc[index]
    test.assertEqual(row["Transaction_ID"], txn)
    test.assertEqual(row["Top_Evidence"], evidence)
    test.assertEqual(row["Simulated_Win_Rate"], win_rate)
    test.assertEqual(row["Potential_Recovery"], recovery)
    test.assertEqual(row["Recommendation"], recommendation)


def _frame(rows):
    records = []
    for txn, product, amount, reason in rows:
        records.append({
            "Transaction_ID": txn,
            "Product": product,
            "Amount": amount,
            "Reason_Code": reason,
            "Evidence_Receipt": "receipt",
            "Evidence_Shipping": "shipped",
            "Evidence_Email": "emailed",
            "Evidence_Photo": "photo",
        })
    return pd.DataFrame(records)


def _process(frame, respond, batch_size=2):
    client = _Client(respond)
    with mock.patch.object(app, "get_openai_client", return_value=client), \
            mock.patch.object(app.time, "sleep"):
        result = app.process_disputes(frame, len(frame), batch_size=batch_size)
    return result, client


def _by_product(rankings):
    """Return the ranking stored for each product named in that batch."""

    def respond(prompt):
        if _is_ranking_prompt(prompt):
            names = _transaction_products(prompt)
            return {
                "rankings": [
                    {"dispute_id": index + 1, "evidence_ranking": rankings[name]}
                    for index, name in enumerate(names)
                ]
            }
        names = _argument_products(prompt)
        return {
            "arguments": [
                {"dispute_id": index + 1, "argument": f"Argument for {name}"}
                for index, name in enumerate(names)
            ]
        }

    return respond


def _fixed(ranking_body, arguments=None, argument_body=None):
    def respond(prompt):
        if _is_ranking_prompt(prompt):
            return ranking_body
        if argument_body is not None:
            return argument_body
        texts = arguments or ["Argument"]
        return {
            "arguments": [
                {"dispute_id": index + 1, "argument": text}
                for index, text in enumerate(texts)
            ]
        }

    return respond


def _is_ranking_prompt(prompt):
    return "Rank evidence for each dispute" in prompt


def _transaction_products(prompt):
    names = []
    for line in prompt.splitlines():
        if line.startswith("Transaction: "):
            names.append(line.split("Transaction: ", 1)[1].split(",", 1)[0])
    return names


def _argument_products(prompt):
    names = []
    for line in prompt.splitlines():
        if line.startswith("Product: "):
            names.append(line.split("Product: ", 1)[1].split(",", 1)[0])
    return names


class _Client:
    """Records each prompt and returns scripted JSON. Nothing is sent."""

    def __init__(self, respond):
        self.respond = respond
        self.ranking_calls = 0
        self.argument_calls = 0

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, *args, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        if _is_ranking_prompt(prompt):
            self.ranking_calls += 1
        else:
            self.argument_calls += 1
        body = self.respond(prompt)
        content = body if isinstance(body, str) else json.dumps(body)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(total_tokens=3),
        )
