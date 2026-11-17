"""The results table can show All, Fight only, or Don't Fight only.

The filter keeps the row order it was given. It does not re-score disputes
or call OpenAI. The batch summary stays on the full batch; the download
is the rows in the table.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import io
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
from streamlit.testing.v1 import AppTest

import app

ROOT = Path(__file__).resolve().parents[1]

CSV_MIXED = (
    "transaction_id,product,amount,reason_code\n"
    "T-fight,Alpha,20.00,duplicate\n"
    "T-skip,Beta,100.00,unauthorized\n"
).encode()

ARGUMENT_WARNING = (
    "Argument generation could not be completed, so a fallback was used. "
    "Something went wrong while calling OpenAI. Try again."
)


class FilterResultsTests(unittest.TestCase):
    def test_all_three_options_keep_given_order(self):
        frame = _frame(
            [
                ("skip-first", "Don't Fight", 50.0),
                ("fight-low", "Fight", 1.0),
                ("review", "Review", 999.0),
                ("fight-high", "Fight", 9.0),
                ("skip-second", "Don't Fight", 4.0),
                ("lower", "fight", 8.0),
                ("blank", "", 2.0),
                ("missing", None, 3.0),
            ]
        )
        everything = app.filter_results_by_recommendation(frame, app.RESULTS_FILTER_ALL)
        self.assertEqual(list(everything["Transaction_ID"]), list(frame["Transaction_ID"]))
        self.assertEqual(list(everything["Recommendation"]), list(frame["Recommendation"]))

        # The helper does not sort. fight-low stays ahead of fight-high.
        fight = app.filter_results_by_recommendation(frame, app.RESULTS_FILTER_FIGHT)
        self.assertEqual(list(fight["Transaction_ID"]), ["fight-low", "fight-high"])
        self.assertEqual(list(fight["Recommendation"]), ["Fight", "Fight"])

        skip = app.filter_results_by_recommendation(frame, app.RESULTS_FILTER_DONT_FIGHT)
        self.assertEqual(list(skip["Transaction_ID"]), ["skip-first", "skip-second"])
        self.assertEqual(list(skip["Recommendation"]), ["Don't Fight", "Don't Fight"])

    def test_sorted_preview_stays_fight_first_after_filtering(self):
        # Don't Fight has the larger recovery. Sort puts Fight first, then
        # the filter must not shuffle the remaining rows.
        preview = app.results_preview(
            _frame(
                [
                    ("skip-high", "Don't Fight", 40.0),
                    ("fight-low", "Fight", 5.0),
                    ("other", "Review", 999.0),
                    ("fight-high", "Fight", 80.0),
                    ("skip-low", "Don't Fight", 1.0),
                ]
            )
        )
        self.assertEqual(
            list(preview["Transaction_ID"]),
            ["fight-high", "fight-low", "skip-high", "skip-low", "other"],
        )
        fight = app.filter_results_by_recommendation(preview, app.RESULTS_FILTER_FIGHT)
        skip = app.filter_results_by_recommendation(
            preview, app.RESULTS_FILTER_DONT_FIGHT
        )
        self.assertEqual(list(fight["Transaction_ID"]), ["fight-high", "fight-low"])
        self.assertEqual(list(skip["Transaction_ID"]), ["skip-high", "skip-low"])
        self.assertEqual(
            [float(value) for value in fight["Potential_Recovery"]],
            [80.0, 5.0],
        )

        parsed = pd.read_csv(io.BytesIO(app.dataframe_to_csv_bytes(fight)))
        self.assertEqual(list(parsed["Transaction_ID"]), ["fight-high", "fight-low"])
        self.assertNotIn("skip-high", app.dataframe_to_csv_bytes(fight).decode("utf-8"))

    def test_unknown_choice_returns_every_row(self):
        frame = _frame(
            [
                ("fight", "Fight", 3.0),
                ("skip", "Don't Fight", 2.0),
                ("review", "Review", 8.0),
            ]
        )
        for choice in ("Maybe", "", None, "Fight", "Don't Fight", "ALL", ["Fight only"]):
            with self.subTest(choice=choice):
                shown = app.filter_results_by_recommendation(frame, choice)
                self.assertEqual(list(shown["Transaction_ID"]), ["fight", "skip", "review"])
                self.assertEqual(app.results_download_label(choice), "Download all results as CSV")
                self.assertIn("every row", app.results_download_help(choice))

    def test_empty_frame_keeps_its_columns_for_every_choice(self):
        empty = pd.DataFrame(columns=list(app.RESULTS_PREVIEW_COLUMNS))
        for choice in (*app.RESULTS_FILTER_OPTIONS, "nope", None):
            with self.subTest(choice=choice):
                shown = app.filter_results_by_recommendation(empty, choice)
                self.assertTrue(shown.empty)
                self.assertEqual(list(shown.columns), list(app.RESULTS_PREVIEW_COLUMNS))
                self.assertEqual(list(shown.index), [])

    def test_narrow_filter_can_match_nothing(self):
        only_fight = _frame([("a", "Fight", 1.0)])
        shown = app.filter_results_by_recommendation(
            only_fight, app.RESULTS_FILTER_DONT_FIGHT
        )
        self.assertTrue(shown.empty)
        self.assertEqual(list(shown.columns), list(only_fight.columns))

    def test_does_not_modify_the_input_or_keep_its_index(self):
        frame = _frame(
            [
                ("b", "Don't Fight", 2.0),
                ("a", "Fight", 1.0),
            ]
        )
        frame.index = [4, 9]
        original = frame.copy()
        shown = app.filter_results_by_recommendation(frame, app.RESULTS_FILTER_ALL)
        shown.loc[0, "Potential_Recovery"] = -1
        pd.testing.assert_frame_equal(frame, original)
        self.assertIsNot(shown, frame)
        self.assertEqual(list(shown.index), [0, 1])

    def test_download_label_names_the_rows(self):
        self.assertEqual(
            app.results_download_label(app.RESULTS_FILTER_ALL),
            "Download all results as CSV",
        )
        self.assertEqual(
            app.results_download_label(app.RESULTS_FILTER_FIGHT),
            "Download Fight results as CSV",
        )
        self.assertEqual(
            app.results_download_label(app.RESULTS_FILTER_DONT_FIGHT),
            "Download Don't Fight results as CSV",
        )
        fight_help = app.results_download_help(app.RESULTS_FILTER_FIGHT)
        self.assertIn("the Fight rows", fight_help)
        self.assertIn(app.RESULTS_CSV_FILENAME, fight_help)
        self.assertIn("same columns and row order", fight_help)
        self.assertIn("does not call OpenAI", fight_help)
        self.assertIn(
            "the Don't Fight rows",
            app.results_download_help(app.RESULTS_FILTER_DONT_FIGHT),
        )

    def test_does_not_call_openai(self):
        frame = _frame([("a", "Fight", 1.0), ("b", "Don't Fight", 2.0)])
        with mock.patch("openai.OpenAI", side_effect=AssertionError("OpenAI")):
            shown = app.filter_results_by_recommendation(frame, app.RESULTS_FILTER_FIGHT)
        self.assertEqual(list(shown["Transaction_ID"]), ["a"])


class ResultsFilterAppTests(unittest.TestCase):
    def setUp(self):
        self._saved_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key-not-used"
        _MixedClient.ranking_calls = 0
        _MixedClient.argument_calls = 0

    def tearDown(self):
        if self._saved_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self._saved_key

    def test_filter_keeps_the_batch_and_does_not_call_openai_again(self):
        fight = app.estimate_dispute_outcome(20.0, "duplicate", ["Shipping", "Receipt"])
        skip = app.estimate_dispute_outcome(100.0, "unauthorized", ["Email"])
        self.assertEqual(fight["recommendation"], "Fight")
        self.assertEqual(skip["recommendation"], "Don't Fight")
        self.assertLess(fight["potential_recovery"], skip["potential_recovery"])

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0, list(at.exception))
        self.assertEqual(list(at.radio), [])

        at.number_input[0].set_value(2)
        at.file_uploader[0].upload("merchant.csv", CSV_MIXED, "text/csv")
        with mock.patch("openai.OpenAI", _MixedClient), mock.patch(
            "time.sleep", return_value=None
        ):
            at.button[0].click().run()
            self._assert_ran(at)
            self.assertEqual(_MixedClient.ranking_calls, 1)
            self.assertEqual(_MixedClient.argument_calls, 1)

            full = _preview(at)
            summary = app.results_summary(full)
            self.assertEqual(list(full["Transaction_ID"]), ["T-fight", "T-skip"])
            self.assertEqual(list(full["Recommendation"]), ["Fight", "Don't Fight"])
            self.assertEqual(full.loc[0, "Potential_Recovery"], fight["potential_recovery"])
            self.assertEqual(full.loc[1, "Potential_Recovery"], skip["potential_recovery"])
            self.assertEqual(summary["fight"], 1)
            self.assertEqual(summary["dont_fight"], 1)
            _assert_summary(self, at, summary)
            self._assert_filter(at, app.RESULTS_FILTER_ALL, "Download all results as CSV")
            self.assertIn("every row", _results_download(at).help)
            self.assertEqual([warning.value for warning in at.warning], [ARGUMENT_WARNING])
            self.assertTrue(any("CSV transformed" in item.value for item in at.success))
            self.assertTrue(any("API Usage" in item.value for item in at.info))
            self.assertTrue(any("Showing 2 of 2." in caption.value for caption in at.caption))

            at.radio[0].set_value(app.RESULTS_FILTER_FIGHT).run()
            self._assert_ran(at)
            self.assertEqual(_MixedClient.ranking_calls, 1)
            self.assertEqual(_MixedClient.argument_calls, 1)
            shown = _preview(at)
            self.assertEqual(list(shown["Transaction_ID"]), ["T-fight"])
            self.assertEqual(list(shown["Recommendation"]), ["Fight"])
            self.assertEqual(shown.loc[0, "Simulated_Win_Rate"], fight["win_rate"])
            _assert_summary(self, at, summary)
            self.assertNotEqual(app.results_summary(shown)["dont_fight"], summary["dont_fight"])
            self._assert_filter(at, app.RESULTS_FILTER_FIGHT, "Download Fight results as CSV")
            self.assertIn("the Fight rows", _results_download(at).help)
            self.assertEqual([warning.value for warning in at.warning], [ARGUMENT_WARNING])
            self.assertTrue(any("CSV transformed" in item.value for item in at.success))
            self.assertTrue(any("Showing 1 of 2." in caption.value for caption in at.caption))
            self.assertNotIn("offline argument failure", shown.to_csv(index=False))

            at.radio[0].set_value(app.RESULTS_FILTER_DONT_FIGHT).run()
            self._assert_ran(at)
            self.assertEqual(_MixedClient.ranking_calls, 1)
            self.assertEqual(_MixedClient.argument_calls, 1)
            shown = _preview(at)
            self.assertEqual(list(shown["Transaction_ID"]), ["T-skip"])
            self.assertEqual(shown.loc[0, "Potential_Recovery"], skip["potential_recovery"])
            _assert_summary(self, at, summary)
            self._assert_filter(
                at,
                app.RESULTS_FILTER_DONT_FIGHT,
                "Download Don't Fight results as CSV",
            )
            self.assertTrue(any("Showing 1 of 2." in caption.value for caption in at.caption))
            self.assertEqual([warning.value for warning in at.warning], [ARGUMENT_WARNING])

            # A new batch starts on All again. Process runs the pipeline once
            # more: the failed argument call is not cached, so it is tried
            # again. The ranking prompt is unchanged and is served from the
            # existing cache, which is why that count stays at 1.
            at.button[0].click().run()
            self._assert_ran(at)
            self.assertEqual(_MixedClient.ranking_calls, 1)
            self.assertEqual(_MixedClient.argument_calls, 2)
            self.assertEqual(at.radio[0].value, app.RESULTS_FILTER_ALL)
            self.assertEqual(at.radio[0].value, app.RESULTS_FILTER_ALL)
            self.assertEqual(list(_preview(at)["Transaction_ID"]), ["T-fight", "T-skip"])
            _assert_summary(self, at, summary)
            self.assertEqual(_results_download(at).label, "Download all results as CSV")

    def _assert_ran(self, at):
        self.assertEqual(len(at.exception), 0, list(at.exception))

    def _assert_filter(self, at, choice, label):
        self.assertEqual(len(at.radio), 1)
        radio = at.radio[0]
        self.assertEqual(radio.label, "Show")
        self.assertEqual(radio.options, list(app.RESULTS_FILTER_OPTIONS))
        self.assertEqual(radio.value, choice)
        self.assertTrue(radio.horizontal)
        self.assertEqual(radio.key, "results_filter")
        self.assertIn("does not call OpenAI", radio.help)
        self.assertIn("batch summary", radio.help)
        download = _results_download(at)
        self.assertEqual(download.label, label)
        self.assertIn(app.RESULTS_CSV_FILENAME, download.help)
        self.assertIn("same columns and row order", download.help)
        self.assertIn("does not call OpenAI", download.help)
        self.assertTrue(download.proto.ignore_rerun)
        headings = [heading.value for heading in at.subheader]
        self.assertLess(headings.index("Batch summary"), headings.index("Preview"))


def _assert_summary(test, at, summary):
    metrics = {metric.label: metric.value for metric in at.metric}
    test.assertEqual(metrics["Fight"], f"{summary['fight']:,}")
    test.assertEqual(metrics["Don't Fight"], f"{summary['dont_fight']:,}")
    test.assertEqual(
        metrics["Potential recovery"],
        f"${summary['total_potential_recovery']:,.2f}",
    )


def _preview(at):
    return next(
        element.value
        for element in at.dataframe
        if "Recommendation" in getattr(element.value, "columns", [])
    )


def _results_download(at):
    return next(
        button
        for button in at.download_button
        if button.label != "Download sample CSV"
    )


def _frame(rows):
    return pd.DataFrame(
        [
            {
                "Transaction_ID": txn,
                "Reason_Code": "Fraud",
                "Recommendation": recommendation,
                "Top_Evidence": "Shipping",
                "AI_Argument": f"Argue {txn}",
                "Simulated_Win_Rate": 0.5,
                "Potential_Recovery": recovery,
            }
            for txn, recommendation, recovery in rows
        ]
    )


def _completion(content, tokens=7):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(total_tokens=tokens),
    )


class _MixedClient:
    """One Fight row and one Don't Fight row. The argument call fails closed.

    Nothing is sent over the network. Ranking uses two labels for the first
    dispute and one label for the second.
    """

    ranking_calls = 0
    argument_calls = 0

    def __init__(self, *args, **kwargs):
        pass

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, *args, **kwargs):
        content = kwargs["messages"][-1]["content"]
        if "evidence_ranking" in content:
            type(self).ranking_calls += 1
            payload = {
                "rankings": [
                    {"dispute_id": 1, "evidence_ranking": ["Shipping", "Receipt"]},
                    {"dispute_id": 2, "evidence_ranking": ["Email"]},
                ]
            }
            return _completion(json.dumps(payload))
        type(self).argument_calls += 1
        raise RuntimeError("offline argument failure")
