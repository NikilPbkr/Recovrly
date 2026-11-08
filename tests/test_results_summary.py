"""Processed results should summarize Fight, Don't Fight, and potential recovery.

The counts and the recovery total come from the rows already produced.
They do not change how a recommendation or a win rate is calculated.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import os
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from streamlit.testing.v1 import AppTest

import app

ROOT = Path(__file__).resolve().parents[1]


class ResultsSummaryTests(unittest.TestCase):
    def test_counts_each_recommendation_and_sums_recovery(self):
        summary = app.results_summary(
            _frame(
                [
                    ("Fight", 120.50),
                    ("Don't Fight", 15.25),
                    ("Fight", 40.00),
                    ("Don't Fight", 0.00),
                ]
            )
        )
        self.assertEqual(summary["fight"], 2)
        self.assertEqual(summary["dont_fight"], 2)
        self.assertEqual(summary["total_potential_recovery"], 175.75)

    def test_all_fight_leaves_dont_fight_at_zero(self):
        summary = app.results_summary(_frame([("Fight", 10.00), ("Fight", 2.50)]))
        self.assertEqual(summary["fight"], 2)
        self.assertEqual(summary["dont_fight"], 0)
        self.assertEqual(summary["total_potential_recovery"], 12.50)

    def test_all_dont_fight_still_sums_recovery(self):
        summary = app.results_summary(
            _frame([("Don't Fight", 8.00), ("Don't Fight", 1.25)])
        )
        self.assertEqual(summary["fight"], 0)
        self.assertEqual(summary["dont_fight"], 2)
        self.assertEqual(summary["total_potential_recovery"], 9.25)

    def test_other_labels_are_not_counted_but_recovery_is_included(self):
        summary = app.results_summary(
            _frame(
                [
                    ("Fight", 1.00),
                    ("Review", 2.00),
                    ("fight", 4.00),
                    ("Don't Fight", 0.50),
                ]
            )
        )
        self.assertEqual(summary["fight"], 1)
        self.assertEqual(summary["dont_fight"], 1)
        self.assertEqual(summary["total_potential_recovery"], 7.50)

    def test_rounds_the_total_to_cents(self):
        summary = app.results_summary(_frame([("Fight", 0.10), ("Fight", 0.20)]))
        self.assertEqual(summary["total_potential_recovery"], 0.30)

        summary = app.results_summary(
            _frame([("Don't Fight", 10.111), ("Fight", 20.222)])
        )
        self.assertEqual(summary["total_potential_recovery"], 30.33)

    def test_empty_batch_is_zero(self):
        summary = app.results_summary(
            pd.DataFrame(columns=["Recommendation", "Potential_Recovery"])
        )
        self.assertEqual(
            summary,
            {"fight": 0, "dont_fight": 0, "total_potential_recovery": 0.0},
        )

    def test_extra_columns_do_not_change_the_numbers(self):
        frame = _frame([("Fight", 5.00), ("Don't Fight", 1.50)])
        frame["Simulated_Win_Rate"] = [0.91, 0.22]
        frame["Transaction_ID"] = ["T1", "T2"]
        summary = app.results_summary(frame)
        self.assertEqual(summary["fight"], 1)
        self.assertEqual(summary["dont_fight"], 1)
        self.assertEqual(summary["total_potential_recovery"], 6.50)


class ResultsSummaryPreviewTests(unittest.TestCase):
    def test_process_shows_the_summary_and_keeps_the_existing_results(self):
        csv_bytes = (
            "transaction_id,amount,reason_code\n"
            "T1,100.00,Fraud\n"
            "T2,20.00,duplicate\n"
        ).encode()
        saved_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key-not-used"
        try:
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertEqual(len(at.exception), 0, list(at.exception))
            at.number_input[0].set_value(2)
            at.file_uploader[0].upload("merchant.csv", csv_bytes, "text/csv")
            with mock.patch("openai.OpenAI", _OfflineClient), mock.patch(
                "time.sleep", return_value=None
            ):
                at.button[0].click().run()
        finally:
            if saved_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = saved_key

        self.assertEqual(len(at.exception), 0, list(at.exception))

        preview = next(
            element.value
            for element in at.dataframe
            if "Recommendation" in getattr(element.value, "columns", [])
        )
        expected = app.results_summary(preview)
        metrics = {metric.label: metric.value for metric in at.metric}
        self.assertEqual(metrics["Fight"], f"{expected['fight']:,}")
        self.assertEqual(metrics["Don't Fight"], f"{expected['dont_fight']:,}")
        self.assertEqual(
            metrics["Potential recovery"],
            f"${expected['total_potential_recovery']:,.2f}",
        )
        self.assertEqual(expected["fight"] + expected["dont_fight"], len(preview))

        headings = [heading.value for heading in at.subheader]
        self.assertLess(headings.index("Batch summary"), headings.index("Preview"))
        self.assertTrue(any("API Usage" in info.value for info in at.info))
        self.assertTrue(
            any(
                "sum of Potential Recovery" in caption.value
                for caption in at.caption
            )
        )
        self.assertEqual(at.download_button[0].label, "Download CSV")
        self.assertEqual(list(preview.columns), [
            "Transaction_ID",
            "Reason_Code",
            "Recommendation",
            "Top_Evidence",
            "AI_Argument",
            "Simulated_Win_Rate",
            "Potential_Recovery",
        ])


def _frame(rows):
    return pd.DataFrame(rows, columns=["Recommendation", "Potential_Recovery"])


class _OfflineClient:
    """Stand-in that fails before any network call. The app already falls back."""

    def __init__(self, *args, **kwargs):
        pass

    @property
    def chat(self):
        raise RuntimeError("offline test client")
