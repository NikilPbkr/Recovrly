"""The results table should be scannable after a batch is processed.

Row order and column formatting are display-only. They do not change how a
win rate or a recommendation is calculated, and the CSV download keeps the
processed order and the original numbers.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import json
import os
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from streamlit.testing.v1 import AppTest

import app

ROOT = Path(__file__).resolve().parents[1]


class ResultsPreviewTests(unittest.TestCase):
    def test_fight_comes_before_a_larger_dont_fight(self):
        preview = app.results_preview(
            _frame(
                [
                    ("B", "Don't Fight", 0.35, 500.00),
                    ("A", "Fight", 0.90, 10.00),
                ]
            )
        )
        self.assertEqual(list(preview["Transaction_ID"]), ["A", "B"])
        self.assertEqual(list(preview["Potential_Recovery"]), [10.00, 500.00])

    def test_higher_recovery_comes_first_within_a_recommendation(self):
        preview = app.results_preview(
            _frame(
                [
                    ("low", "Fight", 0.80, 5.00),
                    ("high", "Fight", 0.42, 80.00),
                    ("mid", "Don't Fight", 0.20, 15.00),
                    ("skip-high", "Don't Fight", 0.30, 40.00),
                ]
            )
        )
        self.assertEqual(
            list(preview["Transaction_ID"]),
            ["high", "low", "skip-high", "mid"],
        )

    def test_equal_recovery_keeps_processed_order(self):
        preview = app.results_preview(
            _frame(
                [
                    ("first", "Fight", 0.50, 10.00),
                    ("second", "Fight", 0.70, 10.00),
                    ("third", "Don't Fight", 0.20, 0.00),
                    ("fourth", "Don't Fight", 0.25, 0.00),
                ]
            )
        )
        self.assertEqual(
            list(preview["Transaction_ID"]),
            ["first", "second", "third", "fourth"],
        )

    def test_other_labels_sort_after_fight_and_dont_fight(self):
        preview = app.results_preview(
            _frame(
                [
                    ("review", "Review", 0.50, 999.00),
                    ("skip", "Don't Fight", 0.20, 1.00),
                    ("fight", "Fight", 0.60, 2.00),
                ]
            )
        )
        self.assertEqual(list(preview["Transaction_ID"]), ["fight", "skip", "review"])

    def test_values_and_input_frame_stay_unchanged(self):
        frame = _frame([("T1", "Fight", 0.42, 120.50), ("T2", "Don't Fight", 0.22, 3.00)])
        original = frame.copy()
        preview = app.results_preview(frame)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(preview.loc[0, "Simulated_Win_Rate"], 0.42)
        self.assertEqual(preview.loc[0, "Potential_Recovery"], 120.50)
        self.assertEqual(preview.loc[0, "AI_Argument"], "Argue T1")
        self.assertNotIn("_priority", preview.columns)
        self.assertNotIn("_original", preview.columns)

    def test_column_order_puts_the_decision_before_the_argument(self):
        preview = app.results_preview(
            _frame([("T1", "Fight", 0.50, 1.00)])
        )
        self.assertEqual(list(preview.columns), list(app.RESULTS_PREVIEW_COLUMNS))
        self.assertLess(
            list(preview.columns).index("Recommendation"),
            list(preview.columns).index("Simulated_Win_Rate"),
        )
        self.assertLess(
            list(preview.columns).index("Potential_Recovery"),
            list(preview.columns).index("AI_Argument"),
        )

    def test_extra_columns_are_left_out(self):
        frame = _frame([("T1", "Fight", 0.50, 1.00)])
        frame["Amount"] = [25.0]
        preview = app.results_preview(frame)
        self.assertNotIn("Amount", preview.columns)

    def test_empty_batch_keeps_the_preview_columns(self):
        empty = pd.DataFrame(columns=list(app.RESULTS_PREVIEW_COLUMNS))
        preview = app.results_preview(empty)
        self.assertTrue(preview.empty)
        self.assertEqual(list(preview.columns), list(app.RESULTS_PREVIEW_COLUMNS))


class ResultsPreviewDisplayTests(unittest.TestCase):
    def test_win_rate_is_a_percent_and_recovery_is_dollars(self):
        display = app.results_preview_display()
        self.assertEqual(tuple(display), app.RESULTS_PREVIEW_COLUMNS)
        self.assertEqual(display["Simulated_Win_Rate"]["label"], "Win rate")
        self.assertEqual(display["Simulated_Win_Rate"]["format"], "percent")
        self.assertIn("40%", display["Simulated_Win_Rate"]["help"])
        self.assertEqual(display["Potential_Recovery"]["label"], "Potential recovery")
        self.assertEqual(display["Potential_Recovery"]["format"], "dollar")
        self.assertNotIn("format", display["AI_Argument"])
        self.assertEqual(display["AI_Argument"]["width"], "large")


class ResultsPreviewAppTests(unittest.TestCase):
    def test_process_lists_the_higher_recovery_first_and_formats_numbers(self):
        # File order is the smaller recovery first. The table should flip it.
        csv_bytes = (
            "transaction_id,amount,reason_code\n"
            "T-small,20.00,duplicate\n"
            "T-big,100.00,Fraud\n"
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

        preview_el = next(
            element
            for element in at.dataframe
            if "Recommendation" in getattr(element.value, "columns", [])
        )
        preview = preview_el.value
        self.assertEqual(list(preview["Transaction_ID"]), ["T-big", "T-small"])
        self.assertGreater(
            preview.loc[0, "Potential_Recovery"],
            preview.loc[1, "Potential_Recovery"],
        )
        self.assertLess(float(preview.loc[0, "Simulated_Win_Rate"]), 1.01)
        self.assertEqual(list(preview.columns), list(app.RESULTS_PREVIEW_COLUMNS))

        expected = app.results_summary(preview)
        metrics = {metric.label: metric.value for metric in at.metric}
        self.assertEqual(
            metrics["Potential recovery"],
            f"${expected['total_potential_recovery']:,.2f}",
        )
        self.assertEqual(expected["fight"] + expected["dont_fight"], len(preview))

        config = json.loads(preview_el.proto.columns)
        self.assertEqual(config["Simulated_Win_Rate"]["label"], "Win rate")
        self.assertEqual(
            config["Simulated_Win_Rate"]["type_config"]["format"], "percent"
        )
        self.assertIn("40%", config["Simulated_Win_Rate"]["help"])
        self.assertEqual(config["Potential_Recovery"]["label"], "Potential recovery")
        self.assertEqual(
            config["Potential_Recovery"]["type_config"]["format"], "dollar"
        )
        self.assertEqual(config["AI_Argument"]["label"], "Argument")
        self.assertEqual(config["_index"]["hidden"], True)
        self.assertTrue(
            any(
                "listed first, highest potential recovery" in caption.value
                for caption in at.caption
            )
        )
        download_labels = [button.label for button in at.download_button]
        self.assertIn("Download CSV", download_labels)
        self.assertIn("Download sample CSV", download_labels)


def _frame(rows):
    return pd.DataFrame(
        [
            {
                "Transaction_ID": txn,
                "Reason_Code": "Fraud",
                "Recommendation": recommendation,
                "Top_Evidence": "Shipping, Receipt",
                "AI_Argument": f"Argue {txn}",
                "Simulated_Win_Rate": win_rate,
                "Potential_Recovery": recovery,
                "Notes": "ignored",
            }
            for txn, recommendation, win_rate, recovery in rows
        ]
    )


class _OfflineClient:
    """Stand-in that fails before any network call. The app already falls back."""

    def __init__(self, *args, **kwargs):
        pass

    @property
    def chat(self):
        raise RuntimeError("offline test client")
