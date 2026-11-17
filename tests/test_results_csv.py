"""The results download is the table on the page, written as CSV bytes.

``dataframe_to_csv_bytes`` only serializes a DataFrame. It does not sort,
reformat numbers, or call OpenAI. The page passes it the rows the table
is showing, including after the recommendation filter.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import io
import unittest
from unittest import mock

import pandas as pd

import app


class DataframeToCsvBytesTests(unittest.TestCase):
    def test_keeps_column_order_and_drops_the_index(self):
        frame = pd.DataFrame(
            {"Potential_Recovery": [1.5], "Recommendation": ["Fight"]},
            index=["keep-me-out"],
        )
        text = app.dataframe_to_csv_bytes(frame).decode("utf-8")
        self.assertTrue(text.startswith("Potential_Recovery,Recommendation\n"))
        self.assertNotIn("keep-me-out", text)
        parsed = pd.read_csv(io.StringIO(text))
        self.assertEqual(list(parsed.columns), ["Potential_Recovery", "Recommendation"])
        self.assertEqual(parsed.loc[0, "Recommendation"], "Fight")
        self.assertEqual(parsed.loc[0, "Potential_Recovery"], 1.5)

    def test_keeps_the_row_order_it_was_given(self):
        frame = pd.DataFrame(
            {
                "Transaction_ID": ["second", "first"],
                "Recommendation": ["Don't Fight", "Fight"],
            }
        )
        parsed = pd.read_csv(io.BytesIO(app.dataframe_to_csv_bytes(frame)))
        self.assertEqual(list(parsed["Transaction_ID"]), ["second", "first"])
        self.assertEqual(list(parsed["Recommendation"]), ["Don't Fight", "Fight"])

    def test_quotes_commas_quotes_and_newlines(self):
        argument = 'He said "no", then\nstopped'
        frame = pd.DataFrame({"AI_Argument": [argument]})
        parsed = pd.read_csv(io.BytesIO(app.dataframe_to_csv_bytes(frame)))
        self.assertEqual(list(parsed.columns), ["AI_Argument"])
        self.assertEqual(parsed.loc[0, "AI_Argument"], argument)

    def test_encodes_non_ascii_as_utf8(self):
        frame = pd.DataFrame({"Notes": ["José — café"]})
        raw = app.dataframe_to_csv_bytes(frame)
        self.assertIsInstance(raw, bytes)
        self.assertIn("José — café".encode("utf-8"), raw)
        parsed = pd.read_csv(io.BytesIO(raw))
        self.assertEqual(parsed.loc[0, "Notes"], "José — café")

    def test_win_rate_stays_a_fraction(self):
        frame = pd.DataFrame({"Simulated_Win_Rate": [0.42]})
        text = app.dataframe_to_csv_bytes(frame).decode("utf-8")
        self.assertIn("0.42", text)
        self.assertNotIn("%", text)
        parsed = pd.read_csv(io.StringIO(text))
        self.assertEqual(float(parsed.loc[0, "Simulated_Win_Rate"]), 0.42)

    def test_empty_frame_is_a_header_only(self):
        empty = pd.DataFrame(columns=list(app.RESULTS_PREVIEW_COLUMNS))
        text = app.dataframe_to_csv_bytes(empty).decode("utf-8")
        self.assertEqual(text, ",".join(app.RESULTS_PREVIEW_COLUMNS) + "\n")

    def test_does_not_change_the_frame(self):
        frame = pd.DataFrame(
            {"Recommendation": ["Fight"], "Potential_Recovery": [10.0]}
        )
        original = frame.copy()
        app.dataframe_to_csv_bytes(frame)
        pd.testing.assert_frame_equal(frame, original)

    def test_does_not_call_openai(self):
        frame = pd.DataFrame({"Recommendation": ["Fight"]})
        with mock.patch("openai.OpenAI", side_effect=AssertionError("OpenAI")):
            raw = app.dataframe_to_csv_bytes(frame)
        self.assertIn(b"Fight", raw)

    def test_displayed_table_round_trips_in_the_same_order(self):
        # Processed order is Don't Fight first. The file follows the table.
        results = pd.DataFrame(
            [
                {
                    "Transaction_ID": "B",
                    "Reason_Code": "Fraud",
                    "Recommendation": "Don't Fight",
                    "Top_Evidence": "Receipt",
                    "AI_Argument": "Skip B",
                    "Simulated_Win_Rate": 0.22,
                    "Potential_Recovery": 500.0,
                    "Notes": "left out of the table",
                },
                {
                    "Transaction_ID": "A",
                    "Reason_Code": "Product Not Received",
                    "Recommendation": "Fight",
                    "Top_Evidence": "Shipping, Receipt",
                    "AI_Argument": 'Fight "A", please',
                    "Simulated_Win_Rate": 0.90,
                    "Potential_Recovery": 10.0,
                    "Notes": "also left out",
                },
            ]
        )
        preview = app.results_preview(results)
        parsed = pd.read_csv(io.BytesIO(app.dataframe_to_csv_bytes(preview)))
        self.assertEqual(list(parsed.columns), list(preview.columns))
        self.assertEqual(list(parsed.columns), list(app.RESULTS_PREVIEW_COLUMNS))
        self.assertEqual(list(parsed["Transaction_ID"]), ["A", "B"])
        self.assertEqual(list(parsed["Recommendation"]), ["Fight", "Don't Fight"])
        self.assertEqual(
            [float(value) for value in parsed["Simulated_Win_Rate"]],
            [0.90, 0.22],
        )
        self.assertEqual(
            [float(value) for value in parsed["Potential_Recovery"]],
            [10.0, 500.0],
        )
        self.assertEqual(parsed.loc[0, "AI_Argument"], 'Fight "A", please')
        self.assertNotIn("Notes", parsed.columns)
        self.assertNotIn("left out", app.dataframe_to_csv_bytes(preview).decode("utf-8"))
