"""The upload screen should say which columns mapped, and which were placeholders.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import os
import unittest
from unittest import mock

import pandas as pd
from streamlit.testing.v1 import AppTest

import app


class ColumnMappingSummaryTests(unittest.TestCase):
    def test_lists_every_required_field_once(self):
        summary = app.column_mapping_summary(["Order ID"])
        self.assertEqual([row["field"] for row in summary], app.REQUIRED_COLUMNS)

    def test_names_the_source_column_or_placeholder(self):
        summary = app.column_mapping_summary(
            ["Order ID", "Cardholder Name", "charge_amount", "Warehouse Bin"]
        )
        by_field = {row["field"]: row["source"] for row in summary}

        self.assertEqual(by_field["Transaction_ID"], "Order ID")
        self.assertEqual(by_field["Customer_Name"], "Cardholder Name")
        self.assertEqual(by_field["Amount"], "charge_amount")
        self.assertEqual(by_field["Product"], "placeholder")
        self.assertEqual(by_field["Date"], "placeholder")
        self.assertEqual(by_field["Notes"], "placeholder")
        self.assertNotIn("Warehouse Bin", by_field.values())

    def test_first_matching_header_is_the_recorded_source(self):
        left_first = {
            row["field"]: row["source"]
            for row in app.column_mapping_summary(
                ["id", "order_id", "name", "customer_name"]
            )
        }
        self.assertEqual(left_first["Transaction_ID"], "id")
        self.assertEqual(left_first["Customer_Name"], "name")

        right_first = {
            row["field"]: row["source"]
            for row in app.column_mapping_summary(
                ["order_id", "id", "customer_name", "name"]
            )
        }
        self.assertEqual(right_first["Transaction_ID"], "order_id")
        self.assertEqual(right_first["Customer_Name"], "customer_name")

    def test_whitespace_header_is_reported_as_typed(self):
        summary = {
            row["field"]: row["source"]
            for row in app.column_mapping_summary(["  notes  "])
        }
        self.assertEqual(summary["Notes"], "  notes  ")

    def test_summary_matches_values_transform_keeps(self):
        frame = pd.DataFrame(
            [
                {
                    "Order ID": "ORD-1",
                    "Cardholder Name": "Ada Lovelace",
                    "charge_amount": "88.10",
                    "Item Description": "Analytical Engine",
                    "notes": "keep me",
                    "Receipt URL": "signed receipt",
                    "Warehouse Bin": "B-12",
                }
            ]
        )
        summary = {
            row["field"]: row["source"]
            for row in app.column_mapping_summary(list(frame.columns))
        }
        result = app.transform_csv(frame)

        self.assertEqual(result.loc[0, "Transaction_ID"], frame.loc[0, "Order ID"])
        self.assertEqual(result.loc[0, "Customer_Name"], "Ada Lovelace")
        self.assertEqual(result.loc[0, "Notes"], "keep me")
        self.assertEqual(result.loc[0, "Product"], "Analytical Engine")
        self.assertEqual(result.loc[0, "Evidence_Receipt"], "signed receipt")
        self.assertNotIn("B-12", result.to_csv(index=False))

        for field, source in summary.items():
            if source != "placeholder":
                self.assertIn(source, frame.columns)
                continue
            default = app.DEFAULTS[field]
            if callable(default):
                self.assertEqual(result.loc[0, field], default(0))
            elif field == "Amount":
                self.assertAlmostEqual(float(result.loc[0, field]), float(default))
            else:
                self.assertEqual(result.loc[0, field], default)


class ColumnMappingPreviewTests(unittest.TestCase):
    def test_process_shows_the_mapping_before_results(self):
        csv_bytes = (
            "Order ID,Cardholder Name,charge_amount,Warehouse Bin\n"
            "ORD-1,Ada Lovelace,10.00,B-12\n"
        ).encode()
        saved_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key-not-used"
        try:
            at = AppTest.from_file(
                os.path.join(os.path.dirname(__file__), "..", "app.py"),
                default_timeout=30,
            ).run()
            self.assertEqual(len(at.exception), 0)

            at.number_input[0].set_value(1)
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

        self.assertEqual(len(at.exception), 0, at.exception)
        self.assertTrue(any("standardized" in s.value for s in at.success))

        expanders = [exp for exp in at.expander if exp.label.startswith("Column mapping")]
        self.assertEqual(len(expanders), 1)
        self.assertEqual(expanders[0].label, "Column mapping — 10 fields used placeholders")

        table = expanders[0].dataframe[0].value
        by_field = dict(zip(table["Standard field"], table["Source"]))
        self.assertEqual(list(table["Standard field"]), app.REQUIRED_COLUMNS)
        self.assertEqual(by_field["Transaction_ID"], "Order ID")
        self.assertEqual(by_field["Customer_Name"], "Cardholder Name")
        self.assertEqual(by_field["Amount"], "charge_amount")
        self.assertEqual(by_field["Product"], "placeholder")
        self.assertEqual(by_field["Date"], "placeholder")
        self.assertNotIn("Warehouse Bin", set(table["Source"]))
        self.assertIn(
            "placeholder means that field was missing",
            expanders[0].caption[0].value,
        )

        # Processing still finishes and shows the dispute preview.
        preview = next(
            element.value
            for element in at.dataframe
            if "Recommendation" in element.value.columns
        )
        self.assertEqual(preview.loc[0, "Transaction_ID"], "ORD-1")
        self.assertIn(preview.loc[0, "Recommendation"], ("Fight", "Don't Fight"))


class _OfflineClient:
    """Stand-in that fails before any network call. The app already falls back."""

    def __init__(self, *args, **kwargs):
        pass

    @property
    def chat(self):
        raise RuntimeError("offline test client")
