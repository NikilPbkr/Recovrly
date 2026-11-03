"""Missing CSV cells should be replaced with the documented placeholders.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

# Importing the Streamlit app constructs an OpenAI client. Tests never call it.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")

import pandas as pd

import app
import transform_csv
from app import transform_csv as transform_dataframe


class CsvPlaceholderTests(unittest.TestCase):
    def test_cli_fills_blank_cells_and_keeps_present_values(self):
        csv_text = "\n".join(
            [
                "transaction_id,customer_name,customer_email,date,amount,product,reason_code,notes,receipt",
                "T1,Ada Lovelace,ada@example.com,2025-01-02,19.99,Analytical Engine,Fraud,kept,https://files.example.com/receipt.pdf",
                "T2,,,,,,,",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "merchant.csv"
            dest = Path(tmp) / "out.csv"
            src.write_text(csv_text, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                result = transform_csv.transform_csv(str(src), str(dest))
            saved = pd.read_csv(dest)

            self.assertTrue(dest.exists())
            self._assert_placeholders(result, transform_csv.DEFAULTS["Date"])
            self._assert_placeholders(saved, transform_csv.DEFAULTS["Date"])

    def test_app_transform_fills_blank_cells(self):
        frame = pd.DataFrame(
            {
                "transaction_id": ["T1", "T2"],
                "customer_name": ["Ada Lovelace", None],
                "customer_email": ["ada@example.com", None],
                "date": ["2025-01-02", None],
                "amount": [19.99, None],
                "product": ["Analytical Engine", None],
                "reason_code": ["Fraud", None],
                "notes": ["kept", None],
                "receipt": ["https://files.example.com/receipt.pdf", None],
            }
        )
        self._assert_placeholders(transform_dataframe(frame), app.DEFAULTS["Date"])

    def test_sample_file_preserves_populated_fields(self):
        sample = Path(__file__).resolve().parents[1] / "Dummy_Chargeback_Data.csv"
        source = pd.read_csv(sample)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "standardized.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                result = transform_csv.transform_csv(str(sample), str(dest))

        for column in [
            "Transaction_ID",
            "Customer_Name",
            "Customer_Email",
            "Date",
            "Amount",
            "Product",
            "Reason_Code",
            "Current_Process",
            "Notes",
        ]:
            pd.testing.assert_series_equal(
                result[column].reset_index(drop=True),
                source[column].reset_index(drop=True),
                check_names=False,
            )

        first_receipt = source.loc[0, "Evidence_Receipt"]
        self.assertEqual(
            result.loc[0, "Evidence_Receipt"],
            f"PDF document provided: {first_receipt}",
        )
        self.assertEqual(len(result), len(source))
        self.assertEqual(list(result.columns), transform_csv.REQUIRED_COLUMNS)

    def _assert_placeholders(self, result: pd.DataFrame, missing_date: str):
        self.assertEqual(list(result.columns), transform_csv.REQUIRED_COLUMNS)

        first = result.iloc[0]
        self.assertEqual(first["Transaction_ID"], "T1")
        self.assertEqual(first["Customer_Name"], "Ada Lovelace")
        self.assertEqual(first["Customer_Email"], "ada@example.com")
        self.assertEqual(first["Date"], "2025-01-02")
        self.assertAlmostEqual(float(first["Amount"]), 19.99)
        self.assertEqual(first["Product"], "Analytical Engine")
        self.assertEqual(first["Reason_Code"], "Fraud")
        self.assertEqual(first["Notes"], "kept")
        self.assertEqual(
            first["Evidence_Receipt"],
            "PDF document provided: https://files.example.com/receipt.pdf",
        )
        self.assertEqual(first["Evidence_Shipping"], "No shipping evidence provided")
        self.assertEqual(first["Current_Process"], "Pending")

        second = result.iloc[1]
        self.assertEqual(second["Transaction_ID"], "T2")
        self.assertEqual(second["Customer_Name"], "Unknown")
        self.assertEqual(second["Customer_Email"], "unknown@example.com")
        self.assertEqual(second["Date"], missing_date)
        self.assertAlmostEqual(float(second["Amount"]), 0.0)
        self.assertEqual(second["Product"], "Unknown Product")
        self.assertEqual(second["Reason_Code"], "Unknown")
        self.assertEqual(second["Notes"], "No notes provided")
        self.assertEqual(second["Current_Process"], "Pending")
        self.assertFalse(pd.isna(second["Amount"]))
        self.assertFalse(pd.isna(second["Customer_Name"]))
        self.assertFalse(pd.isna(second["Reason_Code"]))
