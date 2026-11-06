"""Merchant amount text should become real numbers.

Covers currency symbols, thousands separators, accounting parentheses, and
currency codes in ``transform_csv.py`` and the in-app transform. Already-numeric
columns and the placeholder used for blank cells stay as they were.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import app
import transform_csv


class ParseAmountTests(unittest.TestCase):
    def test_merchant_amount_spellings(self):
        cases = [
            ("$1,234.50", 1234.50),
            (" 45.00 ", 45.00),
            ("(12.00)", -12.00),
            ("USD 30", 30.0),
            ("30 USD", 30.0),
            ("usd 30", 30.0),
            ("($1,234.50)", -1234.50),
            ("£1,000.00", 1000.0),
            ("+45.00", 45.0),
            ("19.99", 19.99),
            ("10", 10.0),
            (19.99, 19.99),
            (10, 10.0),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertAlmostEqual(transform_csv.parse_amount(raw), expected)

    def test_blank_and_unparseable_amounts_are_missing(self):
        for raw in (None, "", "   ", "not-a-number", float("nan")):
            with self.subTest(raw=raw):
                self.assertTrue(math.isnan(transform_csv.parse_amount(raw)))


class AmountColumnTests(unittest.TestCase):
    def test_formatted_amounts_become_numbers_and_rows_stay(self):
        csv_text = "\n".join(
            [
                "transaction_id,customer_name,amount,product,notes",
                'T1,Ada Lovelace,"$1,234.50",Analytical Engine,kept',
                'T2,Grace Hopper," 45.00 ",,',
                'T3,Alan Turing,"(12.00)",Machine,note',
                'T4,Edsger Dijkstra,"USD 30",Book,',
                "T5,Katherine Johnson,19.99,Orbit,clean",
                "T6,Margaret Hamilton,,Software,blank-amount",
                "T7,Donald Knuth,not-a-number,Art,bad",
                "",
            ]
        )
        expected = [1234.50, 45.00, -12.00, 30.0, 19.99, 0.0, 0.0]
        result, saved = self._transform(csv_text)

        self.assertEqual(len(result), 7)
        self.assertEqual(list(result.columns), transform_csv.REQUIRED_COLUMNS)
        self.assertTrue(pd.api.types.is_float_dtype(result["Amount"]))
        self.assertTrue(pd.api.types.is_float_dtype(saved["Amount"]))
        for index, amount in enumerate(expected):
            self.assertAlmostEqual(float(result.iloc[index]["Amount"]), amount)
            self.assertAlmostEqual(float(saved.iloc[index]["Amount"]), amount)
            self.assertFalse(pd.isna(result.iloc[index]["Amount"]))

        first = result.iloc[0]
        self.assertEqual(first["Transaction_ID"], "T1")
        self.assertEqual(first["Customer_Name"], "Ada Lovelace")
        self.assertEqual(first["Product"], "Analytical Engine")
        self.assertEqual(first["Notes"], "kept")
        self.assertEqual(first["Evidence_Receipt"], "No receipt provided")

        # Blank cells in other columns still use the placeholder fill.
        blank_product = result.iloc[1]
        self.assertEqual(blank_product["Customer_Name"], "Grace Hopper")
        self.assertEqual(blank_product["Product"], "Unknown Product")
        self.assertEqual(blank_product["Notes"], "No notes provided")

        # Unparseable text becomes the amount placeholder instead of dropping the row.
        kept = result.iloc[6]
        self.assertEqual(kept["Transaction_ID"], "T7")
        self.assertEqual(kept["Customer_Name"], "Donald Knuth")
        self.assertEqual(kept["Product"], "Art")
        self.assertEqual(kept["Notes"], "bad")

    def test_already_numeric_amounts_keep_value_and_dtype(self):
        cases = [
            "amount,notes\n10,keep\n20,also\n",
            "amount,notes\n19.99,keep\n88.10,also\n",
        ]
        for csv_text in cases:
            with self.subTest(csv_text=csv_text):
                with tempfile.TemporaryDirectory() as tmp:
                    src = Path(tmp) / "merchant.csv"
                    dest = Path(tmp) / "out.csv"
                    src.write_text(csv_text, encoding="utf-8")
                    source = pd.read_csv(src)
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = transform_csv.transform_csv(str(src), str(dest))
                pd.testing.assert_series_equal(
                    result["Amount"].reset_index(drop=True),
                    source["amount"].reset_index(drop=True),
                    check_names=False,
                )
                self.assertEqual(result.loc[0, "Notes"], "keep")
                self.assertEqual(result.loc[1, "Notes"], "also")

    def test_app_transform_parses_currency_text(self):
        frame = pd.DataFrame(
            {
                "transaction_id": ["T1", "T2", "T3"],
                "customer_name": ["Ada Lovelace", "Grace Hopper", "Alan Turing"],
                "amount": ["$1,234.50", " 45.00 ", "not-a-number"],
                "product": ["Analytical Engine", "Compiler", "Machine"],
            }
        )
        result = app.transform_csv(frame)

        self.assertEqual(len(result), 3)
        self.assertTrue(pd.api.types.is_float_dtype(result["Amount"]))
        self.assertAlmostEqual(float(result.iloc[0]["Amount"]), 1234.5)
        self.assertAlmostEqual(float(result.iloc[1]["Amount"]), 45.0)
        self.assertAlmostEqual(float(result.iloc[2]["Amount"]), app.DEFAULTS["Amount"])
        self.assertFalse(pd.isna(result.iloc[2]["Amount"]))
        self.assertEqual(result.iloc[0]["Transaction_ID"], "T1")
        self.assertEqual(result.iloc[0]["Customer_Name"], "Ada Lovelace")
        self.assertEqual(result.iloc[0]["Product"], "Analytical Engine")
        self.assertEqual(result.iloc[2]["Customer_Name"], "Alan Turing")
        self.assertEqual(result.iloc[2]["Product"], "Machine")

    def test_app_transform_keeps_numeric_amounts(self):
        frame = pd.DataFrame(
            {
                "amount": [19.99, 10],
                "notes": ["keep", "also"],
            }
        )
        result = app.transform_csv(frame)

        pd.testing.assert_series_equal(
            result["Amount"].reset_index(drop=True),
            frame["amount"].reset_index(drop=True),
            check_names=False,
        )
        self.assertEqual(result.loc[0, "Notes"], "keep")
        self.assertEqual(result.loc[1, "Notes"], "also")

    def test_blank_cell_in_a_numeric_amount_column_still_uses_placeholder(self):
        csv_text = "amount,customer_name\n19.99,Ada Lovelace\n,Grace Hopper\n"
        result, saved = self._transform(csv_text)

        self.assertAlmostEqual(float(result.iloc[0]["Amount"]), 19.99)
        self.assertAlmostEqual(float(result.iloc[1]["Amount"]), 0.0)
        self.assertAlmostEqual(float(saved.iloc[1]["Amount"]), 0.0)
        self.assertEqual(result.iloc[1]["Customer_Name"], "Grace Hopper")
        self.assertFalse(pd.isna(result.iloc[1]["Amount"]))
        self.assertEqual(result.iloc[1]["Product"], "Unknown Product")

    def _transform(self, csv_text: str):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "merchant.csv"
            dest = Path(tmp) / "out.csv"
            src.write_text(csv_text, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                result = transform_csv.transform_csv(str(src), str(dest))
            saved = pd.read_csv(dest)
        return result, saved
