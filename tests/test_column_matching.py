"""Merchant CSV headers should map onto Recoverly's standard columns.

Covers the fuzzy matcher in ``transform_csv.py`` and the copy in ``app.py``.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import app
import transform_csv


# Headers a merchant file might actually use, including case, spacing,
# separator, and punctuation differences, plus columns that should be ignored.
HEADER_EXPECTATIONS = [
    ("Order ID", "Transaction_ID"),
    ("TXN_ID", "Transaction_ID"),
    ("charge-id", "Transaction_ID"),
    ("Transaction_ID", "Transaction_ID"),
    ("ReasonCode", "Reason_Code"),
    ("CustomerName", "Customer_Name"),
    ("Cardholder Name", "Customer_Name"),
    ("BUYER_NAME", "Customer_Name"),
    ("CLIENT_EMAIL", "Customer_Email"),
    ("  email  ", "Customer_Email"),
    ("e-mail", "Customer_Email"),
    ("Created At", "Date"),
    ("TIMESTAMP", "Date"),
    ("charge_amount", "Amount"),
    ("Price", "Amount"),
    ("Item Description", "Product"),
    ("description", "Product"),
    ("Service", "Product"),
    ("Chargeback Reason", "Reason_Code"),
    ("CODE", "Reason_Code"),
    ("Receipt URL", "Evidence_Receipt"),
    ("proof_of_purchase", "Evidence_Receipt"),
    ("Tracking Number", "Evidence_Shipping"),
    ("delivery-confirmation", "Evidence_Shipping"),
    ("Correspondence", "Evidence_Email"),
    ("communication", "Evidence_Email"),
    ("Product Image", "Evidence_Photo"),
    ("PHOTO", "Evidence_Photo"),
    ("Dispute Status", "Current_Process"),
    ("stage", "Current_Process"),
    ("Additional Info", "Notes"),
    ("REMARKS", "Notes"),
    ("Warehouse Bin", None),
    ("Loyalty Tier", None),
    ("sku", None),
    ("", None),
    ("   ", None),
]

MERCHANT_COLUMNS = [
    "Loyalty Tier",
    "Chargeback Reason",
    "Photo Proof",
    "Order ID",
    "Additional Info",
    "Warehouse Bin",
    "Cardholder Name",
    "Correspondence",
    "Charge Amount",
    "Item Description",
    "Dispute Status",
    "Client Email",
    "Tracking Number",
    "Created At",
    "Receipt URL",
]

MERCHANT_ROWS = [
    {
        "Loyalty Tier": "gold",
        "Chargeback Reason": "Product Not Received",
        "Photo Proof": "widget.jpg",
        "Order ID": "ORD-100",
        "Additional Info": "VIP account",
        "Warehouse Bin": "B-12",
        "Cardholder Name": "Ada Lovelace",
        "Correspondence": "Customer confirmed delivery",
        "Charge Amount": "88.10",
        "Item Description": "Analytical Engine",
        "Dispute Status": "Evidence gathering",
        "Client Email": "ada@example.com",
        "Tracking Number": "1Z999",
        "Created At": "2024-05-06",
        "Receipt URL": "https://cdn.example.com/receipt.pdf",
    },
    {
        "Loyalty Tier": "silver",
        "Chargeback Reason": "Fraud",
        "Photo Proof": "shelf photo",
        "Order ID": "ORD-200",
        "Additional Info": "follow up",
        "Warehouse Bin": "A-1",
        "Cardholder Name": "Grace Hopper",
        "Correspondence": "https://cdn.example.com/thread.pdf",
        "Charge Amount": "12.50",
        "Item Description": "Compiler",
        "Dispute Status": "Pending review",
        "Client Email": "grace@example.com",
        "Tracking Number": "1Z000",
        "Created At": "2024-06-07",
        "Receipt URL": "signed receipt scan",
    },
]

EXPECTED_ROWS = [
    {
        "Transaction_ID": "ORD-100",
        "Customer_Name": "Ada Lovelace",
        "Customer_Email": "ada@example.com",
        "Date": "2024-05-06",
        "Amount": 88.10,
        "Product": "Analytical Engine",
        "Reason_Code": "Product Not Received",
        "Evidence_Receipt": "PDF document provided: https://cdn.example.com/receipt.pdf",
        "Evidence_Shipping": "1Z999",
        "Evidence_Email": "Customer confirmed delivery",
        "Evidence_Photo": "Image evidence provided: widget.jpg",
        "Current_Process": "Evidence gathering",
        "Notes": "VIP account",
    },
    {
        "Transaction_ID": "ORD-200",
        "Customer_Name": "Grace Hopper",
        "Customer_Email": "grace@example.com",
        "Date": "2024-06-07",
        "Amount": 12.50,
        "Product": "Compiler",
        "Reason_Code": "Fraud",
        "Evidence_Receipt": "signed receipt scan",
        "Evidence_Shipping": "1Z000",
        "Evidence_Email": "PDF document provided: https://cdn.example.com/thread.pdf",
        "Evidence_Photo": "shelf photo",
        "Current_Process": "Pending review",
        "Notes": "follow up",
    },
]

UNMATCHED_VALUES = ("gold", "B-12", "silver", "A-1")


def _header_variants(alias: str):
    """Case, surrounding whitespace, and separator spellings of one alias."""
    candidates = [
        alias,
        alias.upper(),
        alias.title(),
        f"  {alias}  ",
        f"  {alias.upper()}  ",
        alias.replace("_", " "),
        alias.replace("_", " ").title(),
        alias.replace("_", "-"),
    ]
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


class FuzzyColumnMatchingTests(unittest.TestCase):
    def test_app_and_cli_share_alias_and_column_lists(self):
        self.assertEqual(app.COLUMN_ALIASES, transform_csv.COLUMN_ALIASES)
        self.assertEqual(app.REQUIRED_COLUMNS, transform_csv.REQUIRED_COLUMNS)

    def test_each_alias_is_listed_once(self):
        # "description" used to be listed for Notes as well. The matcher returns
        # the first owner, so that second entry never matched and Product kept it.
        for module in (transform_csv, app):
            owners = {}
            for std_col, aliases in module.COLUMN_ALIASES.items():
                for alias in aliases:
                    self.assertNotIn(alias, owners)
                    owners[alias] = std_col
            self.assertEqual(owners["description"], "Product")

    def test_documented_header_spellings(self):
        matchers = (transform_csv.fuzzy_match_column, app.fuzzy_match_column)
        for header, expected in HEADER_EXPECTATIONS:
            for match in matchers:
                with self.subTest(header=header, matcher=match.__module__):
                    self.assertEqual(
                        match(header, transform_csv.COLUMN_ALIASES),
                        expected,
                    )

    def test_every_alias_variant_maps_to_its_column(self):
        matchers = (transform_csv.fuzzy_match_column, app.fuzzy_match_column)
        for std_col, aliases in transform_csv.COLUMN_ALIASES.items():
            for alias in aliases:
                for header in _header_variants(alias):
                    for match in matchers:
                        with self.subTest(header=header, column=std_col, matcher=match.__module__):
                            self.assertEqual(
                                match(header, transform_csv.COLUMN_ALIASES),
                                std_col,
                            )

    def test_canonical_headers_match_in_any_case(self):
        matchers = (transform_csv.fuzzy_match_column, app.fuzzy_match_column)
        for column in transform_csv.REQUIRED_COLUMNS:
            for header in (column, column.lower(), column.upper(), f"  {column}  "):
                for match in matchers:
                    with self.subTest(header=header, matcher=match.__module__):
                        self.assertEqual(
                            match(header, transform_csv.COLUMN_ALIASES),
                            column,
                        )

    def test_alias_headers_place_values_in_standard_columns(self):
        frame = pd.DataFrame(MERCHANT_ROWS, columns=MERCHANT_COLUMNS)
        cli_result, saved, app_result, stdout = self._run_both(frame)

        for result in (cli_result, saved, app_result):
            self.assertEqual(list(result.columns), transform_csv.REQUIRED_COLUMNS)
            self.assertEqual(len(result), 2)
            for index, expected in enumerate(EXPECTED_ROWS):
                self._assert_row(result, index, expected)
            self._assert_values_absent(result, UNMATCHED_VALUES)

        self.assertIn("Mapped 'Order ID' → 'Transaction_ID'", stdout)
        self.assertIn("Mapped 'Cardholder Name' → 'Customer_Name'", stdout)
        self.assertIn("No match for 'Warehouse Bin' (will be ignored)", stdout)
        self.assertIn("No match for 'Loyalty Tier' (will be ignored)", stdout)
        self.assertEqual(stdout.count("will be ignored"), 2)
        self.assertNotIn("'Product' missing", stdout)
        self.assertNotIn("'Transaction_ID' missing", stdout)

    def test_unrecognized_headers_are_ignored(self):
        frame = pd.DataFrame(
            {
                "Warehouse Bin": ["B-12", "A-1"],
                "Loyalty Tier": ["gold", "silver"],
            }
        )
        cli_result, saved, app_result, stdout = self._run_both(frame)

        for result, defaults in (
            (cli_result, transform_csv.DEFAULTS),
            (saved, transform_csv.DEFAULTS),
            (app_result, app.DEFAULTS),
        ):
            self.assertEqual(list(result.columns), transform_csv.REQUIRED_COLUMNS)
            self.assertEqual(len(result), 2)
            for index in range(2):
                self.assertEqual(result.iloc[index]["Transaction_ID"], f"TXN_{index:06d}")
                for column, default in defaults.items():
                    if column == "Transaction_ID" or callable(default):
                        continue
                    got = result.iloc[index][column]
                    if column == "Amount":
                        self.assertAlmostEqual(float(got), float(default))
                    else:
                        self.assertEqual(got, default)
            self._assert_values_absent(result, UNMATCHED_VALUES)

        self.assertNotIn("Mapped ", stdout)
        self.assertEqual(stdout.count("will be ignored"), 2)
        for column in frame.columns:
            self.assertIn(f"No match for '{column}' (will be ignored)", stdout)
        for column in transform_csv.REQUIRED_COLUMNS:
            self.assertIn(f"'{column}' missing, using placeholder", stdout)

    def test_surrounding_whitespace_on_a_header_still_maps(self):
        frame = pd.DataFrame(
            {"  notes  ": ["keep me"], "Warehouse Bin": ["UNMATCHED-VALUE"]}
        )
        cli_result, saved, app_result, stdout = self._run_both(frame)

        for result in (cli_result, saved, app_result):
            self.assertEqual(result.loc[0, "Notes"], "keep me")
            self.assertEqual(list(result.columns), transform_csv.REQUIRED_COLUMNS)
            self._assert_values_absent(result, ("UNMATCHED-VALUE",))
        self.assertIn("Mapped '  notes  ' → 'Notes'", stdout)
        self.assertIn("No match for 'Warehouse Bin' (will be ignored)", stdout)

    def test_leftmost_matching_header_supplies_the_value(self):
        values = {
            "id": "short",
            "order_id": "ORD-1",
            "name": "Pat",
            "customer_name": "Ada Lovelace",
        }
        cases = [
            (["id", "order_id", "name", "customer_name"], "short", "Pat"),
            (["order_id", "id", "customer_name", "name"], "ORD-1", "Ada Lovelace"),
        ]
        for columns, txn, person in cases:
            with self.subTest(columns=columns):
                frame = pd.DataFrame([values])[columns]
                cli_result, saved, app_result, _stdout = self._run_both(frame)
                for result in (cli_result, saved, app_result):
                    self.assertEqual(result.loc[0, "Transaction_ID"], txn)
                    self.assertEqual(result.loc[0, "Customer_Name"], person)

    def _run_both(self, frame: pd.DataFrame):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "merchant.csv"
            dest = Path(tmp) / "out.csv"
            frame.to_csv(src, index=False)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_result = transform_csv.transform_csv(str(src), str(dest))
            saved = pd.read_csv(dest)
            app_result = app.transform_csv(pd.read_csv(src))
        pd.testing.assert_frame_equal(cli_result, app_result)
        return cli_result, saved, app_result, stdout.getvalue()

    def _assert_row(self, result: pd.DataFrame, index: int, expected: dict):
        row = result.iloc[index]
        for column, value in expected.items():
            got = row[column]
            if column == "Amount":
                self.assertAlmostEqual(float(got), float(value))
            else:
                self.assertEqual(got, value)

    def _assert_values_absent(self, result: pd.DataFrame, forbidden):
        text = result.to_csv(index=False)
        for value in forbidden:
            self.assertNotIn(value, text)
