"""Merchant date text should become ISO dates.

Covers common export formats in ``transform_csv.py`` and the in-app transform.
An exact ``YYYY-MM-DD`` value stays as stored. Blank and unparseable cells use
the existing Date placeholder.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import contextlib
import io
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import pandas as pd

import app
import transform_csv


class ParseDateTests(unittest.TestCase):
    def test_common_export_formats(self):
        cases = [
            ("01/15/2024", "2024-01-15"),
            ("1/15/24", "2024-01-15"),
            ("2/1/24", "2024-02-01"),
            ("01-15-2024", "2024-01-15"),
            ("1-15-24", "2024-01-15"),
            ("2024/01/15", "2024-01-15"),
            ("2024-1-5", "2024-01-05"),
            ("2024-01-15", "2024-01-15"),
            ("  2024-01-15  ", "2024-01-15"),
            ("Jan 15, 2024", "2024-01-15"),
            ("JAN 15, 2024", "2024-01-15"),
            ("January 15, 2024", "2024-01-15"),
            ("Jan 15 2024", "2024-01-15"),
            ("September 3, 2024", "2024-09-03"),
            ("15-Jan-2024", "2024-01-15"),
            ("15-January-2024", "2024-01-15"),
            ("15 Jan 2024", "2024-01-15"),
            ("15-Jan-24", "2024-01-15"),
            ("2024-01-15T13:45:30", "2024-01-15"),
            ("2024-01-15T13:45:30Z", "2024-01-15"),
            ("2024-01-15t18:30:00z", "2024-01-15"),
            ("2024-01-15T13:45:30.123456+00:00", "2024-01-15"),
            ("2024-01-15 13:45:30", "2024-01-15"),
            ("2024-01-16 01:00:00-05:00", "2024-01-16"),
            ("2024-03-04 07:08:09.123456", "2024-03-04"),
            ("02/01/2024", "2024-02-01"),
            ("02/29/2024", "2024-02-29"),
            ("1/15/69", "1969-01-15"),
            ("1/15/68", "2068-01-15"),
            (date(2024, 1, 15), "2024-01-15"),
            (datetime(2024, 1, 15, 8, 30), "2024-01-15"),
            (pd.Timestamp("2024-01-15 23:30:00+00:00"), "2024-01-15"),
            (pd.Timestamp("2024-01-16 01:00:00-05:00"), "2024-01-16"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(transform_csv.parse_date(raw), expected)

    def test_blank_and_unparseable_dates_are_missing(self):
        blanks = [None, "", "   ", "\u00a0", float("nan"), pd.NA, pd.NaT]
        bad = [
            "not-a-date",
            "02/31/2024",
            "02/29/2023",
            "2024-02-31",
            "2024-13-01",
            "15/01/2024",
            "15-01-2024",
            "yesterday",
            True,
            20240115,
            "2024-01-15T99",
        ]
        for raw in blanks + bad:
            with self.subTest(raw=raw):
                self.assertIsNone(transform_csv.parse_date(raw))

    def test_datetime_dtype_uses_the_calendar_date(self):
        series = pd.to_datetime(
            pd.Series(["2024-07-04 21:05:00", None, "2024-12-01 00:00:00"])
        )
        normalized = transform_csv.normalize_dates(series)
        self.assertEqual(normalized.iloc[0], "2024-07-04")
        self.assertTrue(pd.isna(normalized.iloc[1]))
        self.assertEqual(normalized.iloc[2], "2024-12-01")

        zoned = pd.Series(
            pd.to_datetime(["2024-01-16 01:00:00-05:00", "2024-05-01 00:00:00-05:00"])
        )
        self.assertEqual(
            transform_csv.normalize_dates(zoned).tolist(),
            ["2024-01-16", "2024-05-01"],
        )


class DateColumnTests(unittest.TestCase):
    def test_export_formats_become_iso_and_rows_stay(self):
        csv_text = "\n".join(
            [
                "transaction_id,customer_name,Created At,amount,product,notes",
                "T1,Ada Lovelace,01/15/2024,19.99,Engine,us",
                "T2,Grace Hopper,1/15/24,10,Compiler,short-year",
                "T3,Alan Turing,2024-01-15,11,Machine,iso",
                'T4,Edsger Dijkstra,"Jan 15, 2024",12,Book,named',
                "T5,Katherine Johnson,15-Jan-2024,13,Orbit,dmy",
                "T6,Margaret Hamilton,2024-01-15T18:30:00Z,14,Software,iso-dt",
                "T7,Donald Knuth,2024-03-04 07:08:09,15,Art,space-dt",
                "T8,Barbara Liskov,2024-01-16 01:00:00-05:00,16,Types,offset",
                'T9,John Backus,"  01/15/2024  ",17,Fortran,padded',
                "T10,Fran Allen,,18,Optimizers,blank",
                "T11,Tony Hoare,not-a-date,19,Logic,bad",
                "T12,Niklaus Wirth,02/31/2024,20,Pascal,impossible",
                "",
            ]
        )
        parsed = [
            "2024-01-15",
            "2024-01-15",
            "2024-01-15",
            "2024-01-15",
            "2024-01-15",
            "2024-01-15",
            "2024-03-04",
            "2024-01-16",
            "2024-01-15",
        ]
        cli, saved, app_result = self._transform(csv_text)

        self.assertEqual(len(cli), 12)
        self.assertEqual(list(cli.columns), transform_csv.REQUIRED_COLUMNS)
        for index, expected in enumerate(parsed):
            self.assertEqual(cli.iloc[index]["Date"], expected)
            self.assertEqual(saved.iloc[index]["Date"], expected)
            self.assertEqual(app_result.iloc[index]["Date"], expected)
            self.assertIsInstance(cli.iloc[index]["Date"], str)

        for result, placeholder in (
            (cli, transform_csv.DEFAULTS["Date"]),
            (saved, transform_csv.DEFAULTS["Date"]),
            (app_result, app.DEFAULTS["Date"]),
        ):
            for index in (9, 10, 11):
                self.assertEqual(result.iloc[index]["Date"], placeholder)
                self.assertFalse(pd.isna(result.iloc[index]["Date"]))

        first = cli.iloc[0]
        self.assertEqual(first["Transaction_ID"], "T1")
        self.assertEqual(first["Customer_Name"], "Ada Lovelace")
        self.assertAlmostEqual(float(first["Amount"]), 19.99)
        self.assertEqual(first["Product"], "Engine")
        self.assertEqual(first["Notes"], "us")
        self.assertEqual(first["Evidence_Receipt"], "No receipt provided")
        self.assertEqual(app_result.iloc[0]["Customer_Name"], "Ada Lovelace")
        self.assertEqual(app_result.iloc[0]["Notes"], "us")
        self.assertAlmostEqual(float(app_result.iloc[0]["Amount"]), 19.99)

        kept = cli.iloc[10]
        self.assertEqual(kept["Transaction_ID"], "T11")
        self.assertEqual(kept["Customer_Name"], "Tony Hoare")
        self.assertEqual(kept["Product"], "Logic")
        self.assertEqual(kept["Notes"], "bad")
        self.assertEqual(cli.iloc[9]["Customer_Name"], "Fran Allen")
        self.assertEqual(cli.iloc[9]["Product"], "Optimizers")

        exported = cli.to_csv(index=False)
        for raw in ("01/15/2024", "1/15/24", "15-Jan-2024", "not-a-date", "02/31/2024"):
            self.assertNotIn(raw, exported)

    def test_exact_iso_dates_stay_unchanged(self):
        csv_text = "date,notes\n2024-01-15,keep\n2024-02-02,also\n"
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "merchant.csv"
            dest = Path(tmp) / "out.csv"
            src.write_text(csv_text, encoding="utf-8")
            source = pd.read_csv(src)
            with contextlib.redirect_stdout(io.StringIO()):
                result = transform_csv.transform_csv(str(src), str(dest))
            app_result = app.transform_csv(pd.read_csv(src))

        for transformed in (result, app_result):
            pd.testing.assert_series_equal(
                transformed["Date"].reset_index(drop=True),
                source["date"].reset_index(drop=True),
                check_names=False,
            )
        self.assertEqual(result.loc[0, "Notes"], "keep")
        self.assertEqual(result.loc[1, "Notes"], "also")
        self.assertEqual(app_result.loc[0, "Notes"], "keep")

    def test_datetime_column_is_formatted_for_csv(self):
        created = pd.to_datetime(pd.Series(["2024-07-04 21:05:00", None]))
        frame = pd.DataFrame(
            {
                "Created At": created,
                "Customer_Name": ["Ada Lovelace", "Grace Hopper"],
                "Amount": [10.5, 12.0],
            }
        )
        app_result = app.transform_csv(frame)
        self.assertEqual(app_result.loc[0, "Date"], "2024-07-04")
        self.assertIsInstance(app_result.loc[0, "Date"], str)
        self.assertEqual(app_result.loc[1, "Date"], app.DEFAULTS["Date"])
        self.assertEqual(app_result.loc[0, "Customer_Name"], "Ada Lovelace")
        self.assertEqual(app_result.loc[1, "Customer_Name"], "Grace Hopper")
        self.assertAlmostEqual(float(app_result.loc[0, "Amount"]), 10.5)
        self.assertFalse(pd.isna(app_result.loc[1, "Date"]))

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "merchant.csv"
            dest = Path(tmp) / "out.csv"
            frame.to_csv(src, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                cli = transform_csv.transform_csv(str(src), str(dest))
        self.assertEqual(cli.loc[0, "Date"], "2024-07-04")
        self.assertEqual(cli.loc[1, "Date"], transform_csv.DEFAULTS["Date"])
        self.assertEqual(cli.loc[1, "Customer_Name"], "Grace Hopper")

    def test_missing_date_column_still_uses_the_placeholder(self):
        csv_text = "customer_name,amount\nAda Lovelace,19.99\nGrace Hopper,4.00\n"
        cli, saved, app_result = self._transform(csv_text)
        for result, placeholder in (
            (cli, transform_csv.DEFAULTS["Date"]),
            (saved, transform_csv.DEFAULTS["Date"]),
            (app_result, app.DEFAULTS["Date"]),
        ):
            self.assertEqual(result["Date"].tolist(), [placeholder, placeholder])
            self.assertEqual(result.iloc[0]["Customer_Name"], "Ada Lovelace")
            self.assertFalse(any(pd.isna(value) for value in result["Date"]))

    def _transform(self, csv_text: str):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "merchant.csv"
            dest = Path(tmp) / "out.csv"
            src.write_text(csv_text, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                cli = transform_csv.transform_csv(str(src), str(dest))
            saved = pd.read_csv(dest)
            app_result = app.transform_csv(pd.read_csv(src))
        return cli, saved, app_result
