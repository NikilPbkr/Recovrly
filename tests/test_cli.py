"""The command-line tool accepts an input path and an optional output path.

``main`` keeps the old positional form. One CSV writes
``<name>_standardized.csv`` beside the input, and a second path writes there
instead. ``--help`` names both paths and that default. A finished run prints
how many rows were written and where the file went.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import transform_csv

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "Dummy_Chargeback_Data.csv"


class CliHelpTests(unittest.TestCase):
    def test_help_names_the_input_the_optional_output_and_the_default(self):
        for flag in ("--help", "-h"):
            with self.subTest(flag=flag):
                stdout = io.StringIO()
                stderr = io.StringIO()
                with self.assertRaises(SystemExit) as caught:
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        transform_csv.main([flag])
                self.assertEqual(caught.exception.code, 0)
                help_text = stdout.getvalue()
                self.assertEqual(stderr.getvalue(), "")
                self.assertIn("usage: transform_csv.py", help_text)
                self.assertIn("[output.csv]", help_text)
                self.assertIn("input.csv", help_text)
                self.assertIn("_standardized.csv", help_text)
                self.assertIn("rows were written", help_text)
                self.assertIn("python transform_csv.py merchant_data.csv", help_text)
                self.assertNotIn("Traceback", help_text)

    def test_missing_input_exits_with_usage(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as caught:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                transform_csv.main([])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        usage = stderr.getvalue()
        self.assertIn("usage: transform_csv.py", usage)
        self.assertIn("input.csv", usage)
        self.assertNotIn("Traceback", usage)

    def test_an_extra_path_is_rejected_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            src = folder / "merchant.csv"
            dest = folder / "out.csv"
            src.write_text("transaction_id,amount\nT1,10\n", encoding="utf-8")
            default = Path(transform_csv.default_output_path(str(src)))
            stderr = io.StringIO()
            with self.assertRaises(SystemExit) as caught:
                with contextlib.redirect_stderr(stderr):
                    transform_csv.main([str(src), str(dest), str(folder / "extra.csv")])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("unrecognized arguments", stderr.getvalue())
            self.assertFalse(dest.exists())
            self.assertFalse(default.exists())


class CliRunTests(unittest.TestCase):
    def test_default_output_path_replaces_the_extension_in_place(self):
        self.assertEqual(
            transform_csv.default_output_path("merchant_data.csv"),
            "merchant_data_standardized.csv",
        )
        self.assertEqual(
            transform_csv.default_output_path("exports/merchant_data.csv"),
            "exports/merchant_data_standardized.csv",
        )
        self.assertEqual(
            transform_csv.default_output_path("notes"),
            "notes_standardized.csv",
        )

    def test_main_writes_the_bundled_sample_to_the_given_path(self):
        self.assertTrue(SAMPLE.is_file())
        sample_bytes = SAMPLE.read_bytes()
        source = pd.read_csv(SAMPLE)

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "standardized.csv"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                transform_csv.main([str(SAMPLE), str(dest)])
            text = stdout.getvalue()

            self.assertEqual(stderr.getvalue(), "")
            self.assertTrue(dest.is_file())
            self.assertIn("Transformation complete", text)
            self.assertIn(f"Rows written: {len(source)}", text)
            self.assertIn(f"Output saved to: {dest}", text)
            saved = pd.read_csv(dest)

        self.assertEqual(SAMPLE.read_bytes(), sample_bytes)
        self.assertEqual(list(saved.columns), transform_csv.REQUIRED_COLUMNS)
        self.assertEqual(len(saved), len(source))
        self.assertEqual(saved.loc[0, "Transaction_ID"], source.loc[0, "Transaction_ID"])
        self.assertTrue(str(saved.loc[0, "Evidence_Receipt"]).startswith("PDF document provided:"))

    def test_main_defaults_the_output_beside_a_copy_of_the_sample(self):
        source = pd.read_csv(SAMPLE)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "charges.csv"
            src.write_bytes(SAMPLE.read_bytes())
            expected = Path(transform_csv.default_output_path(str(src)))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                transform_csv.main([str(src)])
            text = stdout.getvalue()

            self.assertTrue(expected.is_file())
            self.assertEqual(expected.name, "charges_standardized.csv")
            self.assertIn(f"Rows written: {len(source)}", text)
            self.assertIn(f"Output saved to: {expected}", text)
            saved = pd.read_csv(expected)

        self.assertEqual(len(saved), len(source))
        self.assertEqual(list(saved.columns), transform_csv.REQUIRED_COLUMNS)

    def test_missing_file_exits_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.csv"
            dest = Path(tmp) / "out.csv"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with self.assertRaises(SystemExit) as caught:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    transform_csv.main([str(missing), str(dest)])
            self.assertEqual(caught.exception.code, 1)
            self.assertIn(f"File '{missing}' not found", stdout.getvalue())
            self.assertNotIn("Traceback", stdout.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertNotIn("Transformation complete", stdout.getvalue())
            self.assertFalse(dest.exists())
