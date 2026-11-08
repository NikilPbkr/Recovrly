"""Unreadable merchant CSVs should stop with a clear message.

Covers an empty file, a header with no data rows, a non-UTF-8 file, and a
CSV that does not parse. The Streamlit app also stops when no column matches
a chargeback field, instead of sending an all-placeholder table to the model.
The command-line transformer still standardizes that last case with
placeholders; that contract is covered by the column-matching tests.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

import app
import transform_csv

ROOT = Path(__file__).resolve().parents[1]

EMPTY = b""
WHITESPACE = b"  \n\n\t"
HEADER_ONLY = b"transaction_id,amount,customer_name\n"
HEADER_ONLY_NO_NEWLINE = b"transaction_id,amount"
LATIN_1 = "customer_name,amount\nJosé García,12.50\n".encode("latin-1")
UTF_16 = "customer_name,amount\nAda,10\n".encode("utf-16")
BINARY = b"\xff\xfe\x00\x00not a csv"
UNCLOSED_QUOTE = b'name,amount\n"unterminated,12\n'
NO_MATCH = b"foo,bar,baz\n1,2,3\n"
VALID = "transaction_id,amount\nT1,$19.99\n".encode("utf-8-sig")
BLANK_ROW = b"amount,customer_name\n,\n"


class LoadMerchantCsvTests(unittest.TestCase):
    def test_empty_and_whitespace_files(self):
        for raw in (EMPTY, WHITESPACE, b"\xef\xbb\xbf"):
            with self.subTest(raw=raw):
                self._assert_load_error(raw, transform_csv.EMPTY_CSV_MESSAGE)

    def test_header_only_files(self):
        for raw in (HEADER_ONLY, HEADER_ONLY_NO_NEWLINE, b"transaction_id,amount\n\n"):
            with self.subTest(raw=raw):
                self._assert_load_error(raw, transform_csv.HEADER_ONLY_CSV_MESSAGE)

    def test_non_utf8_files(self):
        for raw in (LATIN_1, UTF_16, BINARY):
            with self.subTest(raw=raw):
                self._assert_load_error(raw, transform_csv.NOT_UTF8_MESSAGE)

    def test_malformed_csv(self):
        self._assert_load_error(UNCLOSED_QUOTE, transform_csv.MALFORMED_CSV_MESSAGE)

    def test_blank_data_row_is_not_header_only(self):
        frame = transform_csv.load_merchant_csv(io.BytesIO(BLANK_ROW))
        self.assertEqual(len(frame), 1)

    def test_unmatched_headers_still_load(self):
        # The transformer fills this file with placeholders. Rejecting it here
        # would change that contract.
        frame = transform_csv.load_merchant_csv(io.BytesIO(NO_MATCH))
        self.assertEqual(list(frame.columns), ["foo", "bar", "baz"])
        self.assertEqual(len(frame), 1)

    def test_utf8_bom_file_still_loads(self):
        frame = transform_csv.load_merchant_csv(io.BytesIO(VALID))
        self.assertEqual(frame.loc[0, "transaction_id"], "T1")
        self.assertEqual(frame.loc[0, "amount"], "$19.99")

    def test_valid_file_still_transforms(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "merchant.csv"
            dest = Path(tmp) / "out.csv"
            src.write_bytes(VALID)
            with contextlib.redirect_stdout(io.StringIO()):
                result = transform_csv.transform_csv(str(src), str(dest))
        self.assertEqual(result.loc[0, "Transaction_ID"], "T1")
        self.assertAlmostEqual(float(result.loc[0, "Amount"]), 19.99)
        self.assertEqual(len(result), 1)

    def test_one_recognized_column_is_enough_to_continue(self):
        frame = transform_csv.load_merchant_csv(
            io.BytesIO(b"warehouse,amount\nA-1,5\n")
        )
        self.assertEqual(app.resolve_column_sources(frame.columns), {"Amount": "amount"})

    def _assert_load_error(self, raw: bytes, message: str):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "merchant.csv"
            src.write_bytes(raw)
            for source in (src, io.BytesIO(raw)):
                with self.subTest(source=type(source).__name__):
                    with self.assertRaises(transform_csv.CsvInputError) as caught:
                        transform_csv.load_merchant_csv(source)
                    self.assertEqual(str(caught.exception), message)


class CliCsvInputTests(unittest.TestCase):
    def test_transform_exits_with_the_user_message(self):
        cases = [
            (EMPTY, transform_csv.EMPTY_CSV_MESSAGE),
            (HEADER_ONLY, transform_csv.HEADER_ONLY_CSV_MESSAGE),
            (LATIN_1, transform_csv.NOT_UTF8_MESSAGE),
            (UNCLOSED_QUOTE, transform_csv.MALFORMED_CSV_MESSAGE),
        ]
        for raw, message in cases:
            with self.subTest(message=message):
                self._assert_exit(raw, message)

    def test_cli_script_reports_an_empty_file_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "empty.csv"
            src.write_bytes(EMPTY)
            result = subprocess.run(
                [sys.executable, str(ROOT / "transform_csv.py"), str(src)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn(transform_csv.EMPTY_CSV_MESSAGE, result.stdout)
        self.assertNotIn("Traceback", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_missing_file_exits_without_a_traceback(self):
        self._assert_exit_on(
            "/tmp/does-not-exist-recovrly.csv",
            "not found",
        )

    def _assert_exit(self, raw: bytes, message: str):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "merchant.csv"
            dest = Path(tmp) / "out.csv"
            src.write_bytes(raw)
            self._assert_exit_on(str(src), message, output_path=str(dest))
            self.assertFalse(dest.exists())

    def _assert_exit_on(self, path: str, message: str, output_path: str = None):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as caught:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                transform_csv.transform_csv(path, output_path)
        self.assertEqual(caught.exception.code, 1)
        self.assertIn(message, stdout.getvalue())
        self.assertNotIn("Traceback", stdout.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertNotIn("Transformation complete", stdout.getvalue())


class AppCsvInputTests(unittest.TestCase):
    def setUp(self):
        self._saved_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key-not-used"

    def tearDown(self):
        if self._saved_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self._saved_key

    def test_upload_errors_stop_before_processing(self):
        cases = [
            (EMPTY, transform_csv.EMPTY_CSV_MESSAGE),
            (WHITESPACE, transform_csv.EMPTY_CSV_MESSAGE),
            (HEADER_ONLY, transform_csv.HEADER_ONLY_CSV_MESSAGE),
            (LATIN_1, transform_csv.NOT_UTF8_MESSAGE),
            (UTF_16, transform_csv.NOT_UTF8_MESSAGE),
            (BINARY, transform_csv.NOT_UTF8_MESSAGE),
            (UNCLOSED_QUOTE, transform_csv.MALFORMED_CSV_MESSAGE),
            (NO_MATCH, app.NO_MATCHING_COLUMNS_MESSAGE),
        ]
        for raw, message in cases:
            with self.subTest(message=message):
                at = self._process(raw)
                self.assertEqual(len(at.exception), 0, list(at.exception))
                self.assertIn(message, [error.value for error in at.error])
                self.assertEqual(len(at.success), 0)
                self.assertFalse(
                    any("Batch ranking" in warning.value for warning in at.warning)
                )
                self.assertFalse(
                    any("API Usage" in info.value for info in at.info)
                )

    def _process(self, content: bytes):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0, list(at.exception))
        at.file_uploader[0].upload("disputes.csv", content).run()
        at.button[0].click().run()
        return at
