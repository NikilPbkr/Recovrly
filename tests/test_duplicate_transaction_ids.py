"""Repeated merchant transaction IDs should be named when the CLI runs.

Blank cells and generated placeholders (the TXN_000000 fills used when an
ID is missing) are not duplicates of each other. Real IDs are still
reported when the file itself repeats them, and the repeated rows stay in
the standardized CSV.

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


class DuplicateTransactionIdTests(unittest.TestCase):
    def test_file_with_duplicate_real_ids(self):
        csv_text = "\n".join(
            [
                "transaction_id,amount,customer_name",
                "ch_b,10,Ada",
                "ch_a,11,Grace",
                "ch_b,12,Ada",
                "ch_a,13,Grace",
                "ch_b,14,Ada",
                "",
            ]
        )
        result, saved, stdout = self._transform(csv_text)

        self.assertEqual(
            result["Transaction_ID"].tolist(),
            ["ch_b", "ch_a", "ch_b", "ch_a", "ch_b"],
        )
        self.assertEqual(saved["Transaction_ID"].tolist(), result["Transaction_ID"].tolist())
        self.assertEqual([float(amount) for amount in result["Amount"]], [10, 11, 12, 13, 14])
        self.assertEqual(len(result), 5)
        self.assertIn(
            "\n".join(
                [
                    "  ⚠️  Duplicate Transaction_IDs (2):",
                    "      ch_b is repeated 3 times (lines 2, 4, 6)",
                    "      ch_a is repeated 2 times (lines 3, 5)",
                ]
            ),
            stdout,
        )
        self.assertLess(
            stdout.index("Duplicate Transaction_IDs"),
            stdout.index("Transformation complete"),
        )

    def test_file_with_all_unique_ids(self):
        csv_text = "\n".join(
            [
                "transaction_id,amount",
                "ch_1,10",
                "ch_2,11",
                "ch_10,12",
                "",
            ]
        )
        result, saved, stdout = self._transform(csv_text)

        self.assertEqual(result["Transaction_ID"].tolist(), ["ch_1", "ch_2", "ch_10"])
        self.assertEqual(saved["Transaction_ID"].tolist(), ["ch_1", "ch_2", "ch_10"])
        self.assertNotIn("Duplicate Transaction_IDs", stdout)
        self.assertIn("Transformation complete", stdout)

    def test_blank_and_placeholder_ids_are_not_duplicates(self):
        blank = "\n".join(
            [
                "transaction_id,amount,customer_name",
                ",10,Ada",
                ",11,Grace",
                "   ,12,Lin",
                "   ,13,Lin",
                "ch_only,14,Katherine",
                ",15,Mary",
                "",
            ]
        )
        with self.subTest("blank and whitespace ids"):
            result, _saved, stdout = self._transform(blank)
            self.assertEqual(result.loc[4, "Transaction_ID"], "ch_only")
            for index in (0, 1, 2, 3, 5):
                value = result.loc[index, "Transaction_ID"]
                self.assertTrue(
                    pd.isna(value) or str(value).strip() == "",
                    f"row {index} should still be blank, got {value!r}",
                )
            self.assertNotIn("Duplicate Transaction_IDs", stdout)
            self.assertIn("Transformation complete", stdout)

        missing_column = "\n".join(
            [
                "amount,customer_name",
                "10,Ada",
                "11,Grace",
                "12,Lin",
                "",
            ]
        )
        with self.subTest("generated placeholders"):
            result, saved, stdout = self._transform(missing_column)
            generated = ["TXN_000000", "TXN_000001", "TXN_000002"]
            self.assertEqual(result["Transaction_ID"].tolist(), generated)
            self.assertEqual(saved["Transaction_ID"].tolist(), generated)
            self.assertNotIn("Duplicate Transaction_IDs", stdout)
            self.assertIn("Transformation complete", stdout)

        with self.subTest("identical placeholder fills"):
            # The generator assigns a different ID per row. This locks the
            # other case: the same fill on every row is still not a duplicate.
            self.assertEqual(
                transform_csv.find_duplicate_transaction_ids(
                    ["TXN_000000", "TXN_000000", "TXN_000000"],
                    placeholder_rows={0, 1, 2},
                ),
                [],
            )

        mixed = "\n".join(
            [
                "transaction_id,amount",
                "ch_dup,10",
                ",11",
                "ch_dup,12",
                ",13",
                "",
            ]
        )
        with self.subTest("blanks do not hide a real duplicate"):
            result, _saved, stdout = self._transform(mixed)
            self.assertEqual(result["Transaction_ID"].tolist()[0], "ch_dup")
            self.assertEqual(result["Transaction_ID"].tolist()[2], "ch_dup")
            self.assertIn(
                "  ⚠️  Duplicate Transaction_IDs (1):\n"
                "      ch_dup is repeated 2 times (lines 2, 4)",
                stdout,
            )
            self.assertEqual(stdout.count("is repeated"), 1)
            self.assertNotIn("TXN_", stdout)

        repeated_placeholder_text = "\n".join(
            [
                "transaction_id,amount",
                "TXN_000000,10",
                "TXN_000000,11",
                "",
            ]
        )
        with self.subTest("placeholder spelling supplied by the merchant"):
            result, _saved, stdout = self._transform(repeated_placeholder_text)
            self.assertEqual(
                result["Transaction_ID"].tolist(),
                ["TXN_000000", "TXN_000000"],
            )
            self.assertIn(
                "TXN_000000 is repeated 2 times (lines 2, 3)",
                stdout,
            )

        numeric_with_blanks = "\n".join(
            [
                "transaction_id,amount",
                "1001,10",
                ",11",
                "1001,12",
                "1002,13",
                "",
            ]
        )
        with self.subTest("numeric ids beside blank cells"):
            _result, _saved, stdout = self._transform(numeric_with_blanks)
            self.assertIn(
                "  ⚠️  Duplicate Transaction_IDs (1):\n"
                "      1001 is repeated 2 times (lines 2, 4)",
                stdout,
            )
            self.assertNotIn("1001.0", stdout)
            self.assertNotIn("1002", stdout[stdout.index("Duplicate Transaction_IDs"):])

    def _transform(self, csv_text: str):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "merchant.csv"
            dest = Path(tmp) / "out.csv"
            src.write_text(csv_text, encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = transform_csv.transform_csv(str(src), str(dest))
            saved = pd.read_csv(dest)
        return result, saved, stdout.getvalue()
