"""The bundled sample CSV should standardize through the command-line tool.

Runs ``transform_csv.py`` on ``Dummy_Chargeback_Data.csv`` and writes the
result only under a temporary directory. The script does not import the app
or call OpenAI. The child process is started with any API key removed.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import transform_csv

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "Dummy_Chargeback_Data.csv"
ISO_DATE = r"^\d{4}-\d{2}-\d{2}$"


def _nonempty_data_rows(path: Path) -> int:
    """Count data lines that hold a value, skipping the header and blank lines.

    A line of only commas is not a data row. ``transform_csv.py`` keeps one
    output row per non-empty input row of the bundled sample.
    """
    count = 0
    header_seen = False
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        if not header_seen:
            header_seen = True
            continue
        if line.replace(",", "").strip():
            count += 1
    return count


def _present_text(series: pd.Series) -> list:
    """Cell text that is not missing or blank."""
    present = []
    for value in series:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            present.append(text)
    return present


class SampleCsvSmokeTests(unittest.TestCase):
    def test_cli_standardizes_the_bundled_sample_outside_the_repo(self):
        self.assertTrue(SAMPLE.is_file())
        expected_rows = _nonempty_data_rows(SAMPLE)
        self.assertGreater(expected_rows, 0)
        sample_bytes = SAMPLE.read_bytes()
        repo_csvs = set(ROOT.glob("*.csv"))

        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("OPENAI_ADMIN_KEY", None)

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "standardized.csv"
            self.assertFalse(Path(tmp).resolve().is_relative_to(ROOT.resolve()))
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "transform_csv.py"),
                    str(SAMPLE),
                    str(dest),
                ],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"Loaded {expected_rows} rows", result.stdout)
            self.assertIn("Transformation complete", result.stdout)
            self.assertIn(str(dest), result.stdout)
            self.assertNotIn("Traceback", result.stdout)
            self.assertNotIn("Traceback", result.stderr)

            self.assertTrue(dest.is_file())
            self.assertGreater(dest.stat().st_size, 0)
            self.assertTrue(dest.resolve().is_relative_to(Path(tmp).resolve()))
            saved = pd.read_csv(dest)

        self.assertEqual(SAMPLE.read_bytes(), sample_bytes)
        self.assertEqual(set(ROOT.glob("*.csv")), repo_csvs)

        self.assertEqual(list(saved.columns), transform_csv.REQUIRED_COLUMNS)
        self.assertEqual(len(saved), expected_rows)

        self.assertTrue(pd.api.types.is_numeric_dtype(saved["Amount"]))
        amounts = pd.to_numeric(saved["Amount"], errors="coerce")
        self.assertEqual(int(amounts.notna().sum()), expected_rows)

        source = pd.read_csv(SAMPLE)
        present_dates = _present_text(saved["Date"])
        self.assertGreater(len(present_dates), 0)
        self.assertEqual(len(present_dates), len(_present_text(source["Date"])))
        for text in present_dates:
            self.assertRegex(text, ISO_DATE)
