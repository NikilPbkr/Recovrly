"""The bundled sample CSV can be tried from the app without an upload.

Loading it is a local file read. OpenAI runs only after Process, and only
through the same path an uploaded file already uses. An uploaded file still
wins when one is selected.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import io
import os
import unittest
from unittest import mock

import pandas as pd
from streamlit.testing.v1 import AppTest

import app
import transform_csv

ROOT_SAMPLE = app.sample_csv_path()


class SampleCsvHelperTests(unittest.TestCase):
    def test_sample_bytes_match_the_bundled_file_and_load_like_an_upload(self):
        with mock.patch("openai.OpenAI", side_effect=AssertionError("OpenAI")):
            raw = app.load_sample_csv_bytes()
            source = app.sample_csv_source()
            again = app.sample_csv_source()

        self.assertEqual(raw, ROOT_SAMPLE.read_bytes())
        self.assertTrue(raw.startswith(b"Transaction_ID,"))
        self.assertTrue(callable(getattr(source, "getvalue")))
        self.assertIsNot(source, again)
        self.assertEqual(source.getvalue(), again.getvalue())

        from_buffer = transform_csv.load_merchant_csv(source)
        from_path = transform_csv.load_merchant_csv(ROOT_SAMPLE)
        pd.testing.assert_frame_equal(from_buffer, from_path)
        self.assertGreater(len(from_buffer), 1)
        self.assertIn("Transaction_ID", from_buffer.columns)
        self.assertIn("Amount", from_buffer.columns)

    def test_no_choice_returns_no_source(self):
        source, using_sample = app.choose_csv_source(None, False)
        self.assertIsNone(source)
        self.assertFalse(using_sample)

    def test_uploaded_file_wins_over_the_sample(self):
        uploaded = io.BytesIO(b"transaction_id,amount\nUPLOADED,1\n")
        with mock.patch.object(
            app, "sample_csv_source", side_effect=AssertionError("sample read")
        ):
            source, using_sample = app.choose_csv_source(uploaded, True)
        self.assertIs(source, uploaded)
        self.assertFalse(using_sample)

    def test_sample_is_used_only_when_nothing_was_uploaded(self):
        with mock.patch("openai.OpenAI", side_effect=AssertionError("OpenAI")):
            source, using_sample = app.choose_csv_source(None, True)
        self.assertTrue(using_sample)
        self.assertEqual(source.getvalue(), app.load_sample_csv_bytes())
        frame = transform_csv.load_merchant_csv(source)
        self.assertEqual(frame.iloc[0]["Transaction_ID"], _sample_transaction_id())


class SampleCsvAppTests(unittest.TestCase):
    def test_sample_controls_are_next_to_the_uploader(self):
        at = AppTest.from_file(str(ROOT_SAMPLE.parent / "app.py"), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0, list(at.exception))
        self.assertEqual(len(at.file_uploader), 1)
        self.assertEqual(at.file_uploader[0].label, "Upload a CSV file")
        self.assertEqual(at.button[0].label, "Process")
        self.assertEqual([box.label for box in at.checkbox], ["Use sample data"])
        self.assertFalse(at.checkbox[0].value)
        self.assertIn("does not call OpenAI", at.checkbox[0].help)
        self.assertEqual(
            [button.label for button in at.download_button],
            ["Download sample CSV"],
        )
        self.assertIn(app.SAMPLE_CSV_FILENAME, at.download_button[0].help)

    def test_checking_sample_data_does_not_process(self):
        saved_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            at = AppTest.from_file(
                str(ROOT_SAMPLE.parent / "app.py"), default_timeout=30
            ).run()
            at.checkbox[0].check().run()
        finally:
            _restore("OPENAI_API_KEY", saved_key)

        self.assertEqual(len(at.exception), 0, list(at.exception))
        self.assertTrue(at.checkbox[0].value)
        self.assertEqual(len(at.success), 0)
        self.assertFalse(any("API Usage" in info.value for info in at.info))
        self.assertFalse(any("transformed" in success.value for success in at.success))
        self.assertNotIn(
            f"Using the bundled sample, {app.SAMPLE_CSV_FILENAME}.",
            [caption.value for caption in at.caption],
        )

    def test_process_uses_the_sample_when_no_file_is_uploaded(self):
        at = self._process(use_sample=True, upload=None)
        self.assertEqual(len(at.exception), 0, list(at.exception))
        preview = _preview(at)
        self.assertEqual(list(preview["Transaction_ID"]), [_sample_transaction_id()])
        self.assertIn(
            f"Using the bundled sample, {app.SAMPLE_CSV_FILENAME}.",
            [caption.value for caption in at.caption],
        )
        self.assertTrue(any("CSV transformed" in success.value for success in at.success))
        self.assertIn(
            "Download CSV",
            [button.label for button in at.download_button],
        )

    def test_uploaded_file_is_processed_when_sample_is_also_checked(self):
        upload = b"transaction_id,amount,reason_code\nUPLOADED-1,15.00,Fraud\n"
        at = self._process(use_sample=True, upload=upload)
        self.assertEqual(len(at.exception), 0, list(at.exception))
        preview = _preview(at)
        self.assertEqual(list(preview["Transaction_ID"]), ["UPLOADED-1"])
        self.assertNotIn(
            f"Using the bundled sample, {app.SAMPLE_CSV_FILENAME}.",
            [caption.value for caption in at.caption],
        )

    def _process(self, use_sample: bool, upload):
        saved_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key-not-used"
        try:
            at = AppTest.from_file(
                str(ROOT_SAMPLE.parent / "app.py"), default_timeout=30
            ).run()
            self.assertEqual(len(at.exception), 0, list(at.exception))
            at.number_input[0].set_value(1)
            if use_sample:
                at.checkbox[0].check()
            if upload is not None:
                at.file_uploader[0].upload("merchant.csv", upload, "text/csv")
            with mock.patch("openai.OpenAI", _OfflineClient), mock.patch(
                "time.sleep", return_value=None
            ):
                at.button[0].click().run()
            return at
        finally:
            _restore("OPENAI_API_KEY", saved_key)


def _sample_transaction_id() -> str:
    frame = transform_csv.load_merchant_csv(ROOT_SAMPLE)
    return str(frame.iloc[0]["Transaction_ID"])


def _preview(at):
    return next(
        element.value
        for element in at.dataframe
        if "Recommendation" in getattr(element.value, "columns", [])
    )


def _restore(name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


class _OfflineClient:
    """Stand-in that fails before any network call. The app already falls back."""

    def __init__(self, *args, **kwargs):
        pass

    @property
    def chat(self):
        raise RuntimeError("offline test client")
