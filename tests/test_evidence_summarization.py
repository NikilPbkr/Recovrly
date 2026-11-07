"""Evidence cells should become short text summaries.

Covers ``is_url_or_file`` and ``summarize_evidence`` in ``transform_csv.py``.
Blank cells become an empty string. Ordinary notes stay as written. Links and
known file types get a short prefix and keep the original value.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import unittest

import pandas as pd

from transform_csv import is_url_or_file, summarize_evidence


class SummarizeEvidenceTests(unittest.TestCase):
    def test_blank_values_become_empty_string(self):
        blanks = [None, "", "   ", "\n", "\t", float("nan"), pd.NA, pd.NaT]
        for raw in blanks:
            with self.subTest(raw=raw):
                self.assertEqual(summarize_evidence(raw), "")

    def test_plain_text_is_returned_unchanged(self):
        texts = [
            "Customer confirmed delivery",
            "1Z999",
            "signed receipt scan",
            "Cardholder emailed the store about the charge",
            "Tracking emailed via https://carrier.example/track",
            "Customer sent a pdf of the receipt",
        ]
        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(summarize_evidence(text), text)
                self.assertEqual(summarize_evidence(f"  {text}  "), text)

    def test_links_without_a_known_extension(self):
        links = [
            "https://example.com/evidence",
            "http://merchant.example/receipts/12345",
            "https://example.com/view?id=42",
            "https://example.com/download?type=pdf",
            "ftp://files.example.com/drop/evidence",
            "HTTPS://Example.com/Evidence/42",
        ]
        for link in links:
            with self.subTest(link=link):
                self.assertEqual(
                    summarize_evidence(link),
                    f"Evidence link provided: {link}",
                )
                self.assertEqual(
                    summarize_evidence(f"  {link}  "),
                    f"Evidence link provided: {link}",
                )

    def test_pdf_documents(self):
        values = [
            "https://cdn.example.com/receipt.pdf",
            "https://cdn.example.com/receipt.pdf?token=abc",
            "ftp://files.example.com/scan.pdf",
            "receipts/may/receipt.pdf",
            "receipt.PDF",
            "HTTPS://CDN.Example.com/Receipt.PDF",
        ]
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(
                    summarize_evidence(value),
                    f"PDF document provided: {value}",
                )
                self.assertEqual(
                    summarize_evidence(f"  {value}  "),
                    f"PDF document provided: {value}",
                )

    def test_image_evidence(self):
        values = [
            "widget.png",
            "widget.jpg",
            "widget.jpeg",
            "widget.gif",
            "https://cdn.example.com/widget.png",
            "https://cdn.example.com/widget.jpg",
            "https://cdn.example.com/widget.jpeg?w=200",
            "https://cdn.example.com/widget.gif",
            "photos/widget.PNG",
            "widget.JPEG",
            "widget.Gif",
        ]
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(
                    summarize_evidence(value),
                    f"Image evidence provided: {value}",
                )

    def test_word_documents(self):
        values = [
            "letter.doc",
            "letter.docx",
            "https://files.example.com/reply.docx",
            "correspondence/reply.DOC",
            "correspondence/reply.DOCX",
        ]
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(
                    summarize_evidence(value),
                    f"Document provided: {value}",
                )


class IsUrlOrFileTests(unittest.TestCase):
    def test_urls_and_known_extensions(self):
        values = [
            "http://example.com",
            "https://example.com/a",
            "ftp://files.example.com/a",
            "HTTP://EXAMPLE.COM",
            "  https://example.com  ",
            "notes.pdf",
            "photo.png",
            "photo.jpg",
            "photo.jpeg",
            "photo.gif",
            "letter.doc",
            "letter.docx",
            "Photo.JpEg",
            " folder/letter.DOCX ",
            "https://cdn.example.com/receipt.pdf",
            "https://cdn.example.com/widget.png",
        ]
        for value in values:
            with self.subTest(value=value):
                self.assertTrue(is_url_or_file(value))

    def test_ordinary_text_and_non_strings(self):
        values = [
            "Customer confirmed delivery",
            "1Z999",
            "signed receipt scan",
            "",
            "   ",
            "www.example.com/receipt",
            "notes.txt",
            "archive.zip",
            "http",
            "https",
            None,
            0,
            123,
            12.5,
            True,
            False,
            float("nan"),
            pd.NA,
        ]
        for value in values:
            with self.subTest(value=value):
                self.assertFalse(is_url_or_file(value))


if __name__ == "__main__":
    unittest.main()
