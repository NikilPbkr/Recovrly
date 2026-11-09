"""Fight / Don't Fight scoring for amount, reason code, and evidence.

Positive amounts keep the existing formula. Zero, negative, and non-finite
amounts add no penalty and no recovery: a negative amount used to make the
penalty negative, which raised the win rate and made Potential_Recovery
negative.

Run from the repo root:

    python -m unittest discover -s tests -t .
"""

import unittest
from unittest import mock

import pandas as pd

import app


class ReasonCodeRateTests(unittest.TestCase):
    def test_known_codes_and_unknown_codes(self):
        # Amount 0 isolates the base rate: no penalty, no recovery.
        cases = [
            ("Fraud", 0.35, "Don't Fight"),
            ("fraud", 0.35, "Don't Fight"),
            ("Fraud / Unauthorized Transaction", 0.35, "Don't Fight"),
            ("unauthorized", 0.30, "Don't Fight"),
            ("Unauthorized Transaction", 0.30, "Don't Fight"),
            ("Product Not Received", 0.55, "Fight"),
            ("item not received", 0.55, "Fight"),
            ("Product Unacceptable", 0.45, "Fight"),
            ("Product Not as Described", 0.45, "Fight"),
            ("Duplicate Processing", 0.70, "Fight"),
            ("Duplicate Charge", 0.70, "Fight"),
            ("duplicate", 0.70, "Fight"),
            ("Credit Not Processed", 0.60, "Fight"),
            ("credit", 0.60, "Fight"),
            ("Canceled Subscription", 0.55, "Fight"),
            ("subscription", 0.55, "Fight"),
            ("Unknown", 0.50, "Fight"),
            ("", 0.50, "Fight"),
            (None, 0.50, "Fight"),
        ]
        for reason, win_rate, recommendation in cases:
            with self.subTest(reason=reason):
                outcome = app.estimate_dispute_outcome(0, reason, [])
                self.assertEqual(outcome["win_rate"], win_rate)
                self.assertEqual(outcome["potential_recovery"], 0.0)
                self.assertEqual(outcome["recommendation"], recommendation)


class PositiveAmountTests(unittest.TestCase):
    def test_formula_penalty_cap_threshold_and_win_rate_cap(self):
        # Fraud, $100, two evidence items: 0.35 + 0.16 - 0.01 = 0.50.
        self.assertEqual(
            app.estimate_dispute_outcome(100, "Fraud", ["Shipping", "Receipt"]),
            {"win_rate": 0.50, "potential_recovery": 50.0, "recommendation": "Fight"},
        )

        # Unauthorized, $100, no evidence: 0.30 - 0.01 = 0.29.
        self.assertEqual(
            app.estimate_dispute_outcome(100, "unauthorized", []),
            {
                "win_rate": 0.29,
                "potential_recovery": 29.0,
                "recommendation": "Don't Fight",
            },
        )

        # $20,000 caps the penalty at 0.05: 0.55 + 0.16 - 0.05 = 0.66.
        self.assertEqual(
            app.estimate_dispute_outcome(
                20000, "Product Not Received", ["Shipping", "Receipt"]
            ),
            {
                "win_rate": 0.66,
                "potential_recovery": 13200.0,
                "recommendation": "Fight",
            },
        )

        # 0.40 is Fight. Fraud + one item at $300: 0.35 + 0.08 - 0.03 = 0.40.
        self.assertEqual(
            app.estimate_dispute_outcome(300, "Fraud", ["Receipt"]),
            {"win_rate": 0.40, "potential_recovery": 120.0, "recommendation": "Fight"},
        )

        # $355 rounds to 0.39, which is Don't Fight. 355 * 0.39 = 138.45.
        self.assertEqual(
            app.estimate_dispute_outcome(355, "Fraud", ["Receipt"]),
            {
                "win_rate": 0.39,
                "potential_recovery": 138.45,
                "recommendation": "Don't Fight",
            },
        )

        # Extra evidence stops at 0.95. It does not push recovery past the cap.
        many = ["Shipping"] * 10
        self.assertEqual(
            app.estimate_dispute_outcome(100, "duplicate", many),
            {"win_rate": 0.95, "potential_recovery": 95.0, "recommendation": "Fight"},
        )


class NonPositiveAmountTests(unittest.TestCase):
    def test_negative_amount_matches_zero_and_does_not_flip_to_fight(self):
        # Old result for -$5,000 of fraud and no evidence: penalty -0.50,
        # win rate 0.85, Fight, Potential_Recovery -4250.00.
        negative = app.estimate_dispute_outcome(-5000, "Fraud", [])
        zero = app.estimate_dispute_outcome(0, "Fraud", [])
        self.assertEqual(negative, zero)
        self.assertEqual(
            negative,
            {
                "win_rate": 0.35,
                "potential_recovery": 0.0,
                "recommendation": "Don't Fight",
            },
        )

    def test_large_negative_does_not_clamp_the_win_rate_to_the_cap(self):
        # Old result: penalty -1, raw rate 1.38, win rate 0.95, recovery -9500.
        negative = app.estimate_dispute_outcome(-10000, "unauthorized", ["Shipping"])
        zero = app.estimate_dispute_outcome(0, "unauthorized", ["Shipping"])
        self.assertEqual(negative, zero)
        self.assertEqual(
            negative,
            {
                "win_rate": 0.38,
                "potential_recovery": 0.0,
                "recommendation": "Don't Fight",
            },
        )

        # The same dispute with a positive amount still takes the 0.05 penalty.
        # 0.30 + 0.08 - 0.05 = 0.33, recovery 3300.00.
        self.assertEqual(
            app.estimate_dispute_outcome(10000, "unauthorized", ["Shipping"]),
            {
                "win_rate": 0.33,
                "potential_recovery": 3300.0,
                "recommendation": "Don't Fight",
            },
        )

    def test_non_finite_amounts_score_as_zero(self):
        zero = app.estimate_dispute_outcome(0, "duplicate", ["Shipping"])
        self.assertEqual(zero["win_rate"], 0.78)
        self.assertEqual(zero["recommendation"], "Fight")
        for amount in (
            float("nan"),
            float("inf"),
            float("-inf"),
            "nan",
            "inf",
            None,
            "not-a-number",
            "",
        ):
            with self.subTest(amount=amount):
                self.assertEqual(
                    app.estimate_dispute_outcome(amount, "duplicate", ["Shipping"]),
                    zero,
                )

    def test_safe_float_keeps_finite_numbers_including_negatives(self):
        self.assertEqual(app._safe_float(12.5), 12.5)
        self.assertEqual(app._safe_float(-3.25), -3.25)
        self.assertEqual(app._safe_float("19.99"), 19.99)
        self.assertEqual(app._safe_float(None), 0.0)
        self.assertEqual(app._safe_float(""), 0.0)
        self.assertEqual(app._safe_float("not-a-number"), 0.0)
        self.assertEqual(app._safe_float(float("nan")), 0.0)
        self.assertEqual(app._safe_float(float("inf")), 0.0)
        self.assertEqual(app._safe_float(float("-inf")), 0.0)
        self.assertEqual(app._safe_float("nan"), 0.0)
        self.assertEqual(app._safe_float("inf"), 0.0)


class ProcessDisputesScoringTests(unittest.TestCase):
    def test_processed_rows_use_the_amount_rules(self):
        frame = pd.DataFrame(
            [
                _dispute("T-pos", 100.0, "Fraud"),
                _dispute("T-zero", 0.0, "duplicate"),
                _dispute("T-neg", -5000.0, "Fraud"),
                _dispute("T-nan", float("nan"), "Product Not Received"),
                _dispute("T-inf", float("inf"), "unauthorized"),
            ]
        )
        evidence = [
            ["Shipping", "Receipt"],
            ["Shipping"],
            [],
            ["Receipt", "Email"],
            [],
        ]
        with mock.patch.object(app, "batch_rank_evidence", return_value=evidence), \
                mock.patch.object(
                    app, "batch_generate_arguments", return_value=["arg"] * 5
                ), \
                mock.patch.object(app.time, "sleep"):
            result = app.process_disputes(frame, len(frame), batch_size=len(frame))

        self.assertEqual(list(result["Transaction_ID"]), list(frame["Transaction_ID"]))
        for index, items in enumerate(evidence):
            outcome = app.estimate_dispute_outcome(
                frame.loc[index, "Amount"], frame.loc[index, "Reason_Code"], items
            )
            self.assertEqual(result.loc[index, "Recommendation"], outcome["recommendation"])
            self.assertEqual(result.loc[index, "Simulated_Win_Rate"], outcome["win_rate"])
            self.assertEqual(
                result.loc[index, "Potential_Recovery"], outcome["potential_recovery"]
            )

        negative = result.loc[result["Transaction_ID"] == "T-neg"].iloc[0]
        self.assertEqual(negative["Recommendation"], "Don't Fight")
        self.assertEqual(negative["Simulated_Win_Rate"], 0.35)
        self.assertEqual(negative["Potential_Recovery"], 0.0)
        self.assertFalse(result["Simulated_Win_Rate"].isna().any())
        self.assertFalse(result["Potential_Recovery"].isna().any())

        summary = app.results_summary(result)
        self.assertEqual(summary["total_potential_recovery"], 50.0)
        # $100 fraud, $0 duplicate, and the missing amount (scored as $0
        # product-not-received) are Fight. The negative and infinite rows
        # are Don't Fight, and neither one subtracts from the total.
        self.assertEqual(summary["fight"], 3)
        self.assertEqual(summary["dont_fight"], 2)

    def test_accounting_negative_from_the_csv_is_not_a_recovery(self):
        raw = pd.DataFrame(
            {
                "transaction_id": ["T1"],
                "amount": ["($5,000.00)"],
                "reason_code": ["Fraud"],
            }
        )
        standardized = app.transform_csv(raw)
        self.assertAlmostEqual(float(standardized.loc[0, "Amount"]), -5000.0)

        with mock.patch.object(app, "batch_rank_evidence", return_value=[[]]), \
                mock.patch.object(app, "batch_generate_arguments", return_value=["arg"]), \
                mock.patch.object(app.time, "sleep"):
            result = app.process_disputes(standardized, 1, batch_size=1)

        self.assertEqual(result.loc[0, "Reason_Code"], "Fraud")
        self.assertEqual(result.loc[0, "Simulated_Win_Rate"], 0.35)
        self.assertEqual(result.loc[0, "Potential_Recovery"], 0.0)
        self.assertEqual(result.loc[0, "Recommendation"], "Don't Fight")
        self.assertEqual(app.results_summary(result)["total_potential_recovery"], 0.0)


def _dispute(transaction_id, amount, reason):
    return {
        "Transaction_ID": transaction_id,
        "Product": "Widget",
        "Amount": amount,
        "Reason_Code": reason,
        "Evidence_Receipt": "",
        "Evidence_Shipping": "",
        "Evidence_Email": "",
        "Evidence_Photo": "",
    }
