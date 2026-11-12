#!/usr/bin/env python3
"""
CSV Transformation Tool for Chargeback Dispute Data
----------------------------------------------------
Transforms any merchant CSV into standardized Stripe-style format.

Features:
- Fuzzy column name matching
- Automatic placeholder filling for missing fields
- Amount values parsed from currency-formatted merchant exports
- Date values normalized to YYYY-MM-DD
- Evidence text summarization for PDF/image links
- Clear messages for empty, header-only, and unreadable files
- Duplicate transaction IDs called out when running the CLI
- Clean, standardized output ready for Recoverly

Usage:
    python transform_csv.py input.csv [output.csv]
"""

import pandas as pd
import sys
import os
import io
from datetime import date, datetime
from difflib import get_close_matches
from typing import Dict, Any, Optional
import re

class CsvInputError(Exception):
    """A merchant CSV that cannot be transformed.

    ``str(error)`` is safe to show to the person who uploaded the file.
    """


EMPTY_CSV_MESSAGE = (
    "This CSV is empty. It needs a header row and at least one dispute."
)
HEADER_ONLY_CSV_MESSAGE = (
    "This CSV has a header row but no data rows. Add at least one dispute and try again."
)
NOT_UTF8_MESSAGE = (
    "This file is not valid UTF-8. Save the CSV as UTF-8 and try again."
)
MALFORMED_CSV_MESSAGE = (
    "This file could not be read as a CSV. "
    "Check that it is comma-separated, has a header row, and that any quotes are closed."
)


# Standard column names required by Recoverly
REQUIRED_COLUMNS = [
    "Transaction_ID",
    "Customer_Name",
    "Customer_Email",
    "Date",
    "Amount",
    "Product",
    "Reason_Code",
    "Evidence_Receipt",
    "Evidence_Shipping",
    "Evidence_Email",
    "Evidence_Photo",
    "Current_Process",
    "Notes",
]

# Fuzzy matching mappings for common variations
COLUMN_ALIASES = {
    "Transaction_ID": ["transaction_id", "txn_id", "trans_id", "id", "order_id", "payment_id", "charge_id"],
    "Customer_Name": ["customer_name", "name", "customer", "buyer_name", "client_name", "cardholder_name"],
    "Customer_Email": ["customer_email", "email", "buyer_email", "client_email", "contact_email"],
    "Date": ["date", "transaction_date", "txn_date", "created", "timestamp", "created_at", "order_date"],
    "Amount": ["amount", "total", "charge_amount", "transaction_amount", "price", "value"],
    "Product": ["product", "description", "item", "product_name", "service", "item_description"],
    "Reason_Code": ["reason_code", "reason", "dispute_reason", "chargeback_reason", "code"],
    "Evidence_Receipt": ["evidence_receipt", "receipt", "receipt_url", "receipt_link", "proof_of_purchase"],
    "Evidence_Shipping": ["evidence_shipping", "shipping", "tracking", "shipping_proof", "delivery_confirmation"],
    "Evidence_Email": ["evidence_email", "email_proof", "correspondence", "communication"],
    "Evidence_Photo": ["evidence_photo", "photo", "image", "product_image", "photo_proof"],
    "Current_Process": ["current_process", "status", "dispute_status", "process", "stage"],
    "Notes": ["notes", "comments", "remarks", "additional_info"],
}

# Currency codes stripped from amount text. Matched as whole words so a code
# does not eat the start of an unrelated word ("USED" stays "USED").
_CURRENCY_CODES = (
    "USD",
    "EUR",
    "GBP",
    "CAD",
    "AUD",
    "NZD",
    "CHF",
    "JPY",
    "CNY",
    "INR",
    "HKD",
    "SGD",
    "MXN",
    "BRL",
)
_CURRENCY_TOKENS = re.compile(
    r"(?<![A-Za-z])(?:US\$|" + "|".join(_CURRENCY_CODES) + r")(?![A-Za-z])|[$€£¥]",
    re.IGNORECASE,
)
_THOUSANDS_AMOUNT = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_PLAIN_AMOUNT = re.compile(r"(?:\d+(?:\.\d+)?|\.\d+)")


# Default placeholder values
DEFAULTS = {
    "Transaction_ID": lambda idx: f"TXN_{idx:06d}",
    "Customer_Name": "Unknown",
    "Customer_Email": "unknown@example.com",
    "Date": datetime.now().strftime("%Y-%m-%d"),
    "Amount": 0.0,
    "Product": "Unknown Product",
    "Reason_Code": "Unknown",
    "Evidence_Receipt": "No receipt provided",
    "Evidence_Shipping": "No shipping evidence provided",
    "Evidence_Email": "No email evidence provided",
    "Evidence_Photo": "No photo evidence provided",
    "Current_Process": "Pending",
    "Notes": "No notes provided",
}


def fuzzy_match_column(input_col: str, standard_cols: Dict[str, list]) -> Optional[str]:
    """
    Match input column name to standard column using fuzzy matching.
    
    Args:
        input_col: Column name from input CSV
        standard_cols: Dictionary of standard names with their aliases
    
    Returns:
        Standard column name or None if no match
    """
    input_col_lower = input_col.lower().strip()
    
    # Direct match first
    for std_col, aliases in standard_cols.items():
        if input_col_lower in [a.lower() for a in aliases]:
            return std_col
    
    # Fuzzy match using difflib
    all_possible = []
    for std_col, aliases in standard_cols.items():
        all_possible.extend([(alias, std_col) for alias in aliases])
    
    alias_names = [a[0] for a in all_possible]
    matches = get_close_matches(input_col_lower, alias_names, n=1, cutoff=0.6)
    
    if matches:
        matched_alias = matches[0]
        for alias, std_col in all_possible:
            if alias == matched_alias:
                return std_col
    
    return None


def is_url_or_file(text: str) -> bool:
    """Check if text is a URL or file path."""
    if not isinstance(text, str):
        return False
    text_lower = text.lower().strip()
    return (
        text_lower.startswith(('http://', 'https://', 'ftp://')) or
        text_lower.endswith(('.pdf', '.png', '.jpg', '.jpeg', '.gif', '.doc', '.docx'))
    )


def summarize_evidence(value: Any) -> str:
    """Convert evidence values (including URLs/files) into text summaries."""
    if pd.isna(value) or value == "" or value is None:
        return ""
    
    value_str = str(value).strip()
    
    # If it's a URL or file, create a summary
    if is_url_or_file(value_str):
        # Extract file type
        if '.pdf' in value_str.lower():
            return f"PDF document provided: {value_str}"
        elif any(ext in value_str.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif']):
            return f"Image evidence provided: {value_str}"
        elif any(ext in value_str.lower() for ext in ['.doc', '.docx']):
            return f"Document provided: {value_str}"
        else:
            return f"Evidence link provided: {value_str}"
    
    # Otherwise return as-is (already text)
    return value_str


def _wrapped_in_parens(text: str) -> bool:
    return len(text) >= 2 and text.startswith("(") and text.endswith(")")


def parse_amount(value: Any) -> float:
    """Parse one merchant amount into a float.

    Plain numbers are returned unchanged. Currency symbols (``$€£¥``), common
    currency codes, thousands commas, and surrounding whitespace are ignored.
    A leading or trailing minus, or accounting parentheses such as ``(12.00)``,
    marks a negative amount. Blank or unparseable values return NaN so the
    placeholder fill can replace them.
    """
    if isinstance(value, bool) or value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float("nan") if pd.isna(value) else float(value)
    try:
        if pd.isna(value):
            return float("nan")
    except TypeError:
        return float("nan")

    text = str(value).replace("\u00a0", " ").replace("\u202f", " ").strip()
    if not text:
        return float("nan")

    negative = False
    if _wrapped_in_parens(text):
        negative = True
        text = text[1:-1].strip()

    text = " ".join(_CURRENCY_TOKENS.sub(" ", text).split())
    if _wrapped_in_parens(text):
        negative = True
        text = text[1:-1].strip()

    if text.endswith(("-", "−")):
        negative = True
        text = text[:-1].strip()
    if text.startswith("+"):
        text = text[1:].strip()
    elif text.startswith(("-", "−")):
        negative = True
        text = text[1:].strip()

    if "," in text:
        if not _THOUSANDS_AMOUNT.fullmatch(text):
            return float("nan")
        text = text.replace(",", "")
    elif not _PLAIN_AMOUNT.fullmatch(text):
        return float("nan")

    return -float(text) if negative else float(text)


def normalize_amounts(series: pd.Series) -> pd.Series:
    """Parse amount text. An already-numeric column is left unchanged."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series.map(parse_amount), errors="coerce")


# Month names are English on purpose. datetime's %b/%B follow the process
# locale, so a French locale would reject "Jan". Numeric formats are tried in
# this fixed order; each string matches at most one of them. Slash and dash
# dates are US month-first ("02/01/2024" is February 1, not January 2).
# Two-digit years use datetime's %y window (00-68 → 2000-2068, 69-99 → 1969-1999).
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
# A real ISO date, or the same date with a time we ignore. The clock and
# offset are not interpreted, so the calendar date cannot shift.
_ISO_DATE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})"
    r"(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$",
    re.IGNORECASE,
)
_NAMED_MDY = re.compile(
    r"^([A-Za-z]+)\s+(\d{1,2})(?:\s*,\s*|\s+)(\d{2,4})$"
)
_NAMED_DMY = re.compile(
    r"^(\d{1,2})[-\s]+([A-Za-z]+)[-\s]+(\d{2,4})$"
)
_NUMERIC_FORMATS = (
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%Y/%m/%d",
    "%Y-%m-%d",
    "%m/%d/%y",
    "%m-%d-%y",
)


def _format_ymd(year: int, month: int, day: int) -> Optional[str]:
    """Return YYYY-MM-DD, or None when the calendar date does not exist."""
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OverflowError):
        return None


def _expand_year(token: str) -> Optional[int]:
    if len(token) == 4:
        return int(token)
    if len(token) == 2:
        return datetime.strptime(token, "%y").year
    return None


def _parse_iso(text: str) -> Optional[str]:
    match = _ISO_DATE.fullmatch(text)
    if not match:
        return None
    formatted = _format_ymd(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    if formatted is None:
        return None
    # An exact YYYY-MM-DD value is returned as stored.
    if len(text) == 10:
        return text
    return formatted


def _parse_named(text: str) -> Optional[str]:
    month_name = day = year = None
    mdy = _NAMED_MDY.fullmatch(text)
    if mdy:
        month_name, day, year = mdy.group(1), mdy.group(2), mdy.group(3)
    else:
        dmy = _NAMED_DMY.fullmatch(text)
        if dmy:
            day, month_name, year = dmy.group(1), dmy.group(2), dmy.group(3)
    if month_name is None or day is None or year is None:
        return None
    month = _MONTHS.get(month_name.lower())
    expanded = _expand_year(year)
    if month is None or expanded is None:
        return None
    return _format_ymd(expanded, month, int(day))


def _parse_numeric(text: str) -> Optional[str]:
    for fmt in _NUMERIC_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return parsed.strftime("%Y-%m-%d")
    return None


def parse_date(value: Any) -> Optional[str]:
    """Parse one merchant date into ``YYYY-MM-DD``.

    Accepts US numeric dates (``01/15/2024``, ``1/15/24``, ``01-15-2024``),
    year-first numbers (``2024/01/15``), ISO dates, English month names
    (``Jan 15, 2024``, ``15-Jan-2024``), and ISO datetimes. A datetime keeps
    its calendar date and drops the time. ``datetime`` and ``date`` values
    are formatted the same way.

    Blank or unparseable values return None so the existing Date placeholder
    can fill them. A numeric string that cannot be a month, such as
    ``15/01/2024``, is unparseable rather than swapped into day-first order.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    if isinstance(value, (datetime, date)):
        return _format_ymd(value.year, value.month, value.day)
    if isinstance(value, (int, float)):
        return None

    text = str(value).replace("\u00a0", " ").replace("\u202f", " ").strip()
    if not text:
        return None

    for parser in (_parse_iso, _parse_named, _parse_numeric):
        parsed = parser(text)
        if parsed is not None:
            return parsed
    return None


def normalize_dates(series: pd.Series) -> pd.Series:
    """Format a Date column as ``YYYY-MM-DD`` strings.

    A datetime column uses its calendar date. Text is parsed cell by cell.
    Blank and unparseable cells stay missing so placeholder fill can replace
    them. An exact ``YYYY-MM-DD`` value is left unchanged.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.strftime("%Y-%m-%d")
    return series.map(parse_date)


def _source_bytes(source: Any) -> bytes:
    """Read a filesystem path or an uploaded file as bytes."""
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as handle:
            return handle.read()

    getvalue = getattr(source, "getvalue", None)
    if callable(getvalue):
        data = getvalue()
    else:
        read = getattr(source, "read", None)
        if not callable(read):
            raise TypeError("CSV source must be a path or a binary file")
        data = read()

    if data is None:
        return b""
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


def load_merchant_csv(source: Any) -> pd.DataFrame:
    """Read a merchant CSV from a path or an uploaded file.

    ``source`` is a filesystem path or a binary file-like object, such as a
    Streamlit upload. A UTF-8 BOM is accepted. Raises ``CsvInputError`` with
    a short message when the file is empty, has a header but no rows, is not
    UTF-8, or is not a readable CSV. Callers show that message instead of a
    traceback.

    A file whose headers match nothing is still returned. The transformer
    fills those fields with placeholders, which is the existing contract.
    """
    try:
        raw = _source_bytes(source)
    except FileNotFoundError:
        raise
    except OSError as exc:
        detail = exc.strerror or "unknown error"
        raise CsvInputError(f"This file could not be read ({detail}).") from None

    if not raw.strip(b" \t\r\n\x0b\x0c"):
        raise CsvInputError(EMPTY_CSV_MESSAGE)

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise CsvInputError(NOT_UTF8_MESSAGE) from None

    if not text.strip():
        raise CsvInputError(EMPTY_CSV_MESSAGE)

    try:
        frame = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError:
        raise CsvInputError(EMPTY_CSV_MESSAGE) from None
    except (pd.errors.ParserError, UnicodeError):
        raise CsvInputError(MALFORMED_CSV_MESSAGE) from None

    if len(frame) == 0:
        raise CsvInputError(HEADER_ONLY_CSV_MESSAGE)
    return frame


def _missing_transaction_id(value: Any) -> bool:
    """True when a cell has no merchant transaction ID.

    Empty and whitespace-only text counts as missing. Pandas also reads a
    blank CSV cell as NaN, and that is missing too.
    """
    if isinstance(value, str):
        return value.strip() == ""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _transaction_id_label(value: Any) -> str:
    """Text used to compare and report one real transaction ID.

    A whole-number float is shown without a trailing ``.0``. Pandas stores
    an ID column as float when another cell in that column is blank, so a
    merchant ID of ``1001`` would otherwise be reported as ``1001.0``.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _placeholder_transaction_rows(source: Optional[pd.Series], row_count: int) -> set:
    """Row positions whose Transaction_ID did not come from the merchant.

    A missing column means every output ID is a generated placeholder.
    A blank cell in a present column is not a real ID either. Those rows
    must not be compared, or the fills would look like duplicates.
    """
    if source is None:
        return set(range(row_count))
    return {
        index
        for index, value in enumerate(source.tolist())
        if _missing_transaction_id(value)
    }


def find_duplicate_transaction_ids(
    ids: Any,
    placeholder_rows: Optional[Any] = None,
) -> list:
    """Return real transaction IDs that appear on more than one row.

    ``ids`` is a Transaction_ID column in file order. ``placeholder_rows``
    holds positions to ignore: every row when the ID column was missing, and
    blank cells when it was present. Generated placeholders such as
    ``TXN_000000`` are ignored even when every row was given the same fill.
    A merchant-supplied ID is still reported when it is repeated, including
    an ID that happens to use that placeholder spelling.

    Each item is ``(transaction_id, line_numbers)``. Line numbers count the
    header as line 1, matching the merchant file. IDs are in first-seen order.
    """
    skipped = set(placeholder_rows or ())
    rows_for_id: Dict[str, list] = {}
    order = []
    for index, value in enumerate(list(ids)):
        if index in skipped or _missing_transaction_id(value):
            continue
        label = _transaction_id_label(value)
        if not label:
            continue
        if label not in rows_for_id:
            rows_for_id[label] = []
            order.append(label)
        # Header is line 1, so the first data row is line 2.
        rows_for_id[label].append(index + 2)
    return [
        (label, rows_for_id[label])
        for label in order
        if len(rows_for_id[label]) > 1
    ]


def format_duplicate_transaction_ids(duplicates: list) -> str:
    """CLI text naming each repeated ID, how often it appears, and its rows."""
    lines = [f"  ⚠️  Duplicate Transaction_IDs ({len(duplicates)}):"]
    for txn_id, rows in duplicates:
        row_text = ", ".join(str(row) for row in rows)
        lines.append(
            f"      {txn_id} is repeated {len(rows)} times (lines {row_text})"
        )
    return "\n".join(lines)


def transform_csv(input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Transform merchant CSV into standardized Stripe-style format.
    
    Args:
        input_path: Path to input CSV file
        output_path: Optional path for output CSV (if None, uses input_standardized.csv)
    
    Returns:
        Transformed DataFrame

    Exits with status 1, and no traceback, when the file is missing, empty,
    header-only, not UTF-8, or not a readable CSV.

    When a real Transaction_ID appears on more than one row, prints those IDs
    before the completion line. Blank cells and generated placeholders are
    not counted. Repeated rows are still written to the output.
    """
    print(f"📂 Loading CSV: {input_path}")

    try:
        df = load_merchant_csv(input_path)
    except CsvInputError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ Error: File '{input_path}' not found")
        sys.exit(1)
    
    print(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"\nOriginal columns: {list(df.columns)}\n")
    
    # Create column mapping
    column_mapping = {}
    for input_col in df.columns:
        matched = fuzzy_match_column(input_col, COLUMN_ALIASES)
        if matched:
            column_mapping[input_col] = matched
            print(f"  ✓ Mapped '{input_col}' → '{matched}'")
        else:
            print(f"  ⚠️  No match for '{input_col}' (will be ignored)")
    
    # Create new standardized DataFrame
    standardized_data = {}
    # None means the merchant file had no transaction-ID column, so every
    # output ID is a generated placeholder rather than a real ID.
    transaction_source = None
    
    for std_col in REQUIRED_COLUMNS:
        if std_col in column_mapping.values():
            # Find the original column that maps to this standard column
            orig_col = [k for k, v in column_mapping.items() if v == std_col][0]
            if std_col == "Transaction_ID":
                transaction_source = df[orig_col]
            
            # Evidence links are summarized. Amounts and dates are normalized.
            if std_col.startswith("Evidence_"):
                standardized_data[std_col] = df[orig_col].apply(summarize_evidence)
            elif std_col == "Amount":
                standardized_data[std_col] = normalize_amounts(df[orig_col])
            elif std_col == "Date":
                standardized_data[std_col] = normalize_dates(df[orig_col])
            else:
                standardized_data[std_col] = df[orig_col]
        else:
            # Column not found, use default
            default_val = DEFAULTS[std_col]
            if callable(default_val):
                # For Transaction_ID, generate unique IDs
                standardized_data[std_col] = [default_val(i) for i in range(len(df))]
            else:
                standardized_data[std_col] = [default_val] * len(df)
            
            print(f"  ℹ️  '{std_col}' missing, using placeholder")
    
    result_df = pd.DataFrame(standardized_data)
    
    # Fill any remaining NaN values with appropriate placeholders.
    # Assign the result back. Series.fillna(inplace=True) on a column updates
    # a copy under pandas Copy-on-Write and leaves the missing values in place.
    for col in REQUIRED_COLUMNS:
        if col in result_df.columns:
            default_val = DEFAULTS[col]
            if not callable(default_val):
                result_df[col] = result_df[col].fillna(default_val)
    
    # Ensure correct column order
    result_df = result_df[REQUIRED_COLUMNS]

    # Report repeated merchant IDs. Rows are kept either way; this only
    # tells the CLI user which real IDs were copied more than once.
    duplicates = find_duplicate_transaction_ids(
        result_df["Transaction_ID"],
        placeholder_rows=_placeholder_transaction_rows(
            transaction_source, len(result_df)
        ),
    )
    if duplicates:
        print()
        print(format_duplicate_transaction_ids(duplicates))
    
    # Save to output
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_standardized.csv"
    
    result_df.to_csv(output_path, index=False)
    print("\n✅ Transformation complete!")
    print(f"📄 Output saved to: {output_path}")
    print(f"📊 Total rows: {len(result_df)}")
    
    return result_df


def main():
    if len(sys.argv) < 2:
        print("Usage: python transform_csv.py input.csv [output.csv]")
        print("\nExample:")
        print("  python transform_csv.py merchant_data.csv")
        print("  python transform_csv.py merchant_data.csv standardized_output.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' not found")
        sys.exit(1)
    
    transform_csv(input_file, output_file)


if __name__ == "__main__":
    main()
