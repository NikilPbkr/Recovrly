#!/usr/bin/env python3
"""
CSV Transformation Tool for Chargeback Dispute Data
----------------------------------------------------
Transforms any merchant CSV into standardized Stripe-style format.

Features:
- Fuzzy column name matching
- Automatic placeholder filling for missing fields
- Evidence text summarization for PDF/image links
- Clean, standardized output ready for Recoverly

Usage:
    python transform_csv.py input.csv [output.csv]
"""

import pandas as pd
import sys
import os
from datetime import datetime
from difflib import get_close_matches
from typing import Dict, Any, Optional
import re

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
    "Notes": ["notes", "comments", "remarks", "additional_info", "description"],
}

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


def transform_csv(input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Transform merchant CSV into standardized Stripe-style format.
    
    Args:
        input_path: Path to input CSV file
        output_path: Optional path for output CSV (if None, uses input_standardized.csv)
    
    Returns:
        Transformed DataFrame
    """
    print(f"📂 Loading CSV: {input_path}")
    
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
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
    
    for std_col in REQUIRED_COLUMNS:
        if std_col in column_mapping.values():
            # Find the original column that maps to this standard column
            orig_col = [k for k, v in column_mapping.items() if v == std_col][0]
            
            # Special handling for evidence columns
            if std_col.startswith("Evidence_"):
                standardized_data[std_col] = df[orig_col].apply(summarize_evidence)
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
    
    # Fill any remaining NaN values with appropriate placeholders
    for col in REQUIRED_COLUMNS:
        if col in result_df.columns:
            default_val = DEFAULTS[col]
            if not callable(default_val):
                result_df[col].fillna(default_val, inplace=True)
    
    # Ensure correct column order
    result_df = result_df[REQUIRED_COLUMNS]
    
    # Save to output
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_standardized.csv"
    
    result_df.to_csv(output_path, index=False)
    print(f"\n✅ Transformation complete!")
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
