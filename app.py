#!/usr/bin/env python3
"""
AI Chargeback Dispute Optimizer (Streamlit)
- Single-page Streamlit app that wraps the existing chargeback AI logic
- Run with: streamlit run app.py
- Optimized for cost and token efficiency with batching, caching, and structured outputs
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from difflib import get_close_matches

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------
# App & environment setup
# -------------------------
st.set_page_config(page_title="AI Chargeback Dispute Optimizer")
st.title("AI Chargeback Dispute Optimizer")

load_dotenv()  # Load OPENAI_API_KEY from .env if present
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("OPENAI_API_KEY not found. Add it to a .env file in this folder or export it in your shell.")

client = OpenAI()
MODEL = os.getenv("CHARGEBACK_MODEL", "gpt-3.5-turbo")

# Cache for storing repeated AI responses
if 'ai_cache' not in st.session_state:
    st.session_state.ai_cache = {}

# Token usage tracking
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0

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

# Fuzzy matching mappings for CSV transformation
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

# -------------------------
# Helpers
# -------------------------

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def fuzzy_match_column(input_col: str, standard_cols: Dict[str, list]) -> Optional[str]:
    """Match input column name to standard column using fuzzy matching."""
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
        if '.pdf' in value_str.lower():
            return f"PDF document provided: {value_str}"
        elif any(ext in value_str.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif']):
            return f"Image evidence provided: {value_str}"
        elif any(ext in value_str.lower() for ext in ['.doc', '.docx']):
            return f"Document provided: {value_str}"
        else:
            return f"Evidence link provided: {value_str}"
    
    return value_str


def transform_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Transform any CSV into standardized format with fuzzy matching and placeholders."""
    # Create column mapping
    column_mapping = {}
    for input_col in df.columns:
        matched = fuzzy_match_column(input_col, COLUMN_ALIASES)
        if matched:
            column_mapping[input_col] = matched
    
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
                standardized_data[std_col] = [default_val(i) for i in range(len(df))]
            else:
                standardized_data[std_col] = [default_val] * len(df)
    
    result_df = pd.DataFrame(standardized_data)
    
    # Fill any remaining NaN values with appropriate placeholders
    for col in REQUIRED_COLUMNS:
        if col in result_df.columns:
            default_val = DEFAULTS[col]
            if not callable(default_val):
                result_df[col].fillna(default_val, inplace=True)
    
    # Ensure correct column order
    result_df = result_df[REQUIRED_COLUMNS]
    
    return result_df


def _generate_cache_key(prompt: str) -> str:
    """Generate a cache key from prompt content."""
    return hashlib.md5(prompt.encode()).hexdigest()


def batch_rank_evidence(disputes: List[Dict[str, Any]]) -> List[List[str]]:
    """Batch process evidence ranking for multiple disputes in a single API call.
    Returns list of evidence rankings in same order as input disputes.
    """
    # Build batch prompt
    batch_prompt = "You are an expert in chargeback dispute resolution. Rank evidence for each dispute.\n\n"
    
    for idx, dispute in enumerate(disputes):
        batch_prompt += f"""Dispute {idx + 1}:
Transaction: {dispute['product']}, Amount: ${dispute['amount']}, Reason: {dispute['reason']}
Evidence:
- Receipt: {dispute['evidence_receipt']}
- Shipping: {dispute['evidence_shipping']}
- Email: {dispute['evidence_email']}
- Photo: {dispute['evidence_photo']}

"""
    
    batch_prompt += """For each dispute, rank the evidence from most to least likely to win.
Output ONLY valid JSON in this exact format:
{
  "rankings": [
    {"dispute_id": 1, "evidence_ranking": ["Shipping", "Receipt", "Email", "Photo"]},
    {"dispute_id": 2, "evidence_ranking": [...]}
  ]
}"""
    
    # Check cache
    cache_key = _generate_cache_key(batch_prompt)
    if cache_key in st.session_state.ai_cache:
        return st.session_state.ai_cache[cache_key]
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a chargeback dispute AI. Output only valid JSON."},
                {"role": "user", "content": batch_prompt}
            ],
            temperature=0,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        # Track tokens
        st.session_state.total_tokens += response.usage.total_tokens
        
        result = json.loads(response.choices[0].message.content)
        rankings = [r["evidence_ranking"][:2] for r in result["rankings"]]
        
        # Ensure we have the right number of results
        if len(rankings) != len(disputes):
            st.warning(f"Expected {len(disputes)} rankings but got {len(rankings)}. Padding with defaults.")
            while len(rankings) < len(disputes):
                rankings.append(["Shipping", "Receipt"])
        
        # Cache result
        st.session_state.ai_cache[cache_key] = rankings
        return rankings
        
    except Exception as e:
        st.warning(f"Batch ranking failed: {e}. Using fallback.")
        # Fallback: return default for all
        return [["Shipping", "Receipt"] for _ in disputes]


def batch_generate_arguments(disputes: List[Dict[str, Any]], evidence_lists: List[List[str]]) -> List[str]:
    """Batch process argument generation for multiple disputes in a single API call.
    Returns list of arguments in same order as input disputes.
    """
    # Build batch prompt
    batch_prompt = "You are an expert chargeback dispute writer. Generate concise arguments (max 150 words each).\n\n"
    
    for idx, (dispute, top_evidence) in enumerate(zip(disputes, evidence_lists)):
        batch_prompt += f"""Dispute {idx + 1}:
Top Evidence: {top_evidence}
Product: {dispute['product']}, Amount: ${dispute['amount']}, Reason: {dispute['reason']}

"""
    
    batch_prompt += """For each dispute, generate a persuasive argument (max 150 words) focusing on clarity and relevance to the reason code.
Output ONLY valid JSON in this exact format:
{
  "arguments": [
    {"dispute_id": 1, "argument": "text here"},
    {"dispute_id": 2, "argument": "text here"}
  ]
}"""
    
    # Check cache
    cache_key = _generate_cache_key(batch_prompt)
    if cache_key in st.session_state.ai_cache:
        return st.session_state.ai_cache[cache_key]
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a chargeback argument AI. Output only valid JSON. Keep arguments under 150 words."},
                {"role": "user", "content": batch_prompt}
            ],
            temperature=0.5,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        # Track tokens
        st.session_state.total_tokens += response.usage.total_tokens
        
        result = json.loads(response.choices[0].message.content)
        arguments = [a["argument"] for a in result["arguments"]]
        
        # Ensure we have the right number of results
        if len(arguments) != len(disputes):
            st.warning(f"Expected {len(disputes)} arguments but got {len(arguments)}. Padding with defaults.")
            while len(arguments) < len(disputes):
                arguments.append("Argument generation incomplete.")
        
        # Cache result
        st.session_state.ai_cache[cache_key] = arguments
        return arguments
        
    except Exception as e:
        st.warning(f"Batch argument generation failed: {e}. Using fallback.")
        # Fallback: return generic message
        return [f"Dispute argument generation failed." for _ in disputes]


def process_disputes(df: pd.DataFrame, n_rows: int, batch_size: int = 2) -> pd.DataFrame:
    """Process disputes with batched API calls for efficiency.
    
    Args:
        df: DataFrame with dispute data (already standardized)
        n_rows: Number of rows to process
        batch_size: Number of disputes to process per API call
    """
    # Reset token counter for this run
    st.session_state.total_tokens = 0
    
    rows = df.head(n_rows).copy()
    results = []

    # Prepare all disputes for batching
    all_disputes = []
    for idx, row in rows.iterrows():
        all_disputes.append({
            "transaction_id": row.get("Transaction_ID", ""),
            "product": row.get("Product", ""),
            "amount": _safe_float(row.get("Amount", 0)),
            "reason": row.get("Reason_Code", ""),
            "evidence_receipt": row.get("Evidence_Receipt", ""),
            "evidence_shipping": row.get("Evidence_Shipping", ""),
            "evidence_email": row.get("Evidence_Email", ""),
            "evidence_photo": row.get("Evidence_Photo", ""),
        })
    
    # Process in batches
    for i in range(0, len(all_disputes), batch_size):
        batch = all_disputes[i:i + batch_size]
        
        try:
            # Batch API call for evidence ranking
            evidence_lists = batch_rank_evidence(batch)
            
            # Small delay between batches
            time.sleep(0.5)
            
            # Batch API call for argument generation
            arguments = batch_generate_arguments(batch, evidence_lists)
        except Exception as e:
            st.warning(f"Batch {i//batch_size + 1} processing error: {e}. Using fallback for this batch.")
            # Fallback for failed batch
            evidence_lists = [["Shipping", "Receipt"] for _ in batch]
            arguments = ["Dispute argument could not be generated due to processing error." for _ in batch]
        
        # Calculate metrics for each dispute in batch
        for dispute, top_evidence, argument in zip(batch, evidence_lists, arguments):
            reason_code_weights = {
                "Fraud": 0.35,
                "fraud": 0.35,
                "unauthorized": 0.30,
                "Product Not Received": 0.55,
                "not received": 0.55,
                "Product Unacceptable": 0.45,
                "not as described": 0.45,
                "Duplicate Processing": 0.70,
                "duplicate": 0.70,
                "Credit Not Processed": 0.60,
                "credit": 0.60,
                "Unauthorized Transaction": 0.30,
                "canceled": 0.55,
                "subscription": 0.55,
            }
            
            # Flexible matching for reason codes
            reason_str = str(dispute["reason"]).lower()
            base_rate = 0.50  # default
            for key, rate in reason_code_weights.items():
                if key.lower() in reason_str:
                    base_rate = rate
                    break
            evidence_boost = len(top_evidence) * 0.08
            amount_penalty = min(0.05, dispute["amount"] / 10000)
            win_rate = round(min(0.95, max(0.15, base_rate + evidence_boost - amount_penalty)), 2)
            potential_recovery = round(dispute["amount"] * win_rate, 2)
            recommendation = "Fight" if win_rate >= 0.40 else "Don't Fight"

            results.append(
                {
                    "Transaction_ID": dispute["transaction_id"],
                    "Reason_Code": dispute["reason"],
                    "Recommendation": recommendation,
                    "Top_Evidence": ", ".join(top_evidence),
                    "AI_Argument": argument,
                    "Simulated_Win_Rate": win_rate,
                    "Potential_Recovery": potential_recovery,
                }
            )
        
        # Small delay between batches to respect rate limits
        if i + batch_size < len(all_disputes):
            time.sleep(0.5)

    return pd.DataFrame(results)

# -------------------------
# UI
# -------------------------
uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
num_rows = st.number_input(
    label="Number of disputes to process (recommended: 10)",
    min_value=1,
    max_value=50,
    value=10,
    step=1,
)

process_clicked = st.button("Process")

if process_clicked:
    if uploaded is None:
        st.error("Please upload a CSV file.")
        st.stop()

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # Auto-transform CSV to standardized format
    with st.spinner("Standardizing CSV format..."):
        try:
            df = transform_csv(df)
            st.success(f"✓ CSV transformed and standardized ({len(df)} rows)")
        except Exception as e:
            st.error(f"CSV transformation failed: {e}")
            st.stop()

    try:
        with st.spinner("Processing disputes with AI..."):
            results_df = process_disputes(df, int(num_rows))
    except Exception as e:
        st.error(f"Processing failed: {e}")
        st.stop()

    # Display token usage
    st.info(f"📊 API Usage: {st.session_state.total_tokens:,} tokens used for this batch")
    
    st.subheader("Preview")
    preview_cols = [
        "Transaction_ID",
        "Reason_Code",
        "Recommendation",
        "Top_Evidence",
        "AI_Argument",
        "Simulated_Win_Rate",
        "Potential_Recovery",
    ]
    st.dataframe(results_df[preview_cols], use_container_width=True)

    # Download full processed results
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="AI_Dispute_Arguments.csv",
        mime="text/csv",
    )
