#!/usr/bin/env python3
"""
Warp MVP: Chargeback AI Argument Generator
------------------------------------------
Requirements:
- Your CSV must have these columns:
  Transaction_ID, Customer_Name, Customer_Email, Date, Amount, Product,
  Reason_Code, Evidence_Receipt, Evidence_Shipping, Evidence_Email, Evidence_Photo, Current_Process, Notes
- OpenAI API key must be set in your environment
"""

import pandas as pd
from openai import OpenAI
import os
import json
import sys
import time

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Warning: OPENAI_API_KEY not found in environment")
    print("Set it with: export OPENAI_API_KEY='your-key-here'")
    exit(1)

# Initialize OpenAI client
client = OpenAI()

# 1. Upload CSV
if len(sys.argv) < 2:
    print("Usage: python3 chargeback_ai_generator.py <csv_file_path>")
    exit(1)

csv_file = sys.argv[1]
try:
    df = pd.read_csv(csv_file).head(10)
    print(f"\n✓ Loaded {len(df)} disputes from CSV\n")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit(1)

results = []
total_tokens_used = 0

# 2. Process each dispute
for index, row in df.iterrows():
    print(f"Processing dispute {index + 1}/{len(df)}: {row['Transaction_ID']}...")
    
    # --- Evidence ranking prompt ---
    evidence_prompt = f"""
You are an expert in chargeback dispute resolution. 
Here is a transaction and associated evidence:

Transaction: {row['Product']}, Amount: ${row['Amount']}, Reason: {row['Reason_Code']}
Evidence: 
- Receipt: {row['Evidence_Receipt']}
- Shipping: {row['Evidence_Shipping']}
- Email: {row['Evidence_Email']}
- Photo: {row['Evidence_Photo']}

Rank the evidence from most likely to help win this dispute to least. 
Output in JSON format like: {{ "Evidence_Ranking": ["Shipping", "Receipt", "Email", "Photo"] }}
"""

    try:
        ranking_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a chargeback dispute AI."},
                {"role": "user", "content": evidence_prompt}
            ],
            temperature=0
        )

        ranking_json = ranking_response.choices[0].message.content
        # Track tokens
        total_tokens_used += ranking_response.usage.total_tokens
        
        # Extract top 2 evidence fields
        top_evidence = json.loads(ranking_json)["Evidence_Ranking"][:2]

        # --- Argument generation prompt ---
        argument_prompt = f"""
You are an expert chargeback dispute writer. 
Using the top evidence: {top_evidence} 
and the transaction details: Product={row['Product']}, Amount=${row['Amount']}, Reason={row['Reason_Code']},
generate a concise and persuasive dispute argument to submit to the issuing bank.
Focus on clarity, credibility, and relevance to the reason code.
"""
        argument_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a chargeback argument AI."},
                {"role": "user", "content": argument_prompt}
            ],
            temperature=0.7
        )

        argument_text = argument_response.choices[0].message.content
        # Track tokens
        total_tokens_used += argument_response.usage.total_tokens

        # --- Simulated metrics ---
        # Base rate varies by reason code
        reason_code_weights = {
            'Fraud': 0.35,
            'Product Not Received': 0.55,
            'Product Unacceptable': 0.45,
            'Duplicate Processing': 0.70,
            'Credit Not Processed': 0.60,
            'Unauthorized Transaction': 0.30
        }
        base_rate = reason_code_weights.get(row['Reason_Code'], 0.50)
        
        # Evidence quality boost (top evidence adds more)
        evidence_boost = len(top_evidence) * 0.08
        
        # Amount factor (higher amounts slightly harder to win)
        amount_penalty = min(0.05, row['Amount'] / 10000)
        
        simulated_win_rate = round(min(0.95, max(0.15, base_rate + evidence_boost - amount_penalty)), 2)
        potential_recovery = round(row['Amount'] * simulated_win_rate, 2)

        results.append({
            "Transaction_ID": row['Transaction_ID'],
            "Customer_Name": row['Customer_Name'],
            "Product": row['Product'],
            "Reason_Code": row['Reason_Code'],
            "Top_Evidence": ", ".join(top_evidence),
            "AI_Argument": argument_text,
            "Simulated_Win_Rate": simulated_win_rate,
            "Potential_Recovery": potential_recovery
        })
        
        print(f"  ✓ Generated argument (win rate: {simulated_win_rate})\n")
        
        # Add delay to avoid rate limits
        time.sleep(2)
        
    except Exception as e:
        print(f"  ❌ Error processing dispute: {e}\n")
        time.sleep(2)
        continue

# 3. Display results in a table
if results:
    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("=== AI Dispute Argument Results ===")
    print("="*80 + "\n")
    print(results_df.to_string(index=False))
    print("\n")

    # 4. Auto-save CSV export
    results_df.to_csv("AI_Dispute_Arguments.csv", index=False)
    print("\n✅ Results saved as AI_Dispute_Arguments.csv")
    print(f"\n📊 Total tokens used: {total_tokens_used:,}")
    print(f"   Average per dispute: {total_tokens_used // len(results):,}" if results else "")
else:
    print("\n⚠️  No results generated")
