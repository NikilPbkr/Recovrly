# Chargeback AI Argument Generator

Automated chargeback dispute resolution using GPT to rank evidence and generate persuasive arguments.

## Features

✅ Upload CSV with dispute data  
✅ Automatically rank evidence using AI  
✅ Generate persuasive dispute arguments  
✅ Calculate simulated win rates based on reason code, evidence, and amount  
✅ Export results to CSV  
✅ Track token usage and costs  

## Requirements

- Python 3.9+
- OpenAI API key with credits
- Required packages: `pandas`, `openai`

## Installation

```bash
pip install pandas openai
```

## Usage

```bash
export OPENAI_API_KEY='your-api-key-here'
python3 chargeback_ai_generator.py Dummy_Chargeback_Data.csv
```

## CSV Format

Your CSV must have these columns:
- `Transaction_ID`
- `Customer_Name`
- `Customer_Email`
- `Date`
- `Amount`
- `Product`
- `Reason_Code`
- `Evidence_Receipt`
- `Evidence_Shipping`
- `Evidence_Email`
- `Evidence_Photo`
- `Current_Process`
- `Notes`

## Output

The script generates `AI_Dispute_Arguments.csv` with:
- Transaction ID
- Customer Name
- Product
- Reason Code
- Top Evidence (ranked)
- AI-Generated Argument
- Simulated Win Rate
- Potential Recovery Amount

## Token Usage

- **~524 tokens per dispute** (with GPT-3.5-turbo)
- **10 disputes ≈ $0.01**
- **100 disputes ≈ $0.10**
- **1,000 disputes ≈ $1.00**

## Win Rate Calculation

Win rates are calculated based on:
1. **Reason Code** - Different dispute types have different base win rates
2. **Evidence Quality** - More/better evidence increases win rate
3. **Transaction Amount** - Higher amounts slightly reduce win rate

## Example Output

```
Processing dispute 1/10: b6296533-b87c-4968-90bd-ed486671ae43...
  ✓ Generated argument (win rate: 0.64)

📊 Total tokens used: 5,244
   Average per dispute: 524
```
