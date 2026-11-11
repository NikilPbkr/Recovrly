# Chargeback AI Argument Generator

[![Tests](https://github.com/NikilPbkr/Recovrly/actions/workflows/tests.yml/badge.svg)](https://github.com/NikilPbkr/Recovrly/actions/workflows/tests.yml)

Automated chargeback dispute resolution using GPT to rank evidence, generate persuasive arguments, and simulate recovery outcomes.

This project explores how AI can assist with **structured, economically meaningful work** by breaking a real operational workflow into modular, automatable steps.

---

## Overview

Chargeback disputes are time-sensitive, evidence-heavy, and often handled through repetitive manual review and writing. Each case requires:

- Interpreting structured transaction data  
- Reviewing semi-structured evidence (receipts, shipping info, emails, photos)  
- Identifying the most relevant supporting signals  
- Writing a reason-code-specific, persuasive response  

This project treats dispute resolution as a **decision-making workflow** that can be decomposed into ranking, reasoning, and generation steps — making it a strong candidate for AI-assisted automation rather than simple text generation.

---

## Features

✅ Upload CSV with dispute data  
✅ Automatically rank evidence using AI  
✅ Generate persuasive dispute arguments  
✅ Calculate simulated win rates based on reason code, evidence, and amount  
✅ Export results to CSV  
✅ Track token usage and costs  

---

## System Design

The pipeline separates the task into modular, inspectable stages:

### 1. Feature Extraction  
Structured fields (amount, reason code, product type) and unstructured evidence are normalized into a consistent internal representation.

### 2. Evidence Ranking  
GPT is prompted to prioritize evidence based on dispute type, simulating how a human reviewer would identify the strongest supporting signals.

### 3. Argument Generation  
The model produces a structured, reason-code-aware response designed to maximize clarity and persuasiveness.

### 4. Outcome Simulation  
A rules-based model estimates win probability using:
- Base difficulty of the dispute reason code  
- Strength and quantity of supporting evidence  
- Transaction value  

This modular design makes the workflow adjustable and extensible rather than a single opaque model call.

---

## Requirements

- Python 3.9+
- Packages in `requirements.txt`: `pandas`, `openai`, `python-dotenv`, `streamlit`
- `OPENAI_API_KEY` only when you process disputes in the app. Standardizing a CSV and running the tests do not call the API.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the app

From the repository root:

```bash
streamlit run app.py
```

Upload a CSV, choose how many disputes to process (1–50, default 10), and click **Process**. The app standardizes the file, shows which uploaded columns mapped to standard fields, ranks evidence, writes a dispute argument, and estimates a win rate. Download the results as `AI_Dispute_Arguments.csv`.

The page loads without an API key. If `OPENAI_API_KEY` is missing, the app shows:

> OPENAI_API_KEY not found. Add it to a .env file in this folder or export it in your shell.

**Process** stops with that same message until the key is set. The OpenAI client is created on the first API call, not when the app starts, so importing the app does not require a key.

If a call to OpenAI fails, the page explains what happened and fills that step from a fallback. A timeout, a rejected key, a rate limit, a connection problem, and a response the app cannot read each get their own message. The same message is shown once even when several batches fail for the same reason. The raw error is left off the page, because it can include request details or part of the API key. Results still appear, and the fallback text is what gets downloaded for the step that failed. A call that succeeds is unchanged: the model ranks the evidence and writes the argument.

---

## Environment variables

| Variable | When it is used |
| --- | --- |
| `OPENAI_API_KEY` | Required to process disputes. The client reads it when the first API call runs. |
| `CHARGEBACK_MODEL` | Optional. Chat model for ranking and argument generation. Defaults to `gpt-3.5-turbo`. |

On startup the app loads a `.env` file from the working directory (or a parent directory, if that is where `python-dotenv` finds it). Copy `.env.example` to `.env` and set the key. You can also export the variables in the shell before starting the app. A variable already set in the environment is left as-is. `.env` is gitignored.

```bash
OPENAI_API_KEY=sk-...
CHARGEBACK_MODEL=gpt-3.5-turbo
```

Set `CHARGEBACK_MODEL` before startup. It is read once, when `app.py` loads.

---

## CSV input

Both the app and `transform_csv.py` expect a header row and one dispute per data row. Headers do not have to be the standard names. `Dummy_Chargeback_Data.csv` is a sample that already uses those names.

A merchant export can use other headers, currency text, and a non-ISO date:

```csv
Order ID,Cardholder Name,Client Email,Created At,Charge Amount,Item Description,Chargeback Reason,Receipt URL,Tracking Number,Notes
ORD-100,Ada Lovelace,ada@example.com,01/15/2024,"$1,234.50",Analytical Engine,Product Not Received,https://cdn.example.com/receipt.pdf,1Z999,VIP account
```

Both the app and `transform_csv.py` turn that row into `Transaction_ID` `ORD-100`, `Date` `2024-01-15`, `Amount` `1234.5`, `Evidence_Receipt` `PDF document provided: https://cdn.example.com/receipt.pdf`, and `Evidence_Shipping` `1Z999`. Columns that were not in the file are filled with the placeholders below. Both parse currency amounts; see [Amounts](#amounts).

### Recognized columns

Each header is stripped, lowercased, and compared with the aliases below. Spaces, hyphens, and underscores are interchangeable (`charge_amount`, `Charge Amount`, `charge-amount`). Close spellings of an alias also match, including `ReasonCode`, `e-mail`, and `Tracking Number`. The standard name itself matches in any case.

When two headers match the same field, the leftmost one is used. A header that matches nothing is ignored and its values are dropped.

| Standard field | Accepted aliases |
| --- | --- |
| `Transaction_ID` | `transaction_id`, `txn_id`, `trans_id`, `id`, `order_id`, `payment_id`, `charge_id` |
| `Customer_Name` | `customer_name`, `name`, `customer`, `buyer_name`, `client_name`, `cardholder_name` |
| `Customer_Email` | `customer_email`, `email`, `buyer_email`, `client_email`, `contact_email` |
| `Date` | `date`, `transaction_date`, `txn_date`, `created`, `timestamp`, `created_at`, `order_date` |
| `Amount` | `amount`, `total`, `charge_amount`, `transaction_amount`, `price`, `value` |
| `Product` | `product`, `description`, `item`, `product_name`, `service`, `item_description` |
| `Reason_Code` | `reason_code`, `reason`, `dispute_reason`, `chargeback_reason`, `code` |
| `Evidence_Receipt` | `evidence_receipt`, `receipt`, `receipt_url`, `receipt_link`, `proof_of_purchase` |
| `Evidence_Shipping` | `evidence_shipping`, `shipping`, `tracking`, `shipping_proof`, `delivery_confirmation` |
| `Evidence_Email` | `evidence_email`, `email_proof`, `correspondence`, `communication` |
| `Evidence_Photo` | `evidence_photo`, `photo`, `image`, `product_image`, `photo_proof` |
| `Current_Process` | `current_process`, `status`, `dispute_status`, `process`, `stage` |
| `Notes` | `notes`, `comments`, `remarks`, `additional_info` |

After **Process**, the app opens a **Column mapping** table. Each standard field shows the uploaded header that filled it, or `placeholder` when no header matched and a default was used. The table starts expanded when any field used a placeholder. The command-line tool prints the same mapping (`Mapped 'Order ID' → 'Transaction_ID'`, or `No match for '…' (will be ignored)`).

### Placeholders

The output always has the 13 standard columns, in the order in the table above. A missing column is filled for every row. A blank cell in a column that is present is filled with the same value, with two exceptions noted below.

| Field | Placeholder |
| --- | --- |
| `Transaction_ID` | `TXN_000000`, `TXN_000001`, … when the whole column is missing. A blank cell in a present ID column stays blank. |
| `Customer_Name` | `Unknown` |
| `Customer_Email` | `unknown@example.com` |
| `Date` | The calendar date when `app.py` or `transform_csv.py` was imported, as `YYYY-MM-DD` |
| `Amount` | `0.0` |
| `Product` | `Unknown Product` |
| `Reason_Code` | `Unknown` |
| `Evidence_Receipt` | `No receipt provided` |
| `Evidence_Shipping` | `No shipping evidence provided` |
| `Evidence_Email` | `No email evidence provided` |
| `Evidence_Photo` | `No photo evidence provided` |
| `Current_Process` | `Pending` |
| `Notes` | `No notes provided` |

A blank cell in an evidence column stays empty. The "No … provided" text is used when that evidence column is absent.

### Dates

The app and the command-line tool both write `Date` as `YYYY-MM-DD`.

Accepted forms:

- US month-first numbers: `01/15/2024`, `1/15/24`, `01-15-2024`, `1-15-24`. `02/01/2024` is February 1, not January 2.
- Year-first numbers: `2024/01/15`, `2024-01-15`, `2024-1-5`. An exact `YYYY-MM-DD` value is kept as stored.
- English month names: `Jan 15, 2024`, `January 15, 2024`, `Jan 15 2024`, `15-Jan-2024`, `15 Jan 2024`, `15-Jan-24`. Month names are matched in English.
- ISO datetimes, such as `2024-01-15T13:45:30Z` or `2024-01-15 13:45:30`, including fractional seconds and a numeric offset. The calendar date in the text is kept and the time is dropped. The offset does not move the date.
- Real `datetime` and `date` values, using that calendar date.

Two-digit years follow Python's `%y` window: `00`–`68` become 2000–2068, and `69`–`99` become 1969–1999.

A value that is not one of those forms is missing and receives the date placeholder. That includes day-first numbers such as `15/01/2024` (they are not swapped into month-first order) and impossible dates such as `02/31/2024`.

### Amounts

Both the app and `transform_csv.py` parse currency amounts into a float. `$1,234.50` becomes `1234.5` on either path, and that number is what the win-rate calculation uses.

The parser:

- Leaves an already-numeric column unchanged. Plain numeric text such as `19.99` or `10` becomes a float.
- Removes `$`, `€`, `£`, `¥`, and `US$`.
- Removes these currency codes when they appear as whole words: `USD`, `EUR`, `GBP`, `CAD`, `AUD`, `NZD`, `CHF`, `JPY`, `CNY`, `INR`, `HKD`, `SGD`, `MXN`, `BRL`. `USD 30` and `30 USD` both become `30`.
- Removes thousands commas. `$1,234.50` becomes `1234.5`. A comma is only a thousands separator, so `1.234,50` is unparseable.
- Treats a leading or trailing minus, or accounting parentheses such as `(12.00)` and `($1,234.50)`, as a negative amount. A leading `+` is ignored.
- Replaces a blank or unparseable cell with `0.0` and keeps the row.

### Evidence links and files

In both the app and the command-line tool, an evidence cell that starts with `http://`, `https://`, or `ftp://`, or that ends in `.pdf`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.doc`, or `.docx`, is rewritten as a short summary that still includes the original value:

- `.pdf` → `PDF document provided: …`
- `.png`, `.jpg`, `.jpeg`, `.gif` → `Image evidence provided: …`
- `.doc`, `.docx` → `Document provided: …`
- Any other link → `Evidence link provided: …`

Other evidence text is kept as written. A tracking number such as `1Z999` stays `1Z999`.

---

## Standardizing a CSV from the command line

```bash
python transform_csv.py input.csv
python transform_csv.py input.csv output.csv
```

With one argument, the file is written next to the input as `<input>_standardized.csv`. The app applies the same column matching, currency parsing, date normalization, evidence summaries, and placeholder fill when you click **Process**.

---

## Running tests

From the repository root:

```bash
python -m unittest discover -s tests -t .
```

No API key is required. The suite never calls the API. The same command runs in GitHub Actions (Python 3.12) on pushes to `main` and on pull requests.
