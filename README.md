# Chargeback AI Argument Generator

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
- OpenAI API key with credits  
- Required packages: `pandas`, `openai`, `streamlit`

---

## Installation

```bash
pip install -r requirements.txt
