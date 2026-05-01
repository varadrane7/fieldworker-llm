# SDR PDF Extraction Pipeline

## Overview
This project implements a one-pass pipeline to extract structured JSON from Service Detail Report (SDR) PDFs using **OpenAI’s GPT-4.1** model, hosted on Azure.

The system processes raw PDFs directly without traditional parsing libraries. Instead, it relies on the model’s ability to interpret both text and layout.

---

## Key Features

- One-pass extraction (no OCR or preprocessing pipeline)
- Prompt-driven logic (easy to iterate without changing code)
- Schema-guided output
- Works across multiple SDR formats (NJISP and TPSDR)

---

## Project Structure

```
project-root/
├── docs/              # Input PDF files
├── prompts/           # Prompt templates
├── scripts/           # Python pipeline
├── results/           # Output JSON files
├── .env               # API keys
└── README.md
```

---

## Setup

### 1. Install Dependencies
pip install openai python-dotenv

### 2. Configure Environment Variables
Create a `.env` file in the project root:

FW_API_URL=your_api_base_url
FW_API_KEY=your_api_key

---

## Running the Pipeline

From the scripts directory:

python directpass_extraction_new.py

By default:
- Reads PDFs from docs/
- Applies extraction prompt
- Saves results to results/

---

## How It Works

1. Load prompt from /prompts
2. Inject JSON schema
3. Encode PDF
4. Send to Vision-LLM
5. Receive JSON
6. Save output

---

## Known Limitations

### Multi-Column, Cross-Page Needs Sections
These sections are difficult due to layout complexity. Most content is extracted, but grouping may not always be perfect.

### Layout Ambiguity
When structure is unclear, the model prioritizes:
- preserving structure
- avoiding hallucination

---

## Future Improvements

- API-level structured outputs (response_format)
- Post-processing cleanup
- Further prompt refinement

---

## Summary

- No external parsing
- Minimal preprocessing
- Prompt-driven system
- Consistent outputs across formats

---

## Notes

- Optimized for correctness over perfect layout reconstruction
- Prompt engineering is the main improvement lever
- Outputs may require minor cleanup for production
