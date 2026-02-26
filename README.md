# SDR PDF Extractor

Converts **Service Data Record (SDR)** PDF documents into structured JSON using a **two-pass AI pipeline** powered by Azure-hosted **GPT-4.1** with vision capabilities.

---

## How It Works

```
PDF → [Pass 1: Vision] → Verbatim Markdown Transcript → [Pass 2: Text] → Structured JSON
```

| Pass | Model Input | Output |
|------|-------------|--------|
| **Pass 1 — Vision** | Each PDF page as an image | Verbatim Markdown transcript per page |
| **Pass 2 — Text** | Full combined transcript | Schema-mapped structured JSON |

---

## Project Structure

```
.
├── sdr_extractor.py     # Main extraction script
├── requirements.txt     # Python dependencies
├── .env                 # API credentials (not committed)
├── docs/                # Input SDR PDF files
│   ├── 1.pdf
│   └── 2.pdf
└── results/             # Extracted JSON outputs (auto-created)
```

---

## Prerequisites

- Python 3.9+
- **Poppler** (required by `pdf2image` for PDF rendering)
  - Windows: Download from [oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) and add to PATH
  - macOS: `brew install poppler`
  - Linux: `sudo apt-get install poppler-utils`

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/jyothsnar24/fieldworker-llm.git
cd fieldworker-llm
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

**3. Configure environment variables**

Create a `.env` file in the project root:
```env
FW_API_URL=https://your-azure-endpoint/openai/deployments/gpt-4.1
FW_API_KEY=your_azure_api_key
```

---

## Usage

```bash
# Extract a single SDR PDF (output saved to results/)
python sdr_extractor.py docs/1.pdf

# Specify a custom output path
python sdr_extractor.py docs/2.pdf --output results/custom_output.json
```

---

## Output Schema

The extracted JSON follows a fixed schema covering:

- **Customer Information** — name, ID, DOB, gender, Medicaid details
- **Diagnosis Information** — primary & secondary ICD codes
- **Provider & Support Coordination** — company names, NPI, address
- **Service Authorizations** — procedure codes, dates, units, costs, PA numbers
- **Customer Goals** — outcome descriptions
- **Customer Needs** — health, support, and employment needs

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `openai>=1.30.0` | Azure GPT-4.1 API client |
| `python-dotenv>=1.0.0` | Load `.env` credentials |
| `pdf2image>=1.17.0` | Render PDF pages as images |
| `Pillow>=10.0.0` | Image processing & resizing |

---

## Notes

- The script renders PDFs at **150 DPI** and caps image width at **1500px** for optimal token efficiency with GPT-4.1 vision.
- Temperature is set to `0` in both passes for deterministic, consistent extraction.
- The `.env` file is excluded from version control — never commit your API keys.
