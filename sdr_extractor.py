"""
SDR PDF Extractor — Azure gpt-4.1
==================================
Converts Service Data Record (SDR) PDFs into structured JSON using
azure-hosted gpt-4.1 with vision.

Strategy (two-pass):
  Pass 1 — Vision:  Each PDF page image → gpt-4.1 → verbatim Markdown transcript
  Pass 2 — Text:    Full transcript → gpt-4.1 → structured JSON (schema-mapped)

Usage:
    python sdr_extractor.py docs/1.pdf
    python sdr_extractor.py docs/2.pdf --output results/custom.json

Output:
    results/<pdf_stem>_extraction.json  (default)

Requirements:
    pip install openai python-dotenv pdf2image Pillow
    Poppler must be installed and on PATH (for pdf2image)
    Download: https://github.com/oschwartz10612/poppler-windows/releases
    .env file with FW_API_URL and FW_API_KEY
"""

import argparse
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# ---------------------------------------------------------------------------
# 1. Load environment
# ---------------------------------------------------------------------------
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path)

FW_API_URL = os.getenv("FW_API_URL")
FW_API_KEY = os.getenv("FW_API_KEY")

if not FW_API_URL or not FW_API_KEY:
    print("ERROR: FW_API_URL and FW_API_KEY must be set in the .env file.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Client + model config
# ---------------------------------------------------------------------------
client = OpenAI(base_url=FW_API_URL, api_key=FW_API_KEY)

MODEL       = "gpt-4.1"   # Used for BOTH vision (Pass 1) and text (Pass 2)
RENDER_DPI  = 150          # PDF render resolution — balanced quality vs size
MAX_IMG_W   = 1500         # Max image width in pixels (gpt-4.1 handles up to 2048)

# ---------------------------------------------------------------------------
# 3. JSON Schema
#    Must match the structure of the SDR report exactly.
#    The report pages follow the same top-to-bottom order as this schema.
# ---------------------------------------------------------------------------
JSON_SCHEMA = {
    "document_type": "Service Data Record (SDR)",
    "customer_information": {
        "first_name": "",
        "last_name": "",
        "customer_id": "",
        "date_of_birth": "",
        "age": "",
        "gender": "",
        "county": "",
        "program": "",
        "medicaid_id": "",
        "medicaid_type": "",
        "ddd_status": "",
    },
    "diagnosis_information": {
        "primary_diagnosis": {"icd_code": "", "description": ""},
        "secondary_diagnoses": [{"icd_code": "", "description": ""}],
    },
    "provider_company": {
        "company_name": "",
        "billing_npi": "",
        "mailing_address": {"street": "", "city": "", "state": "", "zip": ""},
    },
    "support_coordination_company": {"company_name": ""},
    "service_authorizations": [
        {
            "service_name": "",
            "procedure_code": "",
            "start_date": "",
            "end_date": "",
            "pa_number": "",
            "total_units": "",
            "total_cost": "",
            "rate": "",
            "unit_type": "",
            "frequency": "",
            "service_location": "",
            "source": "",
        }
    ],
    "customer_goals": [{"outcome_number": "", "outcome_description": ""}],
    "customer_needs": {
        "health_needs":      [{"need_description": ""}],
        "support_needs":     [{"need_description": ""}],
        "employment_needs":  [{"need_description": ""}],
    },
}

# ---------------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------------

def resize_image(image: Image.Image) -> Image.Image:
    """Resize so width <= MAX_IMG_W, preserving aspect ratio."""
    if image.width > MAX_IMG_W:
        ratio = MAX_IMG_W / image.width
        image = image.resize((MAX_IMG_W, int(image.height * ratio)), Image.LANCZOS)
    return image


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to base64-encoded JPEG."""
    image = resize_image(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_json_response(text: str) -> dict:
    """
    Extract and parse JSON from an LLM response.
    Handles markdown triple-backtick fences gracefully.
    """
    text = text.strip()
    # Strip ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    # Find outermost { ... }
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    return json.loads(text)


# ---------------------------------------------------------------------------
# 5. Pass 1 — Vision: transcribe each page verbatim
# ---------------------------------------------------------------------------

TRANSCRIBE_SYSTEM = (
    "You are a verbatim document transcription assistant. "
    "Your ONLY job is to reproduce every single character visible on the page "
    "exactly as it appears — do not rephrase, summarize, or omit anything. "
    "Preserve all table structures, labels, values, codes, dates, and numbers."
)

TRANSCRIBE_USER_TMPL = (
    "This is page {page_num} of a Service Data Record (SDR) document.\n\n"
    "INSTRUCTIONS:\n"
    "- Transcribe EVERY piece of text visible on this page into Markdown.\n"
    "- Reproduce table rows and columns with their exact values — do NOT skip any row.\n"
    "- Copy labels and their values exactly as printed — do NOT paraphrase.\n"
    "- Include ALL codes, dates, dollar amounts, units, addresses, and IDs.\n"
    "- Do NOT summarize, interpret, or omit anything — verbatim copy only.\n\n"
    "Begin transcription:"
)


def transcribe_page(image: Image.Image, page_num: int) -> str:
    """Send one page image to gpt-4.1 and get a verbatim Markdown transcription."""
    b64 = image_to_base64(image)
    w, h = resize_image(image).size
    print(f"  [Page {page_num}] {w}x{h}px -> calling {MODEL}...")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": TRANSCRIBE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TRANSCRIBE_USER_TMPL.format(page_num=page_num)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "high",   # high detail for text-dense documents
                        },
                    },
                ],
            },
        ],
        max_tokens=4096,
        temperature=0,
    )
    result = response.choices[0].message.content.strip()
    print(f"  [OK] Page {page_num} transcribed ({len(result)} chars)")
    return result


# ---------------------------------------------------------------------------
# 6. Pass 2 — Text: map full transcript to JSON
#    gpt-4.1 has a large context window, so we send the entire transcript at once.
#    The report pages are in the same order as the JSON schema, making this reliable.
# ---------------------------------------------------------------------------

MAPPING_SYSTEM = (
    "You are a precise data extraction assistant. "
    "You extract information from document transcripts and output valid JSON only. "
    "Never summarize. Copy values verbatim from the transcript exactly as they appear."
)

MAPPING_USER_TMPL = (
    "Below is the full verbatim transcript of a multi-page Service Data Record (SDR).\n"
    "The pages of the report follow the same order as the JSON schema below.\n\n"
    "STRICT RULES:\n"
    "1. Extract EVERY item — do not skip any service authorization, goal, or need.\n"
    "2. Copy ALL values VERBATIM — exact dates, codes, dollar amounts, addresses, descriptions.\n"
    "3. Do NOT summarize, paraphrase, or shorten any description.\n"
    "4. For arrays (service_authorizations, customer_goals, health_needs, support_needs, "
    "employment_needs): include ALL entries found in the transcript.\n"
    "5. Leave a field as an empty string \"\" only if the value is genuinely not present.\n"
    "6. Output ONLY valid JSON — no explanation, no markdown fences.\n\n"
    "JSON SCHEMA TO FILL:\n{schema}\n\n"
    "FULL TRANSCRIPT:\n{transcript}"
)


def map_text_to_json(full_transcript: str, schema: dict) -> dict:
    """
    Map the full verbatim transcript to structured JSON using gpt-4.1.
    Sends the entire transcript in one call — no chunking needed.
    """
    schema_str = json.dumps(schema, indent=2)
    prompt = MAPPING_USER_TMPL.format(schema=schema_str, transcript=full_transcript)

    print(f"\n[Step 2b] Mapping transcript to JSON ({len(full_transcript)} chars) -> {MODEL}...")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": MAPPING_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=8192,
        temperature=0,
    )
    raw = response.choices[0].message.content
    print(f"  [OK] JSON response received ({len(raw)} chars)")
    return parse_json_response(raw)


# ---------------------------------------------------------------------------
# 7. Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_sdr(pdf_path: str) -> dict:
    """
    Full extraction pipeline:
      Pass 1 (vision): Each page -> gpt-4.1 -> verbatim Markdown transcript
      Pass 2 (text):   Full transcript -> gpt-4.1 -> structured JSON
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("ERROR: pdf2image is not installed. Run: pip install pdf2image", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  SDR Extractor -- Azure {MODEL}")
    print(f"  PDF: {pdf_path}")
    print(f"{'='*60}\n")

    # --- Step 1: Render PDF pages ---
    print(f"[Step 1] Rendering PDF at {RENDER_DPI} DPI...")
    try:
        pages = convert_from_path(pdf_path, dpi=RENDER_DPI)
    except Exception as e:
        print(
            f"\nERROR rendering PDF: {e}\n"
            "Make sure Poppler is installed and on PATH.\n"
            "Download: https://github.com/oschwartz10612/poppler-windows/releases",
            file=sys.stderr,
        )
        sys.exit(1)

    total_pages = len(pages)
    print(f"  [OK] {total_pages} page(s) found\n")

    # --- Step 2a: Transcribe each page verbatim ---
    print(f"[Step 2a] Pass 1 Vision: transcribing {total_pages} page(s) with {MODEL}...")
    page_transcripts = []
    for i, page_image in enumerate(pages):
        page_num = i + 1
        transcript = transcribe_page(page_image, page_num)
        page_transcripts.append(
            f"=== PAGE {page_num} ===\n{transcript}\n=== END PAGE {page_num} ==="
        )

    full_transcript = "\n\n".join(page_transcripts)
    print(f"\n  [OK] Full transcript: {len(full_transcript)} chars across {total_pages} pages")

    # --- Step 2b: Map full transcript to JSON ---
    result = map_text_to_json(full_transcript, JSON_SCHEMA)
    return result


# ---------------------------------------------------------------------------
# 8. Entry point
# ---------------------------------------------------------------------------

# Results directory (created automatically if it doesn't exist)
RESULTS_DIR = Path(__file__).parent / "results"


def main():
    parser = argparse.ArgumentParser(
        description=f"Extract structured JSON from SDR PDF using Azure {MODEL}"
    )
    parser.add_argument("pdf", help="Path to the SDR PDF file (e.g. docs/1.pdf)")
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path (default: results/<pdf_stem>_extraction.json)",
        default=None,
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        RESULTS_DIR.mkdir(exist_ok=True)
        output_path = RESULTS_DIR / f"{pdf_path.stem}_extraction.json"

    result = extract_sdr(str(pdf_path))

    print(f"\n[Step 3] Saving JSON to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Done! Saved to: {output_path}\n")

    # Preview first 2000 chars
    preview = json.dumps(result, indent=2, ensure_ascii=False)
    print("--- Extracted JSON Preview ---")
    print(preview[:2000])
    if len(preview) > 2000:
        print("  ... (truncated — see output file for full content)")


if __name__ == "__main__":
    main()
