"""
extractor.py — LangChain SDR PDF Extractor
===========================================
Two-pass extraction pipeline:
  Pass 1 (Vision) : PyMuPDF renders each page → GPT-4.1 vision transcribes → Markdown transcript
  Pass 2 (Text)   : Full transcript + schema → GPT-4.1 → structured JSON

Usage:
    python extractor.py --pdf docs/1.pdf
    python extractor.py --pdf docs/2.pdf --output results/custom.json
    python extractor.py --docs-dir docs

Requirements:
    pip install -r requirements.txt
    .env with FW_API_URL, FW_API_KEY, FW_MODEL_NAME (optional, default gpt-4.1)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import argparse
import base64
import io
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fitz                                           # PyMuPDF — PDF rendering, no Poppler needed
from dotenv import load_dotenv
from PIL import Image

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env")

FW_API_URL   = os.getenv("FW_API_URL")
FW_API_KEY   = os.getenv("FW_API_KEY")
FW_MODEL_NAME = os.getenv("FW_MODEL_NAME", "gpt-4.1")

if not FW_API_URL or not FW_API_KEY:
    print("ERROR: FW_API_URL and FW_API_KEY must be set in the .env file.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Constants
# ---------------------------------------------------------------------------
PASS2_SYSTEM = (
    "You are a precise data extraction assistant. "
    "You must output ONLY valid JSON matching the provided schema exactly. "
    "Do not add extra keys. Do not include markdown fences or code blocks. "
    "Do not explain anything. Copy all values verbatim from the transcript — "
    "never infer, summarize, or hallucinate values."
)

DEFAULT_DPI       = 200
DEFAULT_MAX_WIDTH = 1600

# ---------------------------------------------------------------------------
# 3. File utilities
# ---------------------------------------------------------------------------

def load_text(path: Path) -> str:
    """Read a UTF-8 text file. Raises FileNotFoundError with a clear message."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_schema(path: Path) -> dict:
    """
    Load json_schema_template.txt as a parsed dict.
    The file is valid JSON so we use json.load() directly.
    """
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 4. PDF rendering  (ported from scripts/test_sdr_extractor_v1.py:100–138)
# ---------------------------------------------------------------------------

def render_pdf_pages(
    pdf_path: Path,
    dpi: int = DEFAULT_DPI,
    max_width: int = DEFAULT_MAX_WIDTH,
) -> list:
    """
    Render each page of a PDF to a PIL RGB Image using PyMuPDF (fitz).
    No Poppler dependency required.

    Args:
        pdf_path:  Absolute path to the PDF file.
        dpi:       Render resolution. 200 gives sharp text on dense SDR pages.
        max_width: Downscale pages wider than this (pixels). Preserves aspect ratio.

    Returns:
        List of PIL Images, one per page, in page order.
    """
    doc = fitz.open(str(pdf_path))

    zoom = dpi / 72.0           # PDF default is 72 dpi
    mat  = fitz.Matrix(zoom, zoom)

    images = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if img.width > max_width:
            ratio    = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img      = img.resize(new_size, Image.LANCZOS)

        images.append(img)

    doc.close()
    return images


def pil_to_base64_jpeg(img: Image.Image, quality: int = 85) -> str:
    """
    Convert a PIL Image to a raw base64-encoded JPEG string.
    Converts to RGB first if necessary.

    Returns:
        Raw base64 string (no data-URI prefix).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# 5. JSON utilities
# ---------------------------------------------------------------------------

def parse_json_robust(text: str) -> dict:
    """
    Extract and parse JSON from an LLM response, handling common failure modes.

    Strategy (tried in order):
    1. Strip whitespace and attempt direct json.loads().
    2. Strip ```json ... ``` markdown fences, attempt json.loads().
    3. Find outermost { ... } boundaries, attempt json.loads().
    4. Raise ValueError with raw text so the caller can debug.
    """
    text = text.strip()

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Attempt 3: find outermost { ... }
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse JSON from model response.\n"
        f"--- Raw response (first 500 chars) ---\n{text[:500]}"
    )


def enforce_schema(model_output: Any, schema_template: Any) -> Any:
    """
    Recursively enforce schema_template structure on model_output.

    - Dict:  keep only keys from schema_template; fill missing keys from schema defaults.
    - List:  if schema has a prototype item, apply enforce_schema to each output item.
    - Primitive: coerce type; use schema default if output is None.

    Ported verbatim from scripts/directpass_extraction_new.py:108–167.
    """
    if isinstance(schema_template, dict):
        result = {}
        if not isinstance(model_output, dict):
            model_output = {}
        for key, schema_value in schema_template.items():
            output_value   = model_output.get(key, None)
            result[key]    = enforce_schema(output_value, schema_value)
        return result

    if isinstance(schema_template, list):
        # Empty schema list → preserve output list as-is, or return []
        if len(schema_template) == 0:
            return model_output if isinstance(model_output, list) else []
        item_schema = schema_template[0]
        if not isinstance(model_output, list):
            return []
        return [enforce_schema(item, item_schema) for item in model_output]

    # Primitive — use schema default if missing
    if model_output is None:
        return deepcopy(schema_template)

    # Light type normalization
    if isinstance(schema_template, str):
        return str(model_output)
    if isinstance(schema_template, bool):
        if isinstance(model_output, bool):
            return model_output
        if isinstance(model_output, str):
            lowered = model_output.strip().lower()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0"}:
                return False
        return bool(model_output)
    if isinstance(schema_template, int):
        try:
            return int(model_output)
        except Exception:
            return schema_template
    if isinstance(schema_template, float):
        try:
            return float(model_output)
        except Exception:
            return schema_template

    return model_output


# ---------------------------------------------------------------------------
# 6. LangChain LLM factory
# ---------------------------------------------------------------------------

def build_llm(max_tokens: int = 4096) -> ChatOpenAI:
    """
    Build a ChatOpenAI instance pointed at the Azure AI Foundry endpoint.

    The endpoint (fwfoundry.openai.azure.com/openai/v1) uses the OpenAI-compatible
    API surface, so ChatOpenAI with base_url is correct — not AzureChatOpenAI.

    Wraps the LLM with .with_retry() for automatic exponential-backoff retries
    on transient network or rate-limit errors.

    Args:
        max_tokens: Token budget for the response.
                    Pass 1 uses 4096 (transcription); Pass 2 uses 8192 (JSON).
    """
    llm = ChatOpenAI(
        base_url=FW_API_URL,
        api_key=FW_API_KEY,
        model=FW_MODEL_NAME,
        temperature=0,
        max_tokens=max_tokens,
    )
    return llm.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)


# ---------------------------------------------------------------------------
# 7. Pass 1 — Vision transcription
# ---------------------------------------------------------------------------

def build_page_message(b64_image: str, page_num: int, user_prompt_template: str) -> HumanMessage:
    """
    Build a multimodal HumanMessage containing text instructions + the page image.

    LangChain passes list content directly to the OpenAI API, which supports
    {"type": "image_url", ...} for vision models.

    Args:
        b64_image:            Raw base64 JPEG string (no data-URI prefix).
        page_num:             1-based page number, interpolated into the prompt.
        user_prompt_template: Content of transcription_user_prompt.txt with {page_num}.
    """
    return HumanMessage(content=[
        {
            "type": "text",
            "text": user_prompt_template.format(page_num=page_num),
        },
        {
            "type": "image_url",
            "image_url": {
                "url":    f"data:image/jpeg;base64,{b64_image}",
                "detail": "high",
            },
        },
    ])


def transcribe_page(
    llm: ChatOpenAI,
    system_prompt: str,
    user_prompt_template: str,
    img: Image.Image,
    page_num: int,
) -> str:
    """
    Transcribe a single PDF page image to verbatim Markdown using Pass 1 chain.

    Chain: [SystemMessage, HumanMessage(text+image)] | llm | StrOutputParser
    """
    b64     = pil_to_base64_jpeg(img)
    chain   = llm | StrOutputParser()
    return chain.invoke([
        SystemMessage(content=system_prompt),
        build_page_message(b64, page_num, user_prompt_template),
    ])


def run_pass1(
    llm: ChatOpenAI,
    pages: list,
    system_prompt: str,
    user_prompt_template: str,
    stem: str,
    results_dir: Path,
    save_transcript: bool = True,
) -> str:
    """
    Run Pass 1 over all pages sequentially, returning the full combined transcript.

    Each page transcript is wrapped in PAGE N delimiters for clear segmentation
    in the Pass 2 prompt. Optionally saves the transcript as a .md file for debugging.

    Args:
        llm:                  ChatOpenAI instance (already wrapped with .with_retry).
        pages:                List of PIL Images from render_pdf_pages().
        system_prompt:        Content of transcription_system_prompt.txt.
        user_prompt_template: Content of transcription_user_prompt.txt.
        stem:                 PDF filename stem (e.g. '1' for '1.pdf').
        results_dir:          Directory to save the transcript file.
        save_transcript:      If True, write results/<stem>_transcript.md.

    Returns:
        Full transcript string with all pages joined.
    """
    page_transcripts = []
    total = len(pages)

    for i, img in enumerate(pages, start=1):
        w, h = img.size
        print(f"    [Page {i}/{total}] {w}x{h}px — calling {FW_MODEL_NAME}...")
        transcript = transcribe_page(llm, system_prompt, user_prompt_template, img, i)
        page_transcripts.append(
            f"=== PAGE {i} ===\n{transcript}\n=== END PAGE {i} ==="
        )
        print(f"    [Page {i}/{total}] Done — {len(transcript):,} chars")

    full_transcript = "\n\n".join(page_transcripts)

    if save_transcript:
        results_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = results_dir / f"{stem}_transcript.md"
        transcript_path.write_text(full_transcript, encoding="utf-8")
        print(f"  [OK] Transcript saved: {transcript_path.name}")

    return full_transcript


# ---------------------------------------------------------------------------
# 8. Pass 2 — JSON extraction
# ---------------------------------------------------------------------------

def build_pass2_user_content(
    extraction_prompt_raw: str,
    section_rules: str,
    schema_str: str,
    full_transcript: str,
) -> str:
    """
    Compose the Pass 2 user message by injecting schema + appending rules + transcript.

    IMPORTANT: directpass_sdr_extraction_prompt.txt uses {{JSON_SCHEMA}} (double braces)
    as its placeholder. We use str.replace() BEFORE handing the string to LangChain
    to avoid template variable parsing conflicts.

    Args:
        extraction_prompt_raw: Raw text of directpass_sdr_extraction_prompt.txt.
        section_rules:         Raw text of onepass_section_rules.txt.
        schema_str:            JSON schema serialised to a string (json.dumps).
        full_transcript:       Output of run_pass1().

    Returns:
        Complete user message string for Pass 2.
    """
    # Inject schema into prompt (replaces all occurrences of {{JSON_SCHEMA}})
    prompt_with_schema = extraction_prompt_raw.replace("{{JSON_SCHEMA}}", schema_str)

    return (
        prompt_with_schema
        + "\n\n"
        + "=== SUPPLEMENTARY SECTION EXTRACTION RULES ===\n"
        + section_rules
        + "\n=== END SUPPLEMENTARY RULES ===\n"
        + "\n\n"
        + "=== FULL DOCUMENT TRANSCRIPT ===\n"
        + full_transcript
        + "\n=== END TRANSCRIPT ==="
    )


def run_pass2(
    llm: ChatOpenAI,
    full_transcript: str,
    schema_template: dict,
    extraction_prompt_raw: str,
    section_rules: str,
) -> dict:
    """
    Run Pass 2: map the full transcript to structured JSON.

    Chain: [SystemMessage, HumanMessage(user_content)] | llm | StrOutputParser
    Output is parsed with parse_json_robust() then normalised with enforce_schema().

    Args:
        llm:                   ChatOpenAI instance (max_tokens=8192).
        full_transcript:       Output of run_pass1().
        schema_template:       Parsed dict from json_schema_template.txt.
        extraction_prompt_raw: Raw text of directpass_sdr_extraction_prompt.txt.
        section_rules:         Raw text of onepass_section_rules.txt.

    Returns:
        Schema-enforced dict ready to write to JSON.
    """
    schema_str   = json.dumps(schema_template, indent=2)
    user_content = build_pass2_user_content(
        extraction_prompt_raw, section_rules, schema_str, full_transcript
    )

    chain = llm | StrOutputParser()
    raw   = chain.invoke([
        SystemMessage(content=PASS2_SYSTEM),
        HumanMessage(content=user_content),
    ])

    print(f"  [OK] JSON response received — {len(raw):,} chars")
    parsed = parse_json_robust(raw)
    return enforce_schema(parsed, schema_template)


# ---------------------------------------------------------------------------
# 9. Pipeline orchestration
# ---------------------------------------------------------------------------

def extract_sdr(
    pdf_path: Path,
    schema_path: Path,
    extraction_prompt_path: Path,
    section_rules_path: Path,
    transcription_system_path: Path,
    transcription_user_path: Path,
    results_dir: Path,
    dpi: int = DEFAULT_DPI,
    max_width: int = DEFAULT_MAX_WIDTH,
    save_transcript: bool = True,
) -> dict:
    """
    Full two-pass SDR extraction pipeline for a single PDF.

    Steps:
        1. Load schema, prompts
        2. Render PDF pages with PyMuPDF
        3. Pass 1: transcribe each page (vision)
        4. Pass 2: map transcript to JSON (text)
        5. enforce_schema() on result
        6. Return final dict

    Raises:
        FileNotFoundError: If PDF, prompt, or schema file is missing.
        ValueError:        If JSON cannot be parsed after all retries.
        RuntimeError:      If FW_API_URL or FW_API_KEY is missing from environment.
    """
    # Load assets
    schema_template       = load_schema(schema_path)
    extraction_prompt_raw = load_text(extraction_prompt_path)
    section_rules         = load_text(section_rules_path)
    system_prompt         = load_text(transcription_system_path)
    user_prompt_template  = load_text(transcription_user_path)

    # Build LLM instances (different max_tokens per pass)
    llm_pass1 = build_llm(max_tokens=4096)
    llm_pass2 = build_llm(max_tokens=8192)

    # Render PDF
    print(f"\n[Step 1] Rendering PDF at {dpi} DPI...")
    pages = render_pdf_pages(pdf_path, dpi=dpi, max_width=max_width)
    print(f"  [OK] {len(pages)} page(s) rendered")

    # Pass 1
    stem = pdf_path.stem
    print(f"\n[Step 2] Pass 1 — Transcribing {len(pages)} page(s)...")
    full_transcript = run_pass1(
        llm_pass1, pages, system_prompt, user_prompt_template,
        stem, results_dir, save_transcript,
    )
    print(f"  [OK] Full transcript: {len(full_transcript):,} chars")

    # Pass 2
    print(f"\n[Step 3] Pass 2 — Mapping transcript to JSON...")
    result = run_pass2(
        llm_pass2, full_transcript, schema_template,
        extraction_prompt_raw, section_rules,
    )
    print("  [OK] Schema enforced")

    return result


def process_single_pdf(args: argparse.Namespace) -> Path:
    """Process one PDF. Returns the output JSON path."""
    pdf_path    = Path(args.pdf).resolve()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / f"{pdf_path.stem}_extraction.json"

    print("=" * 60)
    print(f"  SDR Extractor (LangChain) — {FW_MODEL_NAME}")
    print(f"  PDF: {pdf_path.name}")
    print("=" * 60)

    result = extract_sdr(
        pdf_path              = pdf_path,
        schema_path           = Path(args.schema_path),
        extraction_prompt_path= Path(args.extraction_prompt),
        section_rules_path    = Path(args.section_rules),
        transcription_system_path = Path(args.transcription_system),
        transcription_user_path   = Path(args.transcription_user),
        results_dir           = results_dir,
        dpi                   = args.dpi,
        max_width             = args.max_width,
        save_transcript       = not args.no_save_transcript,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[Step 4] Saved: {output_path}")
    preview = json.dumps(result, indent=2, ensure_ascii=False)[:1500]
    print(f"\n--- Preview (first 1500 chars) ---\n{preview}\n...")
    return output_path


def process_batch(args: argparse.Namespace) -> list:
    """Process all PDFs in args.docs_dir. Returns list of output paths."""
    docs_dir = Path(args.docs_dir)
    pdfs     = sorted(docs_dir.glob("*.pdf"), key=lambda p: p.name.lower())

    if not pdfs:
        print(f"No PDF files found in: {docs_dir}")
        return []

    total   = len(pdfs)
    outputs = []
    for idx, pdf in enumerate(pdfs, start=1):
        print(f"\n[{idx}/{total}] Processing: {pdf.name}")
        args.pdf = str(pdf)
        out = process_single_pdf(args)
        outputs.append(out)

    print(f"\nBatch complete — {total} file(s) processed.")
    return outputs


# ---------------------------------------------------------------------------
# 10. CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LangChain SDR PDF Extractor — two-pass vision + text pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdf",           default=None,
                   help="Path to a single SDR PDF (single-file mode).")
    p.add_argument("--docs-dir",      default=str(ROOT / "docs"),
                   help="Directory of PDFs for batch mode.")
    p.add_argument("--output",        default=None,
                   help="Override output JSON path (single-file mode only).")
    p.add_argument("--results-dir",   default=str(ROOT / "results"),
                   help="Directory for output JSON and transcript files.")
    p.add_argument("--schema-path",   default=str(ROOT / "docs" / "json_schema_template.txt"),
                   help="Path to JSON schema file.")
    p.add_argument("--extraction-prompt",
                   default=str(ROOT / "prompts" / "directpass_sdr_extraction_prompt.txt"),
                   help="Pass 2 extraction prompt file.")
    p.add_argument("--section-rules",
                   default=str(ROOT / "prompts" / "onepass_section_rules.txt"),
                   help="Supplementary section extraction rules appended to Pass 2.")
    p.add_argument("--transcription-system",
                   default=str(ROOT / "prompts" / "transcription_system_prompt.txt"),
                   help="Pass 1 system prompt file.")
    p.add_argument("--transcription-user",
                   default=str(ROOT / "prompts" / "transcription_user_prompt.txt"),
                   help="Pass 1 user prompt file.")
    p.add_argument("--dpi",           type=int,  default=DEFAULT_DPI,
                   help="PDF render DPI (higher = sharper but slower).")
    p.add_argument("--max-width",     type=int,  default=DEFAULT_MAX_WIDTH,
                   help="Max rendered page width in pixels.")
    p.add_argument("--no-save-transcript", action="store_true",
                   help="Skip saving intermediate transcript .md file.")
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.pdf:
        process_single_pdf(args)
    else:
        process_batch(args)


if __name__ == "__main__":
    main()
