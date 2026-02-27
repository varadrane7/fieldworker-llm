import argparse
import base64
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

## I tried to use PyMuPDF (fitz) to render PDF pages to images without Poppler.
import fitz
from PIL import Image

## 1) JSON Schema
## JSON Schema based on the schema (required keys) Kaushik shared with us.
JSON_SCHEMA: Dict[str, Any] = {
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
        "ddd_status": ""
    },
    "diagnosis_information": {
        "primary_diagnosis": {"icd_code": "", "description": ""},
        "secondary_diagnoses": [{"icd_code": "", "description": ""}]
    },
    "provider_company": {
        "company_name": "",
        "billing_npi": "",
        "mailing_address": {"street": "", "city": "", "state": "", "zip": ""}
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
            "source": ""
        }
    ],
    "customer_goals": [{"outcome_number": "", "outcome_description": ""}],
    "customer_needs": {
        "health_needs": [{"need_description": ""}],
        "support_needs": [{"need_description": ""}],
        "employment_needs": [{"need_description": ""}]
    }
}


## 2) Loading the environment variables and paths.
## The below helpers are to load env from repo root safely and read prompts.
def repo_root() -> Path:
    """Repo root is one level above /scripts."""
    return Path(__file__).resolve().parents[1]


def load_env() -> None:
    """
    Load .env from repo root.
    """
    env_path = repo_root() / ".env"
    load_dotenv(env_path)

    if not os.getenv("FW_API_URL") or not os.getenv("FW_API_KEY"):
        raise RuntimeError("Missing FW_API_URL / FW_API_KEY. Put them in repo root .env")


def read_prompt(prompt_name: str) -> str:
    """
    Read a prompt text file from /prompts.
    Example: read_prompt('test_system_prompt_v1.txt')
    """
    p = repo_root() / "prompts" / prompt_name
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8")

## 3) Rendering PDF pages --> PIL Images (no Poppler needed)
## The below helpers are to render PDF pages to images and convert them to base64.
def render_pdf_to_images(pdf_path: Path, dpi: int = 200, max_width: int = 1600) -> List[Image.Image]:
    """
    Render each PDF page to a PIL image using PyMuPDF.

    - dpi controls clarity (higher dpi = clearer text but bigger images).
    - max_width downsizes very large pages for speed.
    """
    doc = fitz.open(pdf_path)

    zoom = dpi / 72.0  ## PDF default is 72 dpi.
    mat = fitz.Matrix(zoom, zoom)
    
    images: List[Image.Image] = []

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        ## Downscaling huge pages for speed/size.
        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        images.append(img)

    return images


def pil_to_base64_jpeg(img: Image.Image, quality: int = 85) -> str:
    """
    Convert PIL image -> base64-encoded JPEG for the vision API.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


## 4) Robust JSON extraction from model response
## The below helpers are to extract JSON from a model response, even if it includes code fences.
def strip_code_fences(text: str) -> str:
    """Remove ``` fences if the model adds them."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    return fence.group(1).strip() if fence else text


def parse_json_response(text: str) -> Dict[str, Any]:
    """Extract JSON object from model output and parse."""
    text = strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start:end])


## This is the two-pass approach (I tried another approach first -direct vision to JSON-, 
## but it failed, so I did not push that code).

## 5) Pass 1 --> page-by-page transcription
def transcribe_page(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    img: Image.Image,
    page_num: int
) -> str:
    """
    Send ONE page image to the model and get a Markdown transcription.
    Page-by-page is critical for long documents.
    """
    b64 = pil_to_base64_jpeg(img)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{user_prompt}\n\nThis is page {page_num}. Begin:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
                    }
                ],
            },
        ],
        temperature = 0,
        max_tokens = 4096,
    )

    text = resp.choices[0].message.content or ""
    return strip_code_fences(text)


def build_full_transcript(page_texts: List[str]) -> str:
    """Combine per-page transcripts into one big transcript with clear delimiters."""
    parts = []
    for i, t in enumerate(page_texts, start=1):
        parts.append(f"=== PAGE {i} ===\n{t}\n=== END PAGE {i} ===")
    return "\n\n".join(parts)


## 6) Pass 2 --> transcript to JSON schema
def map_transcript_to_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt_template: str,
    transcript: str
) -> Dict[str, Any]:
    """
    Send transcript + schema and get structured JSON.
    """
    schema_str = json.dumps(JSON_SCHEMA, indent=2)
    user_prompt = user_prompt_template.format(schema=schema_str, transcript=transcript)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature = 0,
        max_tokens = 8192,
    )

    raw = resp.choices[0].message.content or ""
    return parse_json_response(raw)

## 7) Entry point / Main 
def main():
    ## Safer Windows UTF-8 prints to prevent cp1252 errors sometimes.
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Test SDR Extractor v1, prompts from /prompts (one-pass vision to JSON)")
    parser.add_argument("pdf", help="Path to SDR PDF in file in docs/, e.g. docs/1.pdf")
    parser.add_argument("--model", default="gpt-4.1", help="Model name (default: gpt-4.1)")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI (default: 200)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: results/<stem>_test_sdr_extractor_v1.json)")
    args = parser.parse_args()

    load_env()

    ## Loading prompts from /prompts, so prompts are not hardcoded in the script.
    #system_prompt = read_prompt("test_system_prompts_v1.txt")
    #user_prompt_template = read_prompt("test_user_prompt_v1.txt")

    transcribe_system = read_prompt("test_transcribe_page_v1.txt")
    map_system = read_prompt("test_system_prompts_v1.txt")
    map_user_template = read_prompt("test_map_transcript_to_json_v1.txt")

    # For the "user prompt" in transcription, I kept it short to reduce prompt bloat, but I still pushed .txt I used previously.
    transcribe_user = "Transcribe this page verbatim into Markdown following the rules."

    client = OpenAI(
        base_url=os.getenv("FW_API_URL"),
        api_key=os.getenv("FW_API_KEY")
    )

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    images = render_pdf_to_images(pdf_path, dpi=args.dpi)
    print(f"[OK] Rendered {len(images)} page(s) from: {pdf_path.name}")

    ## Pass 1 --> per-page transcription
    page_texts: List[str] = []
    for i, img in enumerate(images, start=1):
        print(f"[Pass 1] Transcribing page {i}/{len(images)}...")
        txt = transcribe_page(
            client=client,
            model=args.model,
            system_prompt=transcribe_system,
            user_prompt=transcribe_user,
            img=img,
            page_num=i
        )
        page_texts.append(txt)

    full_transcript = build_full_transcript(page_texts)
    print(f"[OK] Built transcript ({len(full_transcript)} chars)")

    ## Saving transcript for debug purposes. 
    results_dir = repo_root() / "results"
    results_dir.mkdir(exist_ok=True)
    transcript_path = results_dir / f"{pdf_path.stem}_test_sdr_extractor_v1_transcript.md"
    transcript_path.write_text(full_transcript, encoding="utf-8")
    print(f"[OK] Saved transcript: {transcript_path}")

    ## Pass 2 --> transcript to JSON
    print("[Pass 2] Mapping transcript to JSON...")
    result = map_transcript_to_json(
        client=client,
        model=args.model,
        system_prompt=map_system,
        user_prompt_template=map_user_template,
        transcript=full_transcript
    )
    out_path = Path(args.output) if args.output else (results_dir / f"{pdf_path.stem}_test_sdr_extraction_v1.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved JSON: {out_path}")


if __name__ == "__main__":
    main()