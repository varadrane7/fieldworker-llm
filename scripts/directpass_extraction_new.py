import os
import json
import re
import base64
import argparse
import time
from copy import deepcopy

from dotenv import load_dotenv
from openai import OpenAI


## Config

MODEL_NAME = "gpt-4.1"
DEFAULT_DOCS_DIR = "docs"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_PROMPT_PATH = "prompts/merged_extraction_prompt.txt"
DEFAULT_SCHEMA_PATH = "docs/json_schema_template.txt"


## Env / client
## Creating proper connection to the company's API.

def build_client():
    load_dotenv()

    endpoint = os.getenv("FW_API_URL")
    api_key = os.getenv("FW_API_KEY")

    if not endpoint:
        raise ValueError("FW_API_URL not found in .env")
    if not api_key:
        raise ValueError("FW_API_KEY not found in .env")

    return OpenAI(
        base_url=endpoint,
        api_key=api_key
    )


## File helpers

def read_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_schema(schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


## Encoding the PDF file into base64 format for the API.

def encode_file_base64(file_path):
    with open(file_path, "rb") as f:
        base64_data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:application/pdf;base64,{base64_data}"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


## Prompt builder
## This function builds the prompt (with the JSON schema template) that will be sent to the API.

def build_prompt(prompt_template, schema_dict):
    schema_str = json.dumps(schema_dict, indent=2, ensure_ascii=False)

    return (
        prompt_template
        .replace("{{JSON_SCHEMA}}", schema_str)
    )


## JSON parsing
## This function cleans model output and ensures safe JSON load.

def extract_json_from_text(text):
    text = text.strip()

    ## Removing markdown fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    ## Trying the direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    ## Fallback --> finding first JSON object block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        return json.loads(candidate)

    raise ValueError("Could not parse JSON from model response.")


## Schema enforcement
## This function enforces the schema on the model output.

def enforce_schema(model_output, schema_template):
    """
    Keep only keys that exist in schema_template (no hallucinated keys).
    Fill missing keys from schema_template defaults.
    Recurse for dict/list structure.
    """

    if isinstance(schema_template, dict):
        result = {}

        if not isinstance(model_output, dict):
            model_output = {}

        for key, schema_value in schema_template.items():
            output_value = model_output.get(key, None)
            result[key] = enforce_schema(output_value, schema_value)

        return result

    if isinstance(schema_template, list):
        ## If schema list is empty, I preserve output only if it is list, else [].
        if len(schema_template) == 0:
            return model_output if isinstance(model_output, list) else []

        item_schema = schema_template[0]

        if not isinstance(model_output, list):
            return []

        return [enforce_schema(item, item_schema) for item in model_output]

    ## Primitive values --> if missing/None, I use schema default.
    if model_output is None:
        return deepcopy(schema_template)

    ## Optional light normalization for primitive mismatch.
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


## Model call
## This function calls the model with the PDF and the final prompt.
## One-pass extraction; no PDF parsing libraries; model vision + text understanding.

def call_model_with_pdf(client, pdf_path, final_prompt):
    base64_pdf = encode_file_base64(pdf_path)
    filename = os.path.basename(pdf_path)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful structured-data extraction agent. "
                    "You must extract only information grounded in the provided PDF. "
                    "Do not invent fields, values, section names, or summaries. "
                    "Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": final_prompt,
                    },
                    {
                        "type": "file",
                        "file": {
                            "filename": filename,
                            "file_data": base64_pdf,
                        },
                    },
                ],
            },
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty content.")

    return content


## Single PDF pipeline
## This function runs the entire pipeline for a single PDF file.

def process_single_pdf(client, pdf_path, prompt_path, schema_path, results_dir):
    prompt_template = read_text_file(prompt_path)
    schema_template = load_schema(schema_path)
    final_prompt = build_prompt(prompt_template, schema_template)

    raw_text = call_model_with_pdf(client, pdf_path, final_prompt)
    parsed_output = extract_json_from_text(raw_text)
    final_output = enforce_schema(parsed_output, schema_template)

    ensure_dir(results_dir)
    output_name = os.path.splitext(os.path.basename(pdf_path))[0] + "_merged_directpass_extraction.json"
    output_path = os.path.join(results_dir, output_name)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    return output_path


## Batch pipeline
## This function processes all PDF files in a directory as a batch.

def process_all_pdfs(client, docs_dir, prompt_path, schema_path, results_dir):
    ensure_dir(results_dir)

    pdf_files = sorted(
        [
            os.path.join(docs_dir, name)
            for name in os.listdir(docs_dir)
            if name.lower().endswith(".pdf")
        ]
    )

    if not pdf_files:
        raise ValueError(f"No PDF files found in {docs_dir}")

    saved_paths = []
    for pdf_path in pdf_files:
        saved_path = process_single_pdf(
            client=client,
            pdf_path=pdf_path,
            prompt_path=prompt_path,
            schema_path=schema_path,
            results_dir=results_dir,
        )
        saved_paths.append(saved_path)

    return saved_paths


## CLI
## This function runs the CLI for the script.

def main():
    parser = argparse.ArgumentParser(description="One-pass SDR PDF to JSON extractor")
    parser.add_argument("--pdf", type=str, default=None, help="Path to one PDF file")
    parser.add_argument("--docs_dir", type=str, default=DEFAULT_DOCS_DIR, help="Directory containing PDFs")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Directory for output JSON files")
    parser.add_argument("--prompt_path", type=str, default=DEFAULT_PROMPT_PATH, help="Path to extraction prompt txt")
    parser.add_argument("--schema_path", type=str, default=DEFAULT_SCHEMA_PATH, help="Path to JSON schema template")

    args = parser.parse_args()

    client = build_client()

    if args.pdf:
        output_path = process_single_pdf(
            client=client,
            pdf_path=args.pdf,
            prompt_path=args.prompt_path,
            schema_path=args.schema_path,
            results_dir=args.results_dir,
        )
        print(f"Saved: {output_path}")
    else:
        saved_paths = process_all_pdfs(
            client=client,
            docs_dir=args.docs_dir,
            prompt_path=args.prompt_path,
            schema_path=args.schema_path,
            results_dir=args.results_dir,
        )
        for path in saved_paths:
            print(f"Saved: {path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.4f} seconds")