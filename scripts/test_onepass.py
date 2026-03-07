import os
import json
import base64
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

import time

## This is a trial one-pass extraction script.
## It is not production-ready.


## Config

MODEL_NAME = "gpt-4.1"
PROMPTS_DIR = Path("prompts")
#OUTPUT_DIR = Path("outputs")
#OUTPUT_DIR.mkdir(exist_ok=True)

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)


## JSON template

JSON_SCHEMA = {
    "document_type": "Service Data Record (SDR)",
    "plan_id": "",
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
        "waiver_enrollment_date": "",
	    "contact_details": {
    	    "address": "",
    	    "home_phone": "",
    	    "work_phone": "",
    	    "cell_phone": "",
    	    "email": ""
	    }
    },
    "guardianship_contacts": [
        {
            "name": "", 
            "relationship": "", 
            "contact_details": {
                "address": "",
                "home_phone": "",
                "work_phone": "",
                "cell_phone": "",
                "email": ""
            }
        }
    ],
    "emergency_contacts": [
        {
            "priority": "", 
            "name": "", 
            "relationship": "", 
            "contact_details": {
                "primary_phone": "",
                "secondary_phone": "",
            }
        }
    ],
    "medical_practitioners": [{"name": "", "specialty": "", "phone_number": "", "notes": ""}],
    "preferred_hospital": {"name": "", "contact_details": {"address": "", "phone_number": ""}, "notes": ""},
    "primary_care_physician": {"name": "", "contact_details": {"address": "", "phone_number": ""}, "notes": ""},
    "administrative_service_organization": {"name": "", "cm_name": "", "id_group": "", "contact_details": {"cell_phone": "", "email": ""}},
    "managed_care_organization": {"name": "", "cm_name": "", "id_group": "", "contact_details": {"cell_phone": "", "email": ""}},
    "private_insurance": {"name": "", "cm_name": "", "id_group": "", "contact_details": {"cell_phone": "", "email": ""}},
    "diagnosis_information": {
        "diagnoses": [
            {
                "diagnosis_type": "",
                "icd_code": "",
                "description": ""
            }
        ]
    },
    "support_coordination_company": {
        "company_name": "",
        "company_phone": "",
        "company_email": "", 
        "sc_name": "",
        "sc_email": "",
        "scs_name": "",
        "scs_email": ""
    },
    "service_authorizations": [
        {
            "service_name": "",
            "procedure_code": "",
            "procedure_tier": "",
            "reference": "",
            "start_date": "",
            "end_date": "",
            "total_units": "",
            "total_cost": "",
            "rate": "",
            "unit_type": "",
            "frequency": "",
            "service_location": "",
            "provider_name": "",
            "service_note": "",
            "claims_information": "",
            "associated_goals": [{"outcome_number": "", "outcome_description": ""}]
        }
    ],
    "medication_info": [{"order": "", "medication_name": "", "dosage_info": "", "frequency_info": "", "medication_notes": "", "self_medication": ""}],
    "customer_goals": [{"outcome_number": "", "outcome_description": ""}],
    "customer_needs": {
        "health_and_nutrition_needs": [
            {
                "category_name": "", 
                "subcategories": [
                    {
                        "subcategory_name": "", 
                        "need_description": ""
                    }
                ]
            }
        ],
        "safety_and_support_needs": [
            {
                "category_name": "", 
                "subcategories": [
                    {
                        "subcategory_name": "", 
                        "need_description": ""
                    }
                ]
            }
        ],
    },
    "employment_and_voting": {
        "employment_status": "",
        "employment_history_or_context": "",
        "employment_plan": "",
        "voting_plan": ""
    },
    "team_members_present": [
        {
            "name": "",
            "relationship_or_agency": "",
            "primary_contact": ""
        }
    ],
    "authorizations_and_signatures": {
        "attestation_text": "",
        "support_coordinator_review_note": "",
        "participant": {
            "name": "",
            "signature_present": False,
            "date": ""
        },
        "guardian_legal_representative": {
            "name": "",
            "signature_present": False,
            "date": ""
        }
    }
}


## Prompt Loading

def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_prompts() -> Dict[str, str]:
    return {
        "system": read_text_file(PROMPTS_DIR / "onepass_system_prompt.txt"),
        "instructions": read_text_file(PROMPTS_DIR / "onepass_extraction_instructions.txt"),
        "rules": read_text_file(PROMPTS_DIR / "onepass_section_rules.txt"),
        "repair": read_text_file(PROMPTS_DIR / "onepass_repair_prompt.txt"),
    }


## Helpers

def pdf_to_base64_data_url(pdf_path: Path) -> str:
    with open(pdf_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:application/pdf;base64,{encoded}"


def extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Safely extract the first JSON object from a model response.
    """
    text = text.strip()

    ## Here, I try to extract the JSON object from the model response.
    ## First, I try to directly parse the text.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    ## Then, I try to remove the code fences.
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

    ## Fallback --> find outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find a valid JSON object in the model response.")

    return json.loads(text[start:end + 1])


def sanitize_scalar(template_value: Any, actual_value: Any) -> Any:
    """
    Match scalar types in a forgiving way.
    """
    if isinstance(template_value, bool):
        if isinstance(actual_value, bool):
            return actual_value
        if isinstance(actual_value, str):
            return actual_value.strip().lower() in {"true", "yes", "present", "signed", "1"}
        return bool(actual_value)

    if actual_value is None:
        return "" if isinstance(template_value, str) else template_value

    if isinstance(template_value, str):
        return str(actual_value).strip()

    return actual_value


def enforce_template(template: Any, data: Any) -> Any:
    """
    Force model output into the exact target shape.
    - Missing keys are filled from the template
    - Extra keys are dropped
    - Nested lists/dicts are normalized
    """
    if isinstance(template, dict):
        result = {}
        source = data if isinstance(data, dict) else {}
        for key, template_value in template.items():
            result[key] = enforce_template(template_value, source.get(key))
        return result

    if isinstance(template, list):
        prototype = template[0] if template else None

        if not isinstance(data, list):
            return []

        if prototype is None:
            return data

        return [enforce_template(prototype, item) for item in data]

    return sanitize_scalar(template, data)


def prune_placeholder_rows(data: Any, template: Any) -> Any:
    """
    Remove placeholder-style empty rows from list outputs after normalization.
    """
    if isinstance(template, dict) and isinstance(data, dict):
        return {
            k: prune_placeholder_rows(data[k], template[k])
            for k in template
        }

    if isinstance(template, list) and isinstance(data, list):
        prototype = template[0] if template else None
        cleaned = [prune_placeholder_rows(item, prototype) for item in data]
        return [item for item in cleaned if not is_effectively_empty(item)]

    return data


def is_effectively_empty(value: Any) -> bool:
    if isinstance(value, dict):
        return all(is_effectively_empty(v) for v in value.values())
    if isinstance(value, list):
        return all(is_effectively_empty(v) for v in value)
    if isinstance(value, bool):
        return value is False
    return str(value).strip() == ""


def build_user_prompt(schema_template: Dict[str, Any], prompts: Dict[str, str]) -> str:
    return f"""
        {prompts['instructions']}

        {prompts['rules']}

        JSON TEMPLATE TO FILL:
        {json.dumps(schema_template, indent=2)}

        Return only one valid JSON object matching this exact structure.
        Do not wrap it in markdown.
        Do not add commentary before or after the JSON.
        """.strip()


def call_extraction_model(client: OpenAI, pdf_path: Path, prompts: Dict[str, str]) -> str:
    pdf_data_url = pdf_to_base64_data_url(pdf_path)
    user_prompt = build_user_prompt(JSON_SCHEMA, prompts)

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompts["system"]
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": pdf_path.name,
                        "file_data": pdf_data_url
                    },
                    {
                        "type": "input_text",
                        "text": user_prompt
                    }
                ]
            }
        ]
    )

    return response.output_text


def call_repair_model(client: OpenAI, raw_text: str, prompts: Dict[str, str]) -> str:
    """
    Fallback only if the first response is not valid JSON.
    This is not a second semantic extraction pass. It is only a JSON repair pass.
    """
    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompts["repair"]
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""
                            Fix the following model output into one valid JSON object only.

                            TARGET TEMPLATE:
                            {json.dumps(JSON_SCHEMA, indent=2)}

                            BROKEN OUTPUT:
                            {raw_text}
                            """.strip()
                    }
                ]
            }
        ]
    )

    return response.output_text


def extract_sdr_to_json(pdf_path: str, output_path: str) -> Dict[str, Any]:
    api_url = os.getenv("FW_API_URL")
    api_key = os.getenv("FW_API_KEY")

    if not api_url or not api_key:
        raise EnvironmentError("FW_API_URL / FW_API_KEY is not set in .env")

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    prompts = load_prompts()
    client = OpenAI(
        api_key=api_key,
        base_url=api_url
    )

    raw_response = call_extraction_model(client, pdf_file, prompts)

    try:
        parsed = extract_first_json_object(raw_response)
    except Exception:
        repaired = call_repair_model(client, raw_response, prompts)
        parsed = extract_first_json_object(repaired)

    normalized = enforce_template(deepcopy(JSON_SCHEMA), parsed)
    cleaned = prune_placeholder_rows(normalized, JSON_SCHEMA)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8")

    return cleaned


## Main / Entry point

if __name__ == "__main__":

    INPUT_PDF = "docs/1.pdf"
    OUTPUT_JSON = "results/onepass_extraction.json"

    start = time.time()
    
    result = extract_sdr_to_json(INPUT_PDF, OUTPUT_JSON)
    
    time.sleep(1)
    end = time.time()
    #print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nTotal runtime of the program is {end - start:.4f} seconds.")
    print(f"\nSaved JSON to: {OUTPUT_JSON}")