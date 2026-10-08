"""
Ollama Service Module
=====================
Direct local HTTP communication with Ollama (http://localhost:11434).
Zero OpenAI or third-party cloud API dependencies.
"""

import json
import time
import requests
from typing import Dict, Any, List, Optional
from prompt_generator import build_standard_prompt, build_json_schema_prompt

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:latest"


def check_ollama_health() -> Dict[str, Any]:
    """
    Checks the local Ollama daemon status and model availability.
    """
    start_time = time.time()
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        latency_ms = round((time.time() - start_time) * 1000, 1)

        if response.status_code == 200:
            data = response.json()
            models = [m.get("name") for m in data.get("models", [])]
            # Check if default model or any llama3.2 variant is present
            has_llama = any("llama3.2" in m for m in models)
            
            return {
                "connected": True,
                "status": "Connected",
                "latency_ms": latency_ms,
                "base_url": OLLAMA_BASE_URL,
                "models": models,
                "llama_available": has_llama,
                "selected_model": DEFAULT_MODEL if has_llama else (models[0] if models else DEFAULT_MODEL),
                "message": "Ollama is running locally and responsive." if has_llama else f"Ollama is running, but {DEFAULT_MODEL} was not found. Please run 'ollama pull {DEFAULT_MODEL}'."
            }
        else:
            return {
                "connected": False,
                "status": "Error",
                "latency_ms": latency_ms,
                "base_url": OLLAMA_BASE_URL,
                "models": [],
                "llama_available": False,
                "selected_model": DEFAULT_MODEL,
                "message": f"Ollama returned HTTP status {response.status_code}."
            }
    except requests.exceptions.ConnectionError:
        return {
            "connected": False,
            "status": "Not Connected",
            "latency_ms": None,
            "base_url": OLLAMA_BASE_URL,
            "models": [],
            "llama_available": False,
            "selected_model": DEFAULT_MODEL,
            "message": "Ollama is not running. Please start Ollama using 'ollama serve' and try again."
        }
    except Exception as e:
        return {
            "connected": False,
            "status": "Not Connected",
            "latency_ms": None,
            "base_url": OLLAMA_BASE_URL,
            "models": [],
            "llama_available": False,
            "selected_model": DEFAULT_MODEL,
            "message": f"Connection check failed: {str(e)}"
        }


def format_text_output(data: Dict[str, Any]) -> str:
    """
    Constructs the formatted legal analysis text as required by Experiment 5.
    """
    def list_or_str(val):
        if isinstance(val, list):
            return "\n".join([f"- {item}" if isinstance(item, str) else f"- {item.get('clause_title', '')}: {item.get('description', '')}" for item in val])
        return str(val) if val else "Not specified in the document."

    output_lines = [
        "LEGAL DOCUMENT ANALYSIS",
        "=" * 30,
        f"Document Type:\n{data.get('document_type', 'Not specified in the document.')}\n",
        f"Parties:\n{list_or_str(data.get('parties', []))}\n",
        f"Purpose:\n{data.get('purpose', 'Not specified in the document.')}\n",
        f"Key Obligations:\n{list_or_str(data.get('obligations', []))}\n",
        f"Rights:\n{list_or_str(data.get('rights', []))}\n",
        f"Payment Terms:\n{data.get('payment_terms', 'Not specified in the document.')}\n",
        f"Duration:\n{data.get('duration', 'Not specified in the document.')}\n",
        f"Termination Conditions:\n{data.get('termination_conditions', 'Not specified in the document.')}\n",
        f"Confidentiality:\n{data.get('confidentiality', 'Not specified in the document.')}\n",
        f"Important Dates:\n{list_or_str(data.get('important_dates', []))}\n",
        f"Governing Law:\n{data.get('governing_law', 'Not specified in the document.')}\n",
        f"Key Clauses:\n{list_or_str(data.get('key_clauses', []))}\n",
        f"Potentially Important Clauses:\n{list_or_str(data.get('potentially_important_clauses', []))}\n",
        f"Summary:\n{data.get('summary', 'Not specified in the document.')}"
    ]
    return "\n".join(output_lines)


def query_ollama_for_analysis(
    document_text: str,
    document_type: str = "Employment Agreement",
    analysis_type: str = "Complete Analysis",
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Sends the legal document prompt to Ollama and receives structured response.
    """
    # Verify health first
    health = check_ollama_health()
    if not health["connected"]:
        raise ConnectionError("Ollama is not running. Please start Ollama with 'ollama serve' and try again.")

    actual_model = model if model in health.get("models", []) else (health["models"][0] if health.get("models") else DEFAULT_MODEL)
    
    # 1. Generate the standard display prompt
    display_prompt = build_standard_prompt(document_text, document_type, analysis_type)
    
    # 2. Generate JSON-enforced prompt for reliable parsing
    json_prompt = build_json_schema_prompt(document_text, document_type, analysis_type)

    payload = {
        "model": actual_model,
        "prompt": json_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,  # Low temperature for deterministic factual extraction
            "num_predict": 1500
        }
    }

    start_time = time.time()
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120.0
        )
        elapsed_sec = round(time.time() - start_time, 2)

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API returned error {response.status_code}: {response.text}")

        res_json = response.json()
        raw_ollama_output = res_json.get("response", "")
        
        # Parse JSON output from model
        try:
            parsed_data = json.loads(raw_ollama_output)
        except Exception:
            # Fallback if raw JSON is wrapped
            import re
            match = re.search(r"\{.*\}", raw_ollama_output, re.DOTALL)
            if match:
                parsed_data = json.loads(match.group(0))
            else:
                parsed_data = {
                    "document_type": document_type,
                    "summary": raw_ollama_output,
                    "raw_formatted_analysis": raw_ollama_output
                }

        # Build clean formatted text response
        formatted_text = format_text_output(parsed_data)

        # Ensure all standard keys exist
        standardized_result = {
            "document_type": parsed_data.get("document_type") or document_type,
            "parties": parsed_data.get("parties", []),
            "purpose": parsed_data.get("purpose", "Not specified in the document."),
            "obligations": parsed_data.get("obligations", []),
            "rights": parsed_data.get("rights", []),
            "payment_terms": parsed_data.get("payment_terms", "Not specified in the document."),
            "duration": parsed_data.get("duration", "Not specified in the document."),
            "termination_conditions": parsed_data.get("termination_conditions", "Not specified in the document."),
            "confidentiality": parsed_data.get("confidentiality", "Not specified in the document."),
            "important_dates": parsed_data.get("important_dates", []),
            "governing_law": parsed_data.get("governing_law", "Not specified in the document."),
            "key_clauses": parsed_data.get("key_clauses", []),
            "potentially_important_clauses": parsed_data.get("potentially_important_clauses", []),
            "summary": parsed_data.get("summary", "No summary generated.")
        }

        # Normalization for arrays if LLM returned strings
        for list_key in ["parties", "obligations", "rights", "important_dates", "potentially_important_clauses"]:
            val = standardized_result[list_key]
            if isinstance(val, str):
                standardized_result[list_key] = [v.strip("- ") for v in val.split("\n") if v.strip()]
            elif not isinstance(val, list):
                standardized_result[list_key] = [str(val)]

        return {
            "success": True,
            "model_used": actual_model,
            "execution_time_sec": elapsed_sec,
            "display_prompt": display_prompt,
            "raw_ollama_response": formatted_text,
            "structured_analysis": standardized_result,
            "summary": standardized_result["summary"],
            "ollama_eval_metrics": {
                "eval_count": res_json.get("eval_count"),
                "eval_duration_ms": round(res_json.get("eval_duration", 0) / 1e6, 2),
                "total_duration_ms": round(res_json.get("total_duration", 0) / 1e6, 2)
            }
        }

    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama request timed out after 120 seconds. The document might be too long or the system is busy.")
    except Exception as e:
        raise e
