"""
Prompt Generator Module
=======================
Constructs structured, deterministic prompts for the local Ollama LLM
following legal document analysis guidelines.
"""

from typing import Dict, Any


def build_standard_prompt(document_text: str, document_type: str = "Employment Agreement", analysis_type: str = "Complete Analysis") -> str:
    """
    Builds the standard prompt specified in Experiment 5.
    """
    prompt = f"""ROLE:
You are an automated legal document analysis assistant.

TASK:
Analyze the provided legal document.

DOCUMENT TYPE:
{document_type}

ANALYSIS TYPE:
{analysis_type}

DOCUMENT:
{document_text}

Extract the following information where available:

1. Document Type
2. Parties
3. Purpose
4. Obligations
5. Rights
6. Payment Terms
7. Duration
8. Termination Conditions
9. Confidentiality
10. Important Dates
11. Governing Law
12. Key Clauses
13. Potentially Important Clauses
14. Concise Summary

IMPORTANT RULES:

- Use only information present in the document.
- Do not invent facts.
- If information is not available, write "Not specified in the document."
- Keep the summary concise.
- Clearly separate each extracted field.
- Return a structured response.
"""
    return prompt


def build_json_schema_prompt(document_text: str, document_type: str = "Employment Agreement", analysis_type: str = "Complete Analysis") -> str:
    """
    Builds the prompt with explicit JSON instruction for Ollama's JSON format mode.
    Ensures deterministic keys for high-fidelity structured frontend rendering.
    """
    prompt = f"""ROLE:
You are an automated legal document analysis assistant.

TASK:
Analyze the provided legal document and return a valid JSON object.

DOCUMENT TYPE:
{document_type}

ANALYSIS TYPE:
{analysis_type}

DOCUMENT:
{document_text}

Extract the following legal information and return ONLY valid JSON matching this exact structure:
{{
  "document_type": "...",
  "parties": ["..."],
  "purpose": "...",
  "obligations": ["..."],
  "rights": ["..."],
  "payment_terms": "...",
  "duration": "...",
  "termination_conditions": "...",
  "confidentiality": "...",
  "important_dates": ["..."],
  "governing_law": "...",
  "key_clauses": [
    {{"clause_title": "...", "description": "..."}}
  ],
  "potentially_important_clauses": ["..."],
  "summary": "...",
  "raw_formatted_analysis": "..."
}}

IMPORTANT RULES:
- Use only information present in the document.
- Do not invent facts.
- If information is not available, write "Not specified in the document."
- Keep the summary concise and accurate.
- Ensure all JSON keys are present.
"""
    return prompt
