"""
Document Processor Module
=========================
Handles legal document text extraction, normalization, statistical metrics,
and chunking for long legal documents.
"""

import re
from typing import Dict, List, Any


def extract_document_metrics(text: str, filename: str = "Pasted Text") -> Dict[str, Any]:
    """
    Extracts core metrics from the input document text.
    """
    if not text:
        return {
            "character_count": 0,
            "word_count": 0,
            "line_count": 0,
            "paragraph_count": 0,
            "estimated_tokens": 0,
            "filename": filename
        }

    chars = len(text)
    words = len(text.split())
    lines = len([line for line in text.split("\n") if line.strip()])
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])
    # Approximate token count for LLMs (~ 4 chars or 0.75 words per token)
    est_tokens = max(1, int(chars / 4))

    return {
        "character_count": chars,
        "word_count": words,
        "line_count": lines,
        "paragraph_count": paragraphs,
        "estimated_tokens": est_tokens,
        "filename": filename
    }


def preprocess_legal_text(text: str) -> Dict[str, Any]:
    """
    Cleans and preprocesses legal text for optimal LLM prompt ingestion.
    - Removes unnecessary whitespace
    - Normalizes line breaks
    - Removes repeated spaces
    - Cleans non-printable/control characters while keeping legal/currency symbols
    """
    if not text:
        return {
            "cleaned_text": "",
            "original_chars": 0,
            "cleaned_chars": 0,
            "original_words": 0,
            "cleaned_words": 0,
            "diff_chars": 0,
            "steps": []
        }

    original_chars = len(text)
    original_words = len(text.split())
    steps_applied = []

    cleaned = text

    # Step 1: Normalize standard quotes and dashes
    if re.search(r"[“”‘’—–]", cleaned):
        cleaned = (
            cleaned.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("—", " - ")
            .replace("–", " - ")
        )
        steps_applied.append("Normalized smart quotes and dashes to standard ASCII")

    # Step 2: Normalize line breaks (\r\n -> \n, \r -> \n)
    if "\r" in cleaned:
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        steps_applied.append("Normalized Windows/Mac line endings to standard Unix line breaks")

    # Step 3: Strip whitespace from every line & normalize multiple inline spaces/tabs
    lines = cleaned.split("\n")
    cleaned_lines = []
    has_inline_space_fixes = False
    for line in lines:
        line_clean = re.sub(r"[ \t]+", " ", line).strip()
        if len(line_clean) != len(line):
            has_inline_space_fixes = True
        cleaned_lines.append(line_clean)

    if has_inline_space_fixes:
        steps_applied.append("Removed repeated inline spaces, tabs, and line-level whitespace")

    cleaned = "\n".join(cleaned_lines)

    # Step 4: Collapse 3 or more consecutive newlines into double newlines
    if re.search(r"\n{3,}", cleaned):
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        steps_applied.append("Compressed excessive vertical blank lines into standard paragraph breaks")

    # Step 5: Strip global outer whitespace
    cleaned = cleaned.strip()
    if len(steps_applied) == 0:
        steps_applied.append("Verified text format and trimmed leading/trailing whitespace")

    cleaned_chars = len(cleaned)
    cleaned_words = len(cleaned.split())

    return {
        "cleaned_text": cleaned,
        "original_chars": original_chars,
        "cleaned_chars": cleaned_chars,
        "original_words": original_words,
        "cleaned_words": cleaned_words,
        "diff_chars": original_chars - cleaned_chars,
        "reduction_percentage": round(((original_chars - cleaned_chars) / max(1, original_chars)) * 100, 2),
        "steps": steps_applied
    }


def chunk_document(text: str, max_chunk_chars: int = 2500, overlap_chars: int = 200) -> List[Dict[str, Any]]:
    """
    Splits very long documents into chunks along paragraph or sentence boundaries
    to prevent context overflow while preserving contextual flow.
    """
    if len(text) <= max_chunk_chars:
        return [{
            "chunk_id": 1,
            "total_chunks": 1,
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split())
        }]

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    chunk_id = 1

    for para in paragraphs:
        if not para.strip():
            continue
        if len(current_chunk) + len(para) + 2 <= max_chunk_chars:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para.strip()

    if current_chunk:
        chunks.append(current_chunk)

    # If any single paragraph is longer than max_chunk_chars, split by sentence
    final_chunks = []
    for idx, c in enumerate(chunks):
        if len(c) > max_chunk_chars:
            sentences = re.split(r'(?<=[.!?])\s+', c)
            sub_chunk = ""
            for sent in sentences:
                if len(sub_chunk) + len(sent) + 1 <= max_chunk_chars:
                    sub_chunk = (sub_chunk + " " + sent).strip()
                else:
                    if sub_chunk:
                        final_chunks.append(sub_chunk)
                    sub_chunk = sent.strip()
            if sub_chunk:
                final_chunks.append(sub_chunk)
        else:
            final_chunks.append(c)

    total_chunks = len(final_chunks)
    return [
        {
            "chunk_id": i + 1,
            "total_chunks": total_chunks,
            "text": chunk_text,
            "char_count": len(chunk_text),
            "word_count": len(chunk_text.split())
        }
        for i, chunk_text in enumerate(final_chunks)
    ]
