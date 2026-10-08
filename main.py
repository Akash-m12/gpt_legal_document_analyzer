"""
Automated Legal Document Analyzer - FastAPI Backend
===================================================
Experiment 5: Automated Legal Document Analyzer using Ollama & Llama 3.2
Zero cloud/OpenAI dependencies - 100% Local Execution.
"""

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from document_processor import (
    extract_document_metrics,
    preprocess_legal_text,
    chunk_document
)
from prompt_generator import build_standard_prompt
from ollama_service import (
    check_ollama_health,
    query_ollama_for_analysis,
    DEFAULT_MODEL
)

app = FastAPI(
    title="Automated Legal Document Analyzer API",
    description="Backend API for Legal Document Analysis powered by local Ollama (Llama 3.2)",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_DOC_PATH = os.path.join(os.path.dirname(__file__), "sample_document.txt")


# Pydantic Schemas
class AnalyzeRequest(BaseModel):
    document_text: str = Field(..., description="Raw or preprocessed legal document text")
    document_type: str = Field(default="Employment Agreement", description="Type of legal document")
    analysis_type: str = Field(default="Complete Analysis", description="Analysis scope")
    model: Optional[str] = Field(default=DEFAULT_MODEL, description="Ollama model to use")
    enable_chunking: Optional[bool] = Field(default=False, description="Whether to chunk long documents")


class PreprocessRequest(BaseModel):
    document_text: str


class PromptRequest(BaseModel):
    document_text: str
    document_type: str = "Employment Agreement"
    analysis_type: str = "Complete Analysis"


@app.get("/")
def root():
    return {
        "title": "Automated Legal Document Analyzer",
        "experiment": "Experiment 5 - Legal Document Analyzer using Ollama",
        "engine": "Ollama (Llama 3.2 local)",
        "status": "online"
    }


@app.get("/health")
def get_health():
    """
    Health check endpoint verifying local Ollama connectivity.
    """
    return check_ollama_health()


@app.get("/sample-document")
def get_sample_document():
    """
    Returns the standard laboratory sample Employment Agreement.
    """
    if os.path.exists(SAMPLE_DOC_PATH):
        with open(SAMPLE_DOC_PATH, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = """EMPLOYMENT AGREEMENT

This Employment Agreement is between ABC Technologies Pvt. Ltd. and John Smith.

The employee will work as a Software Developer.

The employment period is 12 months starting from 01-07-2026.

The employee will receive a monthly salary of ₹50,000.

The employee must maintain confidentiality regarding company information.

The employee must perform assigned duties and follow company policies.

Either party may terminate the agreement by providing 30 days written notice.

Any intellectual property created during employment belongs to the company.

The agreement is governed by the applicable laws of India."""

    metrics = extract_document_metrics(content, filename="sample_document.txt")
    return {
        "filename": "sample_document.txt",
        "document_type": "Employment Agreement",
        "content": content,
        "metrics": metrics
    }


@app.post("/extract-file")
async def extract_from_file(file: UploadFile = File(...)):
    """
    Extracts text from an uploaded .txt legal document file.
    """
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported for legal document upload.")

    try:
        content_bytes = await file.read()
        # Decode utf-8 with fallback
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = content_bytes.decode("latin-1")

        if not text.strip():
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        metrics = extract_document_metrics(text, filename=file.filename)
        return {
            "filename": file.filename,
            "text": text,
            "metrics": metrics
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/preprocess")
def preprocess_text(payload: PreprocessRequest):
    """
    Cleans, normalizes and analyzes text preprocessing reduction metrics.
    """
    if not payload.document_text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty.")

    result = preprocess_legal_text(payload.document_text)
    return result


@app.post("/generate-prompt")
def generate_prompt_endpoint(payload: PromptRequest):
    """
    Generates the structured legal prompt for UI inspection.
    """
    prompt = build_standard_prompt(
        document_text=payload.document_text,
        document_type=payload.document_type,
        analysis_type=payload.analysis_type
    )
    return {
        "prompt": prompt,
        "document_type": payload.document_type,
        "analysis_type": payload.analysis_type,
        "char_count": len(prompt),
        "word_count": len(prompt.split())
    }


@app.post("/analyze")
def analyze_document(payload: AnalyzeRequest):
    """
    Runs the complete 7-stage automated legal analysis pipeline:
    1. Input Validation
    2. Document Text Extraction & Metrics
    3. Preprocessing & Normalization
    4. Prompt Generation
    5. Ollama LLM Execution (Llama 3.2)
    6. Structured Entity & Clause Extraction
    7. Executive Summary Generation
    """
    raw_text = payload.document_text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Please provide or paste a legal document text.")

    # Stage 1: Extraction & Metrics
    extraction_metrics = extract_document_metrics(raw_text)

    # Stage 2: Preprocessing
    preprocess_result = preprocess_legal_text(raw_text)
    cleaned_text = preprocess_result["cleaned_text"]

    # Stage 3: Chunking (if text is large or chunking requested)
    chunks = chunk_document(cleaned_text)
    is_chunked = len(chunks) > 1

    # Stage 4: Prompt Construction
    prompt = build_standard_prompt(
        document_text=cleaned_text,
        document_type=payload.document_type,
        analysis_type=payload.analysis_type
    )

    # Stage 5 & 6 & 7: Local Ollama Invocation & Structured Extraction
    try:
        ollama_result = query_ollama_for_analysis(
            document_text=cleaned_text,
            document_type=payload.document_type,
            analysis_type=payload.analysis_type,
            model=payload.model or DEFAULT_MODEL
        )
    except ConnectionError as ce:
        raise HTTPException(status_code=503, detail=str(ce))
    except TimeoutError as te:
        raise HTTPException(status_code=504, detail=str(te))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Assemble comprehensive response payload for frontend visualization
    return {
        "success": True,
        "stage_1_extraction": {
            "title": "Document Text Extraction",
            "raw_text": raw_text,
            "metrics": extraction_metrics
        },
        "stage_2_preprocessing": {
            "title": "Preprocessed Text",
            "cleaned_text": cleaned_text,
            "original_chars": preprocess_result["original_chars"],
            "cleaned_chars": preprocess_result["cleaned_chars"],
            "original_words": preprocess_result["original_words"],
            "cleaned_words": preprocess_result["cleaned_words"],
            "diff_chars": preprocess_result["diff_chars"],
            "reduction_percentage": preprocess_result["reduction_percentage"],
            "steps": preprocess_result["steps"]
        },
        "stage_3_chunks": {
            "is_chunked": is_chunked,
            "chunk_count": len(chunks),
            "chunks": chunks
        },
        "stage_4_prompt": {
            "title": "Prompt Generation",
            "prompt": prompt,
            "document_type": payload.document_type,
            "analysis_type": payload.analysis_type
        },
        "stage_5_ollama_response": {
            "title": "Ollama Response",
            "model": ollama_result["model_used"],
            "execution_time_sec": ollama_result["execution_time_sec"],
            "raw_response": ollama_result["raw_ollama_response"],
            "eval_metrics": ollama_result.get("ollama_eval_metrics", {})
        },
        "stage_6_summary": {
            "title": "Legal Document Summary",
            "summary": ollama_result["summary"]
        },
        "stage_7_structured_analysis": {
            "title": "Structured Legal Analysis",
            "data": ollama_result["structured_analysis"]
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
