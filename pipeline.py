"""
=============================================================================
ADS Hackathon - Payer Policy Intelligence Pipeline
=============================================================================
Extracts 12 structured parameters + Access Score from Payer PA Policy PDFs
using RAG (Retrieval Augmented Generation) approach.

Architecture:
- PDF Parsing: PyMuPDF (fitz)
- Chunking: Hierarchical + Context-aware + Token-based with overlap
- Embedding: sentence-transformers/all-MiniLM-L6-v2
- Vector DB: FAISS (in-memory + persisted to disk)
- LLM: Groq (Llama 3.1 8B / Llama 3.3 70B) with retry & fallback
- LLM Judge: Validates retrieval quality & final answers
- Access Score: Hybrid (rule-based + LLM verification)
- Logging: JSON logs at every step
- Output: CSV matching submission format

Usage:
    python pipeline.py --input_dir <path_to_pdfs> --output_dir <output_path>
    python pipeline.py --input_dir <path_to_pdfs> --output_dir <output_path> --brands_file <csv/xlsx with filename,brand>
=============================================================================
"""

import os
import re
import sys
import json
import time
import random
import logging
import argparse
import hashlib
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# --- SSL Fix for Corporate Proxy ---
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['HF_HUB_DISABLE_SSL'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import httpx
_orig_client_init = httpx.Client.__init__
def _patched_client_init(self, *args, **kwargs):
    kwargs['verify'] = False
    _orig_client_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_client_init

_orig_async_init = httpx.AsyncClient.__init__
def _patched_async_init(self, *args, **kwargs):
    kwargs['verify'] = False
    _orig_async_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = _patched_async_init

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests
requests.packages.urllib3.disable_warnings()
_orig_request = requests.Session.request
def _patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return _orig_request(self, *args, **kwargs)
requests.Session.request = _patched_request
# --- End SSL Fix ---

# PDF parsing
import fitz  # PyMuPDF

# Data handling
import pandas as pd
import numpy as np

# Embeddings & Vector DB
from sentence_transformers import SentenceTransformer
import faiss

# LangChain / LangGraph orchestration
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# Retry
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# .env loading
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Groq API Key (loaded from .env file)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Model configuration with fallback
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

# Embedding model (local path for offline use; falls back to HuggingFace if not found)
_LOCAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "all-MiniLM-L6-v2")
EMBEDDING_MODEL = _LOCAL_MODEL_PATH if os.path.isdir(_LOCAL_MODEL_PATH) else "sentence-transformers/all-MiniLM-L6-v2"

# Chunking configuration
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 100
MAX_CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 400

# Retrieval configuration
TOP_K_RETRIEVAL = 15
RERANK_TOP_K = 8

# Rate limiting
MAX_RETRIES = 5
RETRY_WAIT_MIN = 10
RETRY_WAIT_MAX = 120
RATE_LIMIT_PAUSE = 65  # seconds to wait on rate limit

# LLM Judge sampling - set low for full dataset to save tokens
LLM_JUDGE_SAMPLE_RATE = 0.05  # judge only 5% of extractions in production
ENABLE_LLM_JUDGE = False  # set False to fully disable judge (saves ~40% tokens)
ENABLE_ACCESS_SCORE_LLM = False  # set False to use rule-based only (saves 1 LLM call per entry)

# Context size limits (token optimization)
MAX_CONTEXT_CHARS = 5000  # max chars sent to LLM per extraction (reduce if hitting limits)

# Parameters to extract
PARAMETERS = [
    "Age",
    "Step Therapy Requirements Documented in Policy",
    "Number of Steps through Brands",
    "Number of Steps through Generic",
    "Step through-Phototherapy",
    "TB Test required",
    "Quantity Limits",
    "Specialist Types",
    "Initial Authorization Duration(in-months)",
    "Reauthorization Duration(in-months)",
    "Reauthorization Required",
    "Reauthorization Requirements Documented in Policy",
]

# Business Rules Definitions (embedded for portability)
BUSINESS_RULES = {
    "Age": """Indicates whether the policy includes age-based eligibility criteria for the therapy. This
could include minimum or maximum age thresholds (e.g., "must be >=18 years") or age
specific subpopulations for which the drug is approved or restricted.

Aligned to output FDA labelled age for those where policy is not specifying the actual age
but just what it is indicated for. So age for Tremfya where mentioned as FDA labelled age
will become "FDA labelled age" instead of >=18 if the policy does not specify the numerical
threshold

If the policy lists requirements for two age groups, the parameter should capture the
youngest one.
""",

    "Step Therapy Requirements Documented in Policy": """All step therapy language from the policy, covering both indication/brand-specific steps and any universal criteria.
- Include phototherapy language if it appears within step statements.
- If policy distinguishes between moderate-to-severe and severe PsO, only capture moderate-to-severe criteria.
- Extract the full text of step requirements.""",

    "Number of Steps through Brands": """Count of branded/biologic steps required before the target drug can be approved.
- A preferred ustekinumab or adalimumab product counts as a branded step.
- If a drug class is referenced and target drug belongs to it, class-level step counts as branded.
- Union of universal criteria + indication-specific, joined by AND.
- Take least restrictive path (OR = fewer steps).
- Exclude phototherapy steps.
- Output "NA" if no branded steps required. Output integer count otherwise.""",

    "Number of Steps through Generic": """Count of non-biologic steps required before the target drug.
- Topical agents count as generic steps.
- If parent indication requires a step without naming brand/biologic, defaults to generic.
- Union of universal + indication-specific criteria, joined by AND.
- Take least restrictive path.
- Exclude phototherapy steps.
- Output "NA" if no generic steps required. Output integer count otherwise.""",

    "Step through-Phototherapy": """Whether policy requires stepping through phototherapy (including PUVA).
- Yes: phototherapy is a mandatory required step (not in OR statement)
- No: policy does not mention phototherapy as required step
- N/A: policy lists no criteria at all""",

    "TB Test required": """Whether policy requires a TB test for approval.
- Yes: TB test required
- No: TB test not required
- NA: not mentioned""",

    "Quantity Limits": """Only reference what is explicitly stated as a "quantity limit".
- Do NOT capture if explicitly stated as "dosage" or "dosing limit".
- Extract exact quantity limit text if present.
- Output "NA" if no quantity limits mentioned.""",

    "Specialist Types": """Specific medical specialties acceptable for initiating/managing treatment.
- E.g., dermatologist, rheumatologist, etc.
- Output "NA" if not mentioned.""",

    "Initial Authorization Duration(in-months)": """Time period for which coverage is initially granted upon PA approval.
- Expressed in months (e.g., 6 or 12 months).
- If PA for PsO is 'Yes', output duration or "Unspecified".
- Output "NA" if not applicable.""",

    "Reauthorization Duration(in-months)": """Length of time for reauthorization once initial period ends.
- Often 6 or 12 months.
- If Reauthorization Required is 'Yes', output duration or "Unspecified".
- Output "NA" if not applicable.""",

    "Reauthorization Required": """Whether reassessment/renewed approval needed after initial coverage expires.
- If either reauthorization duration or reauthorization requirements documented is non-NA, output "Yes".
- Output "No" if explicitly stated not required.
- Output "NA" if not mentioned.""",

    "Reauthorization Requirements Documented in Policy": """Explicit criteria for reauthorization such as continued clinical benefit, lack of disease progression, or specific lab values.
- Extract full text of reauthorization criteria.
- Output "NA" if not documented.""",
}

# Access Score Rules
ACCESS_SCORE_RULES = """
Access Score Scale (0-100):
- 0: No access (drug not covered, or coverage explicitly denied)
- 25: Restricted access (many steps, strict age limits, short auth duration vs FDA)
- 50: Parity with FDA label (requirements align with standard FDA label)
- 75: Preferred access (fewer restrictions than typical, longer auth periods)
- 100: Best possible access (no restrictions, unlimited authorization)

Factors to consider:
1. Number of step therapy requirements (more steps = lower score)
2. Age restrictions (stricter = lower score)
3. Phototherapy required (Yes = lower score)
4. TB test (adds burden but standard = neutral)
5. Quantity limits (restrictive = lower score)
6. Specialist requirements (more restrictive = lower score)
7. Initial auth duration (shorter = lower score)
8. Reauthorization requirements (more complex = lower score)
"""


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup comprehensive logging."""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("PAExtractor")
    logger.setLevel(logging.DEBUG)
    
    # File handler - detailed
    fh = logging.FileHandler(os.path.join(log_dir, "pipeline.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Console handler - info only
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ChunkMetadata:
    """Metadata for each chunk."""
    source_file: str
    page_number: int
    chunk_index: int
    section_header: str = ""
    brands_mentioned: List[str] = field(default_factory=list)
    chunk_type: str = "text"  # text, table, header
    hierarchy_level: int = 0
    parent_section: str = ""
    

@dataclass
class RetrievalLog:
    """Log entry for retrieval operations."""
    filename: str
    brand: str
    parameter: str
    query: str
    chunks_retrieved: List[Dict]
    llm_response: str
    judge_score: Optional[float] = None
    judge_feedback: Optional[str] = None
    timestamp: str = ""
    

# ============================================================================
# PDF PARSING MODULE
# ============================================================================

class PDFParser:
    """Parse PDF documents using PyMuPDF with structure awareness."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def parse_pdf(self, pdf_path: str) -> List[Document]:
        """Parse a PDF and return structured documents with metadata."""
        self.logger.info(f"Parsing PDF: {pdf_path}")
        documents = []
        
        try:
            doc = fitz.open(pdf_path)
            filename = os.path.basename(pdf_path)
            
            full_text_pages = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract text with layout preservation
                text = page.get_text("text")
                
                # Also try to extract tables
                tables = self._extract_tables(page)
                
                if text.strip():
                    full_text_pages.append({
                        "page_num": page_num + 1,
                        "text": text,
                        "tables": tables
                    })
            
            doc.close()
            
            # Build documents with hierarchical structure
            documents = self._build_hierarchical_documents(full_text_pages, filename)
            
            # Extract structured fields from full text (authorization durations, prescriber, quantity limits, etc.)
            full_text = '\n'.join([p["text"] for p in full_text_pages])
            structured_docs = self._extract_structured_fields(full_text, filename)
            documents.extend(structured_docs)
            self.logger.info(f"Extracted {len(structured_docs)} structured field documents from {filename}")
            
            self.logger.info(f"Parsed {len(documents)} document sections from {filename}")
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF {pdf_path}: {e}")
            self.logger.error(traceback.format_exc())
        
        return documents
    
    def _extract_tables(self, page) -> List[str]:
        """Extract table content from a page."""
        tables = []
        try:
            # Use PyMuPDF's table extraction
            tabs = page.find_tables()
            if tabs and tabs.tables:
                for table in tabs.tables:
                    df = table.to_pandas()
                    if not df.empty:
                        tables.append(df.to_string(index=False))
        except Exception:
            pass
        return tables
    
    def _extract_structured_fields(self, full_text: str, filename: str) -> List[Document]:
        """Extract structured key-value fields from the full document text.
        
        Detects patterns like:
        - "Length of Approval: X Month(s)"
        - "Authorization of X months may be granted"
        - "Prescriber Specialties" sections
        - "Quantity Limits" sections/tables
        - "Coverage Duration" entries
        - "Prior Authorization (Initial/Reauthorization)" sections
        """
        structured_docs = []
        lines = full_text.split('\n')
        
        # Pattern 1: "Length of Approval: X Month(s)" with surrounding context
        for i, line in enumerate(lines):
            match = re.search(r'Length of Approval:\s*(\d+)\s*Month', line, re.IGNORECASE)
            if match:
                months = match.group(1)
                # Look backwards for Initial/Reauthorization context
                context_start = max(0, i - 5)
                context_end = min(len(lines), i + 10)
                context_lines = lines[context_start:context_end]
                context_text = '\n'.join(context_lines)
                
                # Determine if initial or reauthorization
                auth_type = "unknown"
                for cl in lines[max(0, i-5):i+1]:
                    if 'reauthorization' in cl.lower() or 'renewal' in cl.lower() or 'continuation' in cl.lower():
                        auth_type = "reauthorization"
                        break
                    elif 'initial' in cl.lower():
                        auth_type = "initial"
                        break
                
                # Detect indication
                indication = "general"
                for cl in lines[i:min(len(lines), i+5)]:
                    if 'plaque psoriasis' in cl.lower() or 'pso' in cl.lower():
                        indication = "PsO"
                        break
                    elif 'psoriatic arthritis' in cl.lower() or 'psa' in cl.lower():
                        indication = "PsA"
                        break
                
                doc = Document(
                    page_content=f"[STRUCTURED FIELD: Authorization Duration]\nType: {auth_type}\nDuration: {months} months\nIndication: {indication}\nContext: {context_text}",
                    metadata={
                        "source": filename,
                        "type": "structured_field",
                        "field_type": "authorization_duration",
                        "auth_type": auth_type,
                        "duration_months": months,
                        "indication": indication,
                        "is_common": True,
                    }
                )
                structured_docs.append(doc)
        
        # Pattern 2: "Authorization of X months may be granted" 
        auth_pattern = re.compile(r'[Aa]uthorization of (\d+) months? (?:may be|is) granted', re.IGNORECASE)
        for i, line in enumerate(lines):
            match = auth_pattern.search(line)
            if match:
                months = match.group(1)
                context_start = max(0, i - 3)
                context_end = min(len(lines), i + 8)
                context_text = '\n'.join(lines[context_start:context_end])
                
                # Determine auth type from surrounding lines
                auth_type = "initial"
                for cl in lines[max(0, i-5):i+1]:
                    if 'continuation' in cl.lower() or 'renewal' in cl.lower() or 'reauthorization' in cl.lower():
                        auth_type = "reauthorization"
                        break
                
                indication = "general"
                for cl in lines[max(0, i-3):min(len(lines), i+5)]:
                    if 'plaque psoriasis' in cl.lower() or 'pso' in cl.lower():
                        indication = "PsO"
                        break
                    elif 'psoriatic arthritis' in cl.lower() or 'psa' in cl.lower():
                        indication = "PsA"
                        break
                
                doc = Document(
                    page_content=f"[STRUCTURED FIELD: Authorization Duration]\nType: {auth_type}\nDuration: {months} months\nIndication: {indication}\nContext: {context_text}",
                    metadata={
                        "source": filename,
                        "type": "structured_field",
                        "field_type": "authorization_duration",
                        "auth_type": auth_type,
                        "duration_months": months,
                        "indication": indication,
                        "is_common": True,
                    }
                )
                structured_docs.append(doc)
        
        # Pattern 3: Prescriber Specialties / Restrictions sections
        prescriber_patterns = [
            r'(?:prescribed by|in consultation with)\s+(?:a\s+|one of the following[:\s]*)(.*?)(?:\.|$)',
            r'Prescriber Specialties?[:\s]*(.*?)(?:\n\n|\Z)',
            r'Prescriber Restrictions?[:\s]*(.*?)(?:\n\n|\Z)',
        ]
        for i, line in enumerate(lines):
            if re.search(r'prescrib(?:er|ed by|ing)', line, re.IGNORECASE) or \
               re.search(r'in consultation with', line, re.IGNORECASE):
                context_start = max(0, i - 2)
                context_end = min(len(lines), i + 12)
                context_text = '\n'.join(lines[context_start:context_end])
                
                # Extract specialties mentioned
                specialties = []
                for cl in lines[i:min(len(lines), i+10)]:
                    if 'dermatolog' in cl.lower():
                        specialties.append('Dermatologist')
                    if 'rheumatolog' in cl.lower():
                        specialties.append('Rheumatologist')
                    if 'gastroenterolog' in cl.lower():
                        specialties.append('Gastroenterologist')
                    if 'oncolog' in cl.lower():
                        specialties.append('Oncologist')
                
                if specialties:
                    # Detect indication context
                    indication = "general"
                    for cl in lines[max(0, i-5):min(len(lines), i+5)]:
                        if 'plaque psoriasis' in cl.lower() or 'pso' in cl.lower():
                            indication = "PsO"
                            break
                        elif 'psoriatic arthritis' in cl.lower() or 'psa' in cl.lower():
                            indication = "PsA"
                            break
                    
                    doc = Document(
                        page_content=f"[STRUCTURED FIELD: Prescriber Restrictions]\nSpecialties: {', '.join(list(set(specialties)))}\nIndication: {indication}\nContext: {context_text}",
                        metadata={
                            "source": filename,
                            "type": "structured_field",
                            "field_type": "prescriber_restrictions",
                            "specialties": list(set(specialties)),
                            "indication": indication,
                            "is_common": True,
                        }
                    )
                    structured_docs.append(doc)
        
        # Pattern 4: Quantity Limits sections
        in_quantity_section = False
        quantity_lines = []
        quantity_start = -1
        for i, line in enumerate(lines):
            if re.search(r'Quantity\s*(?:Level\s*)?Limit', line, re.IGNORECASE) and not re.search(r'See\s+', line, re.IGNORECASE):
                in_quantity_section = True
                quantity_start = i
                quantity_lines = [line]
                continue
            if in_quantity_section:
                # End section on double blank or new major section header
                if (line.strip() == '' and i > quantity_start + 1 and 
                    (i+1 < len(lines) and lines[i+1].strip() == '')) or \
                   (re.match(r'^[A-Z][A-Z\s]{5,}$', line.strip()) and 
                    not re.search(r'quantity|limit|medication', line, re.IGNORECASE)):
                    # Save what we have
                    if len(quantity_lines) > 1:
                        ql_text = '\n'.join(quantity_lines)
                        doc = Document(
                            page_content=f"[STRUCTURED FIELD: Quantity Limits]\n{ql_text}",
                            metadata={
                                "source": filename,
                                "type": "structured_field",
                                "field_type": "quantity_limits",
                                "is_common": True,
                            }
                        )
                        structured_docs.append(doc)
                    in_quantity_section = False
                    quantity_lines = []
                else:
                    quantity_lines.append(line)
                    # Safety: don't capture more than 40 lines
                    if len(quantity_lines) > 40:
                        ql_text = '\n'.join(quantity_lines)
                        doc = Document(
                            page_content=f"[STRUCTURED FIELD: Quantity Limits]\n{ql_text}",
                            metadata={
                                "source": filename,
                                "type": "structured_field",
                                "field_type": "quantity_limits",
                                "is_common": True,
                            }
                        )
                        structured_docs.append(doc)
                        in_quantity_section = False
                        quantity_lines = []
        
        # Capture remaining quantity section if file ended
        if in_quantity_section and len(quantity_lines) > 1:
            ql_text = '\n'.join(quantity_lines)
            doc = Document(
                page_content=f"[STRUCTURED FIELD: Quantity Limits]\n{ql_text}",
                metadata={
                    "source": filename,
                    "type": "structured_field",
                    "field_type": "quantity_limits",
                    "is_common": True,
                }
            )
            structured_docs.append(doc)
        
        # Pattern 5: "Coverage Duration" key-value pairs (form-style PDFs)
        for i, line in enumerate(lines):
            if re.search(r'^Coverage Duration', line, re.IGNORECASE):
                # Look at next few lines for the value
                context_end = min(len(lines), i + 5)
                value_lines = [l.strip() for l in lines[i+1:context_end] if l.strip() and l.strip() != '-']
                if value_lines:
                    value = value_lines[0]
                    # Check if it's a meaningful value (not just another field label)
                    if not re.match(r'^(?:Other Criteria|Prior Authorization|Exclusion|Required)', value, re.IGNORECASE):
                        context_start = max(0, i - 5)
                        context_text = '\n'.join(lines[context_start:context_end])
                        doc = Document(
                            page_content=f"[STRUCTURED FIELD: Coverage Duration]\nValue: {value}\nContext: {context_text}",
                            metadata={
                                "source": filename,
                                "type": "structured_field",
                                "field_type": "coverage_duration",
                                "is_common": True,
                            }
                        )
                        structured_docs.append(doc)
        
        # Pattern 6: TB/Tuberculosis test requirements
        for i, line in enumerate(lines):
            if re.search(r'tuberculosis|(?<!\w)TB(?:\s+test|\s+screen)', line, re.IGNORECASE):
                context_start = max(0, i - 2)
                context_end = min(len(lines), i + 6)
                context_text = '\n'.join(lines[context_start:context_end])
                
                doc = Document(
                    page_content=f"[STRUCTURED FIELD: TB Test Requirement]\nContext: {context_text}",
                    metadata={
                        "source": filename,
                        "type": "structured_field",
                        "field_type": "tb_test",
                        "is_common": True,
                    }
                )
                structured_docs.append(doc)
        
        # Pattern 7: Reauthorization/Renewal/Continuation criteria sections
        for i, line in enumerate(lines):
            if re.search(r'(?:Reauthorization|Renewal|Continuation)\s*(?:Criteria|of Therapy|Requirements?)', line, re.IGNORECASE):
                context_start = i
                context_end = min(len(lines), i + 20)
                # Capture until next major section
                section_lines = [line]
                for j in range(i+1, context_end):
                    if re.match(r'^(?:[A-Z][A-Z\s]{5,}|(?:Prior Authorization|Criteria for)\s)', lines[j].strip()):
                        break
                    section_lines.append(lines[j])
                
                if len(section_lines) > 2:
                    # Detect indication
                    indication = "general"
                    section_text = '\n'.join(section_lines)
                    if 'plaque psoriasis' in section_text.lower() or 'pso' in section_text.lower():
                        indication = "PsO"
                    
                    doc = Document(
                        page_content=f"[STRUCTURED FIELD: Reauthorization Criteria]\nIndication: {indication}\n{section_text}",
                        metadata={
                            "source": filename,
                            "type": "structured_field",
                            "field_type": "reauthorization_criteria",
                            "indication": indication,
                            "is_common": True,
                        }
                    )
                    structured_docs.append(doc)
        
        # Deduplicate structured docs by content similarity
        seen_content = set()
        unique_docs = []
        for doc in structured_docs:
            # Use first 150 chars as dedup key
            key = doc.page_content[:150]
            if key not in seen_content:
                seen_content.add(key)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _build_hierarchical_documents(self, pages: List[Dict], filename: str) -> List[Document]:
        """Build hierarchically structured documents."""
        documents = []
        current_section = ""
        current_subsection = ""
        
        # Patterns for section detection
        section_patterns = [
            r'^[A-Z][A-Z\s&/]+$',  # ALL CAPS headers
            r'^\d+\.\s+[A-Z]',  # Numbered sections
            r'^(?:SECTION|PART|CHAPTER)\s+\d+',
            r'^(?:CRITERIA|REQUIREMENTS|AUTHORIZATION|QUANTITY)',
            r'^(?:PRIOR AUTHORIZATION|STEP THERAPY|REAUTHORIZATION)',
        ]
        
        for page_data in pages:
            page_num = page_data["page_num"]
            text = page_data["text"]
            tables = page_data["tables"]
            
            # Split text into lines and detect sections
            lines = text.split('\n')
            current_block = []
            
            for line in lines:
                stripped = line.strip()
                
                # Check if this line is a section header
                is_header = False
                for pattern in section_patterns:
                    if re.match(pattern, stripped) and len(stripped) > 3 and len(stripped) < 100:
                        is_header = True
                        break
                
                if is_header and current_block:
                    # Save previous block
                    block_text = '\n'.join(current_block)
                    if block_text.strip() and len(block_text.strip()) > 50:
                        doc = Document(
                            page_content=block_text,
                            metadata={
                                "source": filename,
                                "page": page_num,
                                "section": current_section,
                                "subsection": current_subsection,
                                "type": "text"
                            }
                        )
                        documents.append(doc)
                    current_block = []
                    current_section = stripped
                
                current_block.append(line)
            
            # Save remaining block
            if current_block:
                block_text = '\n'.join(current_block)
                if block_text.strip() and len(block_text.strip()) > 50:
                    doc = Document(
                        page_content=block_text,
                        metadata={
                            "source": filename,
                            "page": page_num,
                            "section": current_section,
                            "subsection": current_subsection,
                            "type": "text"
                        }
                    )
                    documents.append(doc)
            
            # Add tables as separate documents
            for i, table_text in enumerate(tables):
                if table_text.strip():
                    doc = Document(
                        page_content=f"[TABLE from page {page_num}]\n{table_text}",
                        metadata={
                            "source": filename,
                            "page": page_num,
                            "section": current_section,
                            "type": "table"
                        }
                    )
                    documents.append(doc)
        
        return documents


# ============================================================================
# CHUNKING MODULE
# ============================================================================

class HierarchicalChunker:
    """Hierarchical + context-aware + semantic chunking."""
    
    # Generic drug names mapped to brand names for detection
    GENERIC_NAMES = [
        "guselkumab", "ustekinumab", "adalimumab", "secukinumab",
        "etanercept", "apremilast", "certolizumab", "brodalumab",
        "bimekizumab", "tildrakizumab", "risankizumab", "ixekizumab",
        "infliximab", "deucravacitinib", "acitretin"
    ]
    
    def __init__(self, logger: logging.Logger, 
                 chunk_size: int = MAX_CHUNK_SIZE_CHARS,
                 chunk_overlap: int = CHUNK_OVERLAP_CHARS,
                 brands: List[str] = None):
        self.logger = logger
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # LangChain recursive splitter for fallback
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        # Brands to detect in chunks - built dynamically from input data
        # Include both brand names and their known generic equivalents
        if brands:
            self.known_brands = [b.upper() for b in brands] + self.GENERIC_NAMES
        else:
            self.known_brands = self.GENERIC_NAMES
        
        # Build a reverse map: generic -> brand for tagging
        self.generic_to_brand = {}
        for brand_name, generic_name in BRAND_GENERIC_MAP.items():
            self.generic_to_brand[generic_name.upper()] = brand_name.upper()
        
        self.known_brands = list(set(self.known_brands))
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Apply hierarchical chunking with context awareness."""
        self.logger.info(f"Chunking {len(documents)} document sections...")
        
        all_chunks = []
        
        for doc in documents:
            text = doc.page_content
            metadata = doc.metadata.copy()
            
            # Never split structured field documents - they are atomic retrieval units
            if metadata.get("type") == "structured_field":
                brands = self._detect_brands(text)
                metadata["brands_mentioned"] = brands
                metadata["chunk_index"] = 0
                all_chunks.append(Document(page_content=text, metadata=metadata))
                continue
            
            # Strategy 1: If document is small enough, keep as-is
            if len(text) <= self.chunk_size:
                # Detect brands mentioned
                brands = self._detect_brands(text)
                metadata["brands_mentioned"] = brands
                metadata["chunk_index"] = 0
                all_chunks.append(Document(page_content=text, metadata=metadata))
                continue
            
            # Strategy 2: Try to split by logical sections within the document
            section_chunks = self._split_by_sections(text, metadata)
            
            if section_chunks:
                all_chunks.extend(section_chunks)
            else:
                # Fallback: Use recursive character splitter
                chunks = self.text_splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    chunk_meta = metadata.copy()
                    chunk_meta["chunk_index"] = i
                    chunk_meta["brands_mentioned"] = self._detect_brands(chunk)
                    all_chunks.append(Document(page_content=chunk, metadata=chunk_meta))
        
        # Add context headers to chunks
        all_chunks = self._add_context_headers(all_chunks)
        
        self.logger.info(f"Generated {len(all_chunks)} chunks total")
        return all_chunks
    
    def _split_by_sections(self, text: str, metadata: Dict) -> List[Document]:
        """Split text by detected section boundaries."""
        chunks = []
        
        # Split on double newlines or section patterns
        sections = re.split(r'\n{2,}', text)
        
        current_chunk = ""
        chunk_idx = 0
        
        for section in sections:
            if len(current_chunk) + len(section) <= self.chunk_size:
                current_chunk += "\n\n" + section if current_chunk else section
            else:
                if current_chunk.strip():
                    chunk_meta = metadata.copy()
                    chunk_meta["chunk_index"] = chunk_idx
                    chunk_meta["brands_mentioned"] = self._detect_brands(current_chunk)
                    chunks.append(Document(page_content=current_chunk, metadata=chunk_meta))
                    chunk_idx += 1
                
                # Start new chunk with overlap
                if len(section) > self.chunk_size:
                    # Section itself is too large, split further
                    sub_chunks = self.text_splitter.split_text(section)
                    for sub in sub_chunks:
                        chunk_meta = metadata.copy()
                        chunk_meta["chunk_index"] = chunk_idx
                        chunk_meta["brands_mentioned"] = self._detect_brands(sub)
                        chunks.append(Document(page_content=sub, metadata=chunk_meta))
                        chunk_idx += 1
                    current_chunk = ""
                else:
                    current_chunk = section
        
        # Don't forget last chunk
        if current_chunk.strip():
            chunk_meta = metadata.copy()
            chunk_meta["chunk_index"] = chunk_idx
            chunk_meta["brands_mentioned"] = self._detect_brands(current_chunk)
            chunks.append(Document(page_content=current_chunk, metadata=chunk_meta))
        
        return chunks
    
    def _detect_brands(self, text: str) -> List[str]:
        """Detect brand names mentioned in text. Also resolves generic names to brands."""
        found = []
        text_upper = text.upper()
        text_lower = text.lower()
        
        for brand in self.known_brands:
            if brand.upper() in text_upper or brand.lower() in text_lower:
                found.append(brand.upper())
                # If this is a generic name, also tag with its brand name
                if brand.upper() in self.generic_to_brand:
                    found.append(self.generic_to_brand[brand.upper()])
        
        return list(set(found))
    
    def _add_context_headers(self, chunks: List[Document]) -> List[Document]:
        """Add contextual headers to chunks for better retrieval."""
        enhanced_chunks = []
        
        for chunk in chunks:
            meta = chunk.metadata
            header_parts = []
            
            if meta.get("source"):
                header_parts.append(f"Source: {meta['source']}")
            if meta.get("section"):
                header_parts.append(f"Section: {meta['section']}")
            if meta.get("page"):
                header_parts.append(f"Page: {meta['page']}")
            if meta.get("brands_mentioned"):
                header_parts.append(f"Brands: {', '.join(meta['brands_mentioned'])}")
            
            if header_parts:
                context_header = " | ".join(header_parts)
                enhanced_text = f"[{context_header}]\n{chunk.page_content}"
            else:
                enhanced_text = chunk.page_content
            
            enhanced_chunks.append(Document(
                page_content=enhanced_text,
                metadata=meta
            ))
        
        return enhanced_chunks


# ============================================================================
# VECTOR STORE MODULE
# ============================================================================

class FAISSVectorStore:
    """FAISS-based vector store with metadata."""
    
    def __init__(self, logger: logging.Logger, embedding_model_name: str = EMBEDDING_MODEL):
        self.logger = logger
        self.logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        self.embeddings = None
    
    def build_index(self, documents: List[Document]):
        """Build FAISS index from documents."""
        self.logger.info(f"Building FAISS index with {len(documents)} documents...")
        self.documents = documents
        
        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        self.embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True, 
            batch_size=64,
            normalize_embeddings=True
        )
        
        # Build FAISS index (using Inner Product for cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(np.array(self.embeddings).astype('float32'))
        
        self.logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL, 
               filter_source: Optional[str] = None,
               filter_brand: Optional[str] = None) -> List[Tuple[Document, float]]:
        """Search for similar documents with optional filtering."""
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).astype('float32')
        
        # Search more than needed to account for filtering
        search_k = min(top_k * 5, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            
            doc = self.documents[idx]
            
            # Apply filters
            if filter_source and doc.metadata.get("source") != filter_source:
                continue
            
            if filter_brand:
                # Always include structured_field documents marked as common
                doc_type = doc.metadata.get("type", "")
                is_common = doc.metadata.get("is_common", False)
                if doc_type == "structured_field" and is_common:
                    # Always include common structured fields - they apply to all brands
                    pass
                else:
                    brands = doc.metadata.get("brands_mentioned", [])
                    # Include chunks that:
                    # 1. Mention the brand explicitly
                    # 2. Don't mention ANY brand (general/universal criteria)
                    # 3. Are from relevant policy sections (Coverage Duration, Prescriber, Quantity, etc.)
                    # 4. Are tables (often contain key structured data)
                    if brands and filter_brand.upper() not in [b.upper() for b in brands]:
                        # Check if chunk mentions the brand in text (including generic name)
                        brand_upper = filter_brand.upper()
                        generic_name = BRAND_GENERIC_MAP.get(filter_brand.upper(), "").upper()
                        brand_in_text = brand_upper in doc.page_content.upper()
                        if not brand_in_text and generic_name:
                            brand_in_text = generic_name in doc.page_content.upper()
                        if not brand_in_text:
                            # Still include if it's a general policy section or table
                            section = doc.metadata.get("section", "").upper()
                            content_upper = doc.page_content.upper()
                            is_relevant_section = any(kw in section or kw in content_upper for kw in [
                                "CRITERIA", "REQUIREMENT", "AUTHORIZATION", "QUANTITY", "PRIOR", 
                                "STEP", "GENERAL", "COVERAGE DURATION", "PRESCRIBER", "RENEWAL",
                                "DOCUMENTATION", "TB", "TUBERCULOSIS", "SPECIALIST", "LIMIT",
                                "INDICATION", "ALL INDICATION", "INITIAL", "CONTINUATION"
                            ])
                            if not is_relevant_section and doc_type != "table":
                                continue
            
            results.append((doc, float(score)))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save_index(self, path: str):
        """Save FAISS index and documents to disk."""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        # Save documents metadata
        docs_data = []
        for doc in self.documents:
            docs_data.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        
        with open(os.path.join(path, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        
        # Save embeddings
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        
        self.logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str) -> bool:
        """Load FAISS index and documents from disk."""
        try:
            index_path = os.path.join(path, "index.faiss")
            docs_path = os.path.join(path, "documents.json")
            embeddings_path = os.path.join(path, "embeddings.npy")
            
            if not all(os.path.exists(p) for p in [index_path, docs_path, embeddings_path]):
                return False
            
            self.index = faiss.read_index(index_path)
            
            with open(docs_path, "r", encoding="utf-8") as f:
                docs_data = json.load(f)
            
            self.documents = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in docs_data
            ]
            
            self.embeddings = np.load(embeddings_path)
            
            self.logger.info(f"Index loaded from {path} ({self.index.ntotal} vectors)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
            return False


# ============================================================================
# LLM MODULE WITH RETRY & FALLBACK
# ============================================================================

class LLMClient:
    """LLM client with retry mechanisms and fallback."""
    
    def __init__(self, logger: logging.Logger, api_key: Optional[str] = None):
        self.logger = logger
        # Use provided api_key or fall back to the runtime GROQ_API_KEY
        self.api_key = api_key if api_key is not None else GROQ_API_KEY
        self.primary_model = PRIMARY_MODEL
        self.fallback_model = FALLBACK_MODEL
        self.call_count = 0
        self.total_tokens = 0
        self.primary_exhausted = False  # Set True when daily token limit hit on primary
        self.all_models_exhausted = False  # Set True when both models are exhausted
        self._rate_lock = threading.Lock()  # Serialize API calls to prevent rate limit cascades
        self._last_call_time = 0.0  # Track last API call timestamp
        self._rate_lock = threading.Lock()  # Serialize API calls to prevent rate limit cascades
        
        # Initialize LLM instances
        self.primary_llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.primary_model,
            temperature=0.1,
            max_tokens=4096,
        )
        
        self.fallback_llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.fallback_model,
            temperature=0.1,
            max_tokens=4096,
        )
    
    def invoke(self, prompt: str, system_prompt: str = "", use_fallback: bool = False) -> str:
        """Invoke LLM with retry and fallback logic."""
        if self.all_models_exhausted:
            raise RuntimeError("ALL_MODELS_EXHAUSTED")
        # Rate-limit spacing: ensure minimum 2s between API calls across all threads
        with self._rate_lock:
            elapsed = time.time() - self._last_call_time
            if elapsed < 2.0:
                time.sleep(2.0 - elapsed)
            self._last_call_time = time.time()
        return self._invoke_with_retry(prompt, system_prompt, use_fallback)
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=2, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
        retry=retry_if_exception_type((Exception,)) & ~retry_if_exception_type((RuntimeError,)),
        reraise=True,
        before_sleep=lambda retry_state: print(f"  [Retry] Attempt {retry_state.attempt_number}/{MAX_RETRIES}, waiting...")
    )
    def _invoke_with_retry(self, prompt: str, system_prompt: str = "", use_fallback: bool = False) -> str:
        """Internal invoke with retry decorator."""
        try:
            # If primary is exhausted, always use fallback
            if self.primary_exhausted and not use_fallback:
                use_fallback = True
            
            llm = self.fallback_llm if use_fallback else self.primary_llm
            model_name = self.fallback_model if use_fallback else self.primary_model
            
            messages = []
            if system_prompt:
                messages.append(("system", system_prompt))
            messages.append(("human", prompt))
            
            response = llm.invoke(messages)
            self.call_count += 1
            
            if hasattr(response, 'response_metadata'):
                token_usage = response.response_metadata.get('token_usage', {})
                self.total_tokens += token_usage.get('total_tokens', 0)
            
            return response.content
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Daily token quota exhausted - permanently pivot to fallback
            if any(kw in error_str for kw in ["resource_exhausted", "quota", "daily", "tokens per day", "limit has been reached"]):
                if not use_fallback:
                    self.primary_exhausted = True
                    self.logger.warning(f"*** Daily token quota exhausted on {model_name}. Permanently switching to {self.fallback_model} ***")
                    return self._invoke_with_retry(prompt, system_prompt, use_fallback=True)
                else:
                    # Both models exhausted - mark as fully exhausted and return empty
                    self.all_models_exhausted = True
                    self.logger.error("*** Both primary and fallback models exhausted! Remaining parameters will default to NA. ***")
                    raise RuntimeError("ALL_MODELS_EXHAUSTED")
            
            # Rate limit handling (per-minute)
            if "rate_limit" in error_str or "429" in error_str or "too many" in error_str or "tokens per minute" in error_str:
                self.logger.warning(f"Rate limit hit on {model_name}. Waiting {RATE_LIMIT_PAUSE}s...")
                time.sleep(RATE_LIMIT_PAUSE)
                
                # Try fallback model first; if already on fallback, retry same model (rate limit is temporary)
                if not use_fallback:
                    self.logger.info("Switching to fallback model...")
                    return self._invoke_with_retry(prompt, system_prompt, use_fallback=True)
                else:
                    # Already on fallback - retry same model after waiting (rate limit is per-minute, resets)
                    self.logger.info("Retrying fallback model after rate limit cooldown...")
                    return self._invoke_with_retry(prompt, system_prompt, use_fallback=True)
            
            # Token limit exceeded (single request too large)
            if "token" in error_str and ("limit" in error_str or "exceed" in error_str):
                self.logger.warning("Token limit exceeded, truncating prompt...")
                # Truncate prompt to ~60% of original
                truncated_prompt = prompt[:int(len(prompt) * 0.6)]
                truncated_prompt += "\n\n[Note: Context was truncated due to token limits. Extract what you can from available text.]"
                
                llm = self.fallback_llm if use_fallback else self.primary_llm
                messages = []
                if system_prompt:
                    messages.append(("system", system_prompt))
                messages.append(("human", truncated_prompt))
                response = llm.invoke(messages)
                self.call_count += 1
                return response.content
            
            # Model access/auth errors - fallback to other model
            if any(kw in error_str for kw in ["model not found", "not found", "unauthorized", "permission", "access denied", "invalid model", "does not exist", "not available", "decommissioned"]):
                if not use_fallback:
                    self.logger.warning(f"Model access error on {model_name}: {str(e)[:100]}. Switching to {self.fallback_model}...")
                    self.primary_exhausted = True
                    return self._invoke_with_retry(prompt, system_prompt, use_fallback=True)
                else:
                    self.logger.error(f"Both models inaccessible. Error: {str(e)[:200]}")
                    raise
            
            raise


# ============================================================================
# LLM JUDGE MODULE
# ============================================================================

class LLMJudge:
    """LLM Judge for validation of retrieval and extraction quality."""
    
    def __init__(self, llm_client: LLMClient, logger: logging.Logger):
        self.llm = llm_client
        self.logger = logger
        self.judgments = []
    
    def judge_retrieval(self, query: str, chunks: List[str], parameter: str, brand: str) -> Dict:
        """Judge the quality of retrieved chunks for a parameter."""
        if not ENABLE_LLM_JUDGE:
            return {"score": 3, "feedback": "Judge disabled", "needs_rerank": False}
        
        # Use only first 3 chunks and limit text to save tokens
        chunks_text = "\n---\n".join([c[:500] for c in chunks[:3]])
        
        prompt = f"""Rate retrieval quality (1-5) for extracting "{parameter}" for {brand}.
Chunks:
{chunks_text}

JSON: {{"score": <1-5>, "feedback": "<10 words>", "needs_rerank": <true/false>}}"""
        
        try:
            response = self.llm.invoke(prompt, use_fallback=True)
            # Parse JSON from response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"score": 3, "feedback": "Could not parse judge response", "needs_rerank": False}
        except Exception as e:
            self.logger.warning(f"Judge error: {e}")
            result = {"score": 3, "feedback": f"Judge error: {str(e)}", "needs_rerank": False}
        
        self.judgments.append({
            "parameter": parameter,
            "brand": brand,
            "result": result
        })
        
        return result
    
    def judge_extraction(self, parameter: str, extracted_value: str, 
                         context: str, brand: str) -> Dict:
        """Judge the quality of an extracted parameter value."""
        if not ENABLE_LLM_JUDGE:
            return {"groundedness": 3, "faithfulness": 3, "completeness": 3, "overall": 3, "issues": "Judge disabled"}
        
        # Concise prompt to save tokens
        prompt = f"""Rate extraction (1-5 each): Parameter={parameter}, Brand={brand}, Value="{extracted_value[:200]}"
Context (excerpt): {context[:1500]}

JSON: {{"groundedness": <1-5>, "faithfulness": <1-5>, "completeness": <1-5>, "overall": <1-5>, "issues": "<brief>"}}"""
        
        try:
            response = self.llm.invoke(prompt, use_fallback=True)
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"groundedness": 3, "faithfulness": 3, "completeness": 3, "overall": 3, "issues": "Parse error"}
        except Exception as e:
            self.logger.warning(f"Judge extraction error: {e}")
            result = {"groundedness": 3, "faithfulness": 3, "completeness": 3, "overall": 3, "issues": str(e)}
        
        return result
    
    def get_summary(self) -> Dict:
        """Get summary of all judgments."""
        if not self.judgments:
            return {"total": 0, "avg_score": 0}
        
        scores = [j["result"].get("score", j["result"].get("overall", 3)) for j in self.judgments]
        return {
            "total": len(self.judgments),
            "avg_score": sum(scores) / len(scores),
            "judgments": self.judgments
        }


# ============================================================================
# RAG EXTRACTION ENGINE
# ============================================================================

class RAGExtractor:
    """Main RAG extraction engine using LangGraph for orchestration."""
    
    def __init__(self, llm_client: LLMClient, vector_store: FAISSVectorStore, 
                 judge: LLMJudge, logger: logging.Logger):
        self.llm = llm_client
        self.vector_store = vector_store
        self.judge = judge
        self.logger = logger
        self.retrieval_logs = []
    
    def extract_parameters(self, filename: str, brand: str) -> Dict[str, str]:
        """Extract all parameters for a given filename-brand combination."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Extracting parameters for: {filename} | Brand: {brand}")
        self.logger.info(f"{'='*60}")
        
        results = {}
        
        # Group parameters for efficient extraction
        param_groups = self._group_parameters()
        
        for group_name, params in param_groups.items():
            self.logger.info(f"\n  Processing group: {group_name}")
            group_results = self._extract_parameter_group(filename, brand, params)
            results.update(group_results)
        
        # Calculate Access Score
        results["Access Score"] = self._calculate_access_score(results, filename, brand)
        
        return results
    
    def _group_parameters(self) -> Dict[str, List[str]]:
        """Group parameters for efficient retrieval - each param gets its own retrieval."""
        return {
            "step_therapy": [
                "Step Therapy Requirements Documented in Policy",
                "Number of Steps through Brands",
                "Number of Steps through Generic",
                "Step through-Phototherapy",
            ],
            "clinical_and_auth": [
                "Age",
                "TB Test required",
                "Quantity Limits",
                "Specialist Types",
                "Initial Authorization Duration(in-months)",
                "Reauthorization Duration(in-months)",
                "Reauthorization Required",
                "Reauthorization Requirements Documented in Policy",
            ],
        }
    
    def _extract_parameter_group(self, filename: str, brand: str, 
                                  parameters: List[str]) -> Dict[str, str]:
        """Extract a group of related parameters with per-parameter retrieval."""
        results = {}
        
        # Map parameters to structured field types for direct lookup
        param_to_field_type = {
            "Specialist Types": "prescriber_restrictions",
            "Initial Authorization Duration(in-months)": "authorization_duration",
            "Reauthorization Duration(in-months)": "authorization_duration",
            "Reauthorization Required": "reauthorization_criteria",
            "Reauthorization Requirements Documented in Policy": "reauthorization_criteria",
            "Quantity Limits": "quantity_limits",
            "TB Test required": "tb_test",
        }
        
        # Pre-collect structured field documents for this file
        structured_field_docs = {}
        for doc_item in self.vector_store.documents:
            if doc_item.metadata.get("source") == filename and doc_item.metadata.get("type") == "structured_field":
                ft = doc_item.metadata.get("field_type", "")
                if ft not in structured_field_docs:
                    structured_field_docs[ft] = []
                structured_field_docs[ft].append(doc_item)
        
        # Do a broad retrieval for context
        broad_query = self._build_retrieval_query(parameters, brand)
        broad_retrieved = self.vector_store.search(
            query=broad_query,
            top_k=TOP_K_RETRIEVAL,
            filter_source=filename,
            filter_brand=brand
        )
        
        broad_context = ""
        if broad_retrieved:
            broad_chunks = [doc.page_content for doc, score in broad_retrieved]
            broad_context = "\n\n---\n\n".join(broad_chunks[:RERANK_TOP_K])
        
        # For each parameter, do targeted retrieval and prepare context
        param_contexts = {}
        param_queries = {}
        for param in parameters:
            # Targeted retrieval for this specific parameter
            targeted_query = self._build_single_param_query(param, brand)
            param_queries[param] = targeted_query
            targeted_retrieved = self.vector_store.search(
                query=targeted_query,
                top_k=10,
                filter_source=filename,
                filter_brand=brand
            )
            
            # Prepend directly-matched structured field documents
            structured_prefix = ""
            field_type = param_to_field_type.get(param)
            if field_type and field_type in structured_field_docs:
                sf_docs = structured_field_docs[field_type]
                # Filter by relevant indication for auth duration
                if field_type == "authorization_duration":
                    if "Initial" in param:
                        sf_docs = [d for d in sf_docs if d.metadata.get("auth_type") in ("initial", "unknown")]
                    elif "Reauthorization" in param:
                        sf_docs = [d for d in sf_docs if d.metadata.get("auth_type") in ("reauthorization", "unknown")]
                    # Prefer PsO indication
                    pso_docs = [d for d in sf_docs if d.metadata.get("indication") == "PsO"]
                    if pso_docs:
                        sf_docs = pso_docs
                elif field_type == "prescriber_restrictions":
                    # Prefer PsO indication
                    pso_docs = [d for d in sf_docs if d.metadata.get("indication") == "PsO"]
                    if pso_docs:
                        sf_docs = pso_docs
                elif field_type == "reauthorization_criteria":
                    pso_docs = [d for d in sf_docs if d.metadata.get("indication") == "PsO"]
                    if pso_docs:
                        sf_docs = pso_docs
                
                if sf_docs:
                    structured_prefix = "\n\n---\n\n".join([d.page_content for d in sf_docs[:3]])
                    structured_prefix = f"[DIRECTLY MATCHED STRUCTURED DATA]\n{structured_prefix}\n\n---\n\n"
            
            # Merge broad + targeted context (deduplicated)
            seen_hashes = set()
            merged_chunks = []
            
            all_results = (targeted_retrieved or []) + (broad_retrieved or [])
            for doc, score in all_results:
                doc_hash = hashlib.md5(doc.page_content[:200].encode()).hexdigest()
                if doc_hash not in seen_hashes:
                    seen_hashes.add(doc_hash)
                    merged_chunks.append(doc.page_content)
                if len(merged_chunks) >= RERANK_TOP_K:
                    break
            
            context = "\n\n---\n\n".join(merged_chunks) if merged_chunks else broad_context
            
            # Prepend structured field data to context
            if structured_prefix:
                context = structured_prefix + context
            
            param_contexts[param] = (context, len(merged_chunks), bool(structured_prefix))
        
        # Extract all parameters in parallel using ThreadPoolExecutor
        def _extract_param(param):
            context, num_chunks, has_structured = param_contexts[param]
            if not context:
                self.logger.warning(f"  No chunks retrieved for {param}")
                return param, "NA", num_chunks, has_structured
            value = self._extract_single_parameter(param, brand, context, filename)
            return param, value, num_chunks, has_structured
        
        PARALLEL_WORKERS = 3  # Conservative: accuracy > speed, avoids rate limit cascades
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_extract_param, param): param for param in parameters}
            for future in as_completed(futures):
                param = futures[future]
                try:
                    param_name, value, num_chunks, has_structured = future.result()
                    results[param_name] = value
                    self.logger.info(f"    Extracted: {param_name} = {str(value)[:80]}")
                    
                    # Log retrieval
                    log_entry = {
                        "filename": filename,
                        "brand": brand,
                        "parameter": param_name,
                        "query": param_queries[param_name],
                        "num_chunks": num_chunks,
                        "has_structured_field": has_structured,
                        "extracted_value": value,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self.retrieval_logs.append(log_entry)
                except Exception as e:
                    self.logger.error(f"    Error extracting {param}: {e}")
                    results[param] = "NA"
        
        return results
    
    def _build_single_param_query(self, parameter: str, brand: str) -> str:
        """Build a highly targeted retrieval query for a single parameter."""
        queries = {
            "Age": f"{brand} age years old eligibility plaque psoriasis coverage criteria members authorization granted",
            "Step Therapy Requirements Documented in Policy": f"{brand} step therapy documentation criteria prior authorization inadequate response intolerance contraindication trial failure biologic generic plaque psoriasis coverage criteria",
            "Number of Steps through Brands": f"{brand} step therapy biologic targeted synthetic drug previously received trial inadequate response preferred product",
            "Number of Steps through Generic": f"{brand} step therapy methotrexate cyclosporine acitretin topical pharmacologic treatment generic conventional",
            "Step through-Phototherapy": f"{brand} phototherapy UVB PUVA light therapy requirement step criteria plaque psoriasis",
            "TB Test required": f"tuberculosis TB test tuberculin skin test TST interferon-release assay IGRA screening documented negative biologic",
            "Quantity Limits": f"{brand} quantity level limit vial syringe dose days supply exception limit quantity limit",
            "Specialist Types": f"{brand} prescriber restrictions prescribed consultation dermatologist rheumatologist specialist",
            "Initial Authorization Duration(in-months)": f"{brand} coverage duration initial authorization months approval period plaque psoriasis",
            "Reauthorization Duration(in-months)": f"{brand} coverage duration renewal reauthorization months continuation approval period",
            "Reauthorization Required": f"{brand} renewal reauthorization continuation criteria required documentation clinical response",
            "Reauthorization Requirements Documented in Policy": f"{brand} renewal criteria reauthorization requirements documentation positive clinical response body surface area BSA improvement symptoms continuation",
        }
        return queries.get(parameter, f"{brand} {parameter} plaque psoriasis")
    
    def _build_retrieval_query(self, parameters: List[str], brand: str) -> str:
        """Build an optimized retrieval query for a parameter group."""
        param_keywords = {
            "Age": f"age eligibility criteria requirement {brand} psoriasis plaque indication years old",
            "Step Therapy Requirements Documented in Policy": f"step therapy prior authorization criteria requirements {brand} psoriasis biologic generic trial failure inadequate response",
            "Number of Steps through Brands": f"step therapy biologic branded steps {brand} psoriasis trial failure TNF inhibitor",
            "Number of Steps through Generic": f"step therapy generic non-biologic topical methotrexate cyclosporine {brand} psoriasis",
            "Step through-Phototherapy": f"phototherapy PUVA UVB light therapy step requirement {brand} psoriasis",
            "TB Test required": f"tuberculosis TB test screening requirement {brand} prior authorization",
            "Quantity Limits": f"quantity limit supply days units {brand} dosing quantity",
            "Specialist Types": f"specialist prescriber dermatologist rheumatologist physician {brand}",
            "Initial Authorization Duration(in-months)": f"initial authorization duration approval period months {brand}",
            "Reauthorization Duration(in-months)": f"reauthorization renewal duration months continuation {brand}",
            "Reauthorization Required": f"reauthorization renewal required continuation criteria {brand}",
            "Reauthorization Requirements Documented in Policy": f"reauthorization criteria requirements documentation clinical response {brand}",
        }
        
        queries = [param_keywords.get(p, p) for p in parameters]
        return " ".join(queries)
    
    def _build_alternative_query(self, parameters: List[str], brand: str) -> str:
        """Build alternative query for re-ranking."""
        alt_keywords = {
            "Age": f"{brand} adult pediatric age restriction indication approval criteria",
            "Step Therapy Requirements Documented in Policy": f"{brand} must have tried failed prior treatment criteria authorization",
            "Number of Steps through Brands": f"biologic targeted therapy failure {brand} prior authorization step",
            "Number of Steps through Generic": f"conventional therapy topical systemic {brand} prior treatment",
            "Step through-Phototherapy": f"phototherapy ultraviolet light PUVA UVB {brand} requirement",
            "TB Test required": f"tuberculin test PPD interferon gamma screening {brand}",
            "Quantity Limits": f"maximum quantity supply limit dispense {brand}",
            "Specialist Types": f"prescribing restrictions specialty dermatology {brand}",
            "Initial Authorization Duration(in-months)": f"initial approval valid period length authorization {brand}",
            "Reauthorization Duration(in-months)": f"renewal approval period reauthorization continuing therapy {brand}",
            "Reauthorization Required": f"continued authorization reassessment renewal needed {brand}",
            "Reauthorization Requirements Documented in Policy": f"renewal criteria documentation clinical improvement {brand}",
        }
        
        queries = [alt_keywords.get(p, p) for p in parameters]
        return " ".join(queries)
    
    def _extract_single_parameter(self, parameter: str, brand: str, 
                                   context: str, filename: str) -> str:
        """Extract a single parameter value using LLM."""
        business_rule = BUSINESS_RULES.get(parameter, "")
        
        # Parameter-specific extraction instructions
        param_instructions = {
            "Age": f"""Look for age-related eligibility criteria for {brand} in Plaque Psoriasis (PsO).
Check for phrases like "members X years of age and older", "age >=X", "adults", "pediatric".
Look in Coverage Criteria sections. Output the minimum age (e.g., ">=6", ">=18", "Any").""",

            "Step Therapy Requirements Documented in Policy": f"""Extract ALL step therapy/prior authorization criteria text for {brand} in Plaque Psoriasis.
Look for:
1. Universal/Documentation criteria that apply to ALL drugs (e.g., "patient is unable to take X product")
2. Indication-specific criteria for Plaque Psoriasis (e.g., "previously received a biologic", "inadequate response to...")
3. BSA requirements, phototherapy mentions, pharmacologic treatment requirements
Combine both universal AND indication-specific criteria. Extract the FULL relevant text.""",

            "Number of Steps through Brands": f"""Count branded/biologic steps required before {brand} approval for PsO.
COUNTING LOGIC:
1. Identify UNIVERSAL criteria (applies to all drugs) - e.g., "unable to take [preferred product]" = branded step
2. Identify INDICATION-SPECIFIC criteria for PsO - e.g., "previously received a biologic or targeted synthetic drug"
3. Combine universal + indication-specific with AND logic
4. Within each level, if conditions are connected by OR, take LEAST restrictive path (fewer steps)
5. A preferred product/biosimilar/biologic counts as a branded step
6. Exclude phototherapy and generic/conventional therapies
Output ONLY an integer or "NA".""",

            "Number of Steps through Generic": f"""Count non-biologic/generic steps required before {brand} approval for PsO.
COUNTING LOGIC:
1. Identify UNIVERSAL criteria - generic/conventional therapy requirements
2. Identify INDICATION-SPECIFIC criteria - e.g., "methotrexate, cyclosporine, acitretin", "topical agents", "pharmacologic treatment"
3. Combine universal + indication-specific with AND logic
4. Within each level, if connected by OR, take least restrictive path
5. Generic steps = methotrexate, cyclosporine, acitretin, topicals, conventional systemic therapies
6. Exclude phototherapy steps and biologic/branded steps
7. If phototherapy appears in an OR with pharmacologic treatment, and it's not mandatory alone, count the pharmacologic option as 1 generic step
Output ONLY an integer or "NA".""",

            "Step through-Phototherapy": f"""Determine if phototherapy is a MANDATORY step for {brand} PsO approval.
- "Yes" ONLY if phototherapy is a standalone mandatory requirement (not in OR condition)
- "No" if phototherapy appears in an OR condition with other options (e.g., "phototherapy OR pharmacologic treatment")
- "No" if phototherapy is not mentioned as a step requirement
- "N/A" if policy has no step criteria at all""",

            "TB Test required": f"""Look for tuberculosis/TB test requirements.
Search for: "tuberculosis", "TB test", "tuberculin skin test", "TST", "interferon-release assay", "IGRA", "documented negative".
Usually stated as "member has had a documented negative tuberculosis (TB) test" or similar.
Output "Yes" if TB test is required, "No" if explicitly not required, "NA" if not mentioned.""",

            "Quantity Limits": f"""Extract the EXACT quantity limit text for {brand} / its generic name.
Look for sections labeled "Quantity Level Limit" or "Quantity Limit".
Include: drug name, strength/formulation, number of vials/syringes, days supply, exception limits.
Do NOT confuse dosing/dosage information with quantity limits.
Output the full quantity limit text exactly as stated, or "NA" if not present.""",

            "Specialist Types": f"""Look for prescriber restriction requirements for {brand} specifically for Plaque Psoriasis (PsO).
Search for "Prescriber Restrictions", "prescribed by or in consultation with", specific specialties.
IMPORTANT: Only extract specialties listed for Plaque Psoriasis / PsO indication.
- Dermatologist is typically for PsO
- Rheumatologist is typically for PsA (psoriatic arthritis) - do NOT include unless explicitly listed for PsO
- If specialties are listed by indication (e.g., "Plaque Psoriasis: dermatologist"), only report the PsO one
Output the specialty name(s) for PsO only, or "NA" if no prescriber restriction mentioned.""",

            "Initial Authorization Duration(in-months)": f"""Find the INITIAL coverage/authorization duration for {brand} PsO.
Look for "Coverage Duration", "Initial: X months", "Authorization of X months may be granted".
Check tables with "Coverage Duration" rows.
Output the number of months (e.g., "6", "12"), "Unspecified" if PA is required but duration not stated, or "NA" if not applicable.""",

            "Reauthorization Duration(in-months)": f"""Find the RENEWAL/reauthorization duration for {brand} PsO.
Look for "Renewal: X months", "reauthorization period", "Coverage Duration" section.
Check tables with "Coverage Duration" rows that show both Initial and Renewal periods.
Output the number of months (e.g., "12", "6"), or "NA" if not mentioned.""",

            "Reauthorization Required": f"""Determine if reauthorization/renewal is required for {brand} PsO.
- "Yes" if there is any mention of renewal criteria, continuation requests, reauthorization requirements, or renewal duration
- "No" if explicitly stated that reauthorization is not needed
- "NA" if not mentioned at all
If a Renewal Duration or Renewal Criteria section exists, output "Yes".""",

            "Reauthorization Requirements Documented in Policy": f"""Extract the FULL reauthorization/renewal criteria text for {brand} PsO.
Look for: "Renewal Criteria", "Continuation requests", "Reauthorization requirements", "continuation of therapy".
Common criteria include: clinical response documentation, BSA improvement, symptom improvement.
Extract the complete text of renewal/reauthorization requirements, or "NA" if not documented.""",
        }

        system_prompt = f"""You are an expert at extracting structured information from payer Prior Authorization (PA) policy documents.
You must extract precise, accurate values based ONLY on the provided context.

CRITICAL RULES:
- Only extract information for the brand: {brand} (and its generic name / biosimilars)
- Only extract for Psoriasis (PsO) / Plaque Psoriasis (moderate-to-severe) indication
- If moderate-to-severe and severe criteria exist separately, use moderate-to-severe
- Universal criteria (applying to all brands/indications) should be combined with brand-specific using AND logic
- For OR conditions within a level, take the LEAST restrictive path
- Look carefully for tables, section headers like "Coverage Duration", "Prescriber Restrictions", "Quantity Level Limit"
- Output "NA" only if the information is truly absent from the context
- Be precise and exact - copy text verbatim where applicable"""

        specific_instruction = param_instructions.get(parameter, "")
        
        prompt = f"""Extract the following parameter from the policy document context.

PARAMETER: {parameter}

SPECIFIC INSTRUCTIONS:
{specific_instruction}

BUSINESS RULE:
{business_rule}

BRAND: {brand}
INDICATION: Psoriasis (PsO) / Plaque Psoriasis (moderate-to-severe)

DOCUMENT CONTEXT:
{context[:MAX_CONTEXT_CHARS]}

OUTPUT: Provide ONLY the extracted value for "{parameter}". Be concise.
Your answer:"""

        try:
            response = self.llm.invoke(prompt, system_prompt)
            # Clean up response
            value = response.strip()
            # Remove any markdown formatting
            value = re.sub(r'^```\w*\n?', '', value)
            value = re.sub(r'\n?```$', '', value)
            value = value.strip('"\'')
            
            # Strip reasoning/explanation text that some models append
            # Common patterns: "Reasoning:", "Explanation:", "Note:", multi-line after answer
            for sep in ['\n\nReasoning:', '\n\nExplanation:', '\n\nNote:', '\n\nJustification:']:
                if sep in value:
                    value = value.split(sep)[0].strip()
            # If multi-line and first line is short (likely the answer), take first line only
            if '\n' in value:
                first_line = value.split('\n')[0].strip()
                if len(first_line) < 100 and parameter not in ["Step Therapy Requirements Documented in Policy", "Reauthorization Requirements Documented in Policy", "Quantity Limits"]:
                    value = first_line
            
            # Validate response isn't too long for numeric fields
            if parameter in ["Number of Steps through Brands", "Number of Steps through Generic"]:
                # Try to extract just the number
                num_match = re.search(r'(\d+|NA|N/A)', value, re.IGNORECASE)
                if num_match:
                    val = num_match.group(1)
                    value = "NA" if val.upper() in ["NA", "N/A"] else val
            
            if parameter in ["Step through-Phototherapy", "TB Test required", "Reauthorization Required"]:
                # Normalize yes/no/na - check first line for the answer
                val_lower = value.lower().strip()
                if "yes" in val_lower:
                    value = "Yes"
                elif "no" in val_lower and "n/a" not in val_lower:
                    value = "No"
                elif "n/a" in val_lower or "na" == val_lower:
                    value = "NA"
            
            if parameter == "Initial Authorization Duration(in-months)" or parameter == "Reauthorization Duration(in-months)":
                # Try to extract months number
                months_match = re.search(r'(\d+)\s*(?:months?|mos?)', value, re.IGNORECASE)
                if months_match:
                    value = months_match.group(1)
                elif value.strip().isdigit():
                    pass  # already a number
                elif "unspecified" in value.lower():
                    value = "Unspecified"
                elif "na" in value.lower() or "not" in value.lower():
                    value = "NA"
            
            return value
            
        except Exception as e:
            self.logger.error(f"    Error extracting {parameter}: {e}")
            return "NA"
    
    def _calculate_access_score(self, params: Dict[str, str], filename: str, brand: str) -> str:
        """Calculate Access Score using hybrid approach (rules + LLM verification)."""
        self.logger.info(f"  Calculating Access Score...")
        
        # Rule-based initial score
        score = 50  # Start at parity
        
        # Age factor
        age_val = params.get("Age", "NA")
        if age_val.lower() in ["any", "na"]:
            score += 5  # No age restriction = slightly better
        elif "fda" in age_val.lower():
            score += 0  # Parity
        
        # Step therapy factors
        try:
            brand_steps = params.get("Number of Steps through Brands", "NA")
            if brand_steps != "NA":
                brand_steps_num = int(re.search(r'\d+', str(brand_steps)).group())
                if brand_steps_num == 0:
                    score += 15
                elif brand_steps_num == 1:
                    score -= 5
                elif brand_steps_num >= 2:
                    score -= 15
        except (ValueError, TypeError, AttributeError):
            pass
        
        try:
            generic_steps = params.get("Number of Steps through Generic", "NA")
            if generic_steps != "NA":
                generic_steps_num = int(re.search(r'\d+', str(generic_steps)).group())
                if generic_steps_num == 0:
                    score += 10
                elif generic_steps_num == 1:
                    score -= 5
                elif generic_steps_num >= 2:
                    score -= 10
        except (ValueError, TypeError, AttributeError):
            pass
        
        # Phototherapy
        photo = params.get("Step through-Phototherapy", "NA")
        if photo.lower() == "yes":
            score -= 10
        elif photo.lower() == "no":
            score += 5
        
        # TB Test (standard requirement, neutral impact)
        
        # Quantity Limits
        ql = params.get("Quantity Limits", "NA")
        if ql.lower() == "na" or ql.lower() == "none":
            score += 5  # No quantity limits = better
        elif ql.lower() != "na":
            score -= 5  # Has quantity limits
        
        # Specialist Types
        specialist = params.get("Specialist Types", "NA")
        if specialist.lower() == "na" or specialist.lower() == "none":
            score += 5  # No specialist needed
        
        # Authorization Duration
        try:
            init_auth = params.get("Initial Authorization Duration(in-months)", "NA")
            if init_auth != "NA" and init_auth.lower() != "unspecified":
                months = int(re.search(r'\d+', str(init_auth)).group())
                if months >= 12:
                    score += 10
                elif months >= 6:
                    score += 0
                else:
                    score -= 10
        except (ValueError, TypeError, AttributeError):
            pass
        
        # Reauthorization
        reauth = params.get("Reauthorization Required", "NA")
        if reauth.lower() == "no":
            score += 10
        elif reauth.lower() == "yes":
            score -= 5
        
        # Clamp to 0-100
        score = max(0, min(100, score))
        
        # Bucket to nearest 25
        buckets = [0, 25, 50, 75, 100]
        bucketed_score = min(buckets, key=lambda x: abs(x - score))
        
        # LLM verification (optional - disabled by default to save tokens)
        if ENABLE_ACCESS_SCORE_LLM:
            try:
                verify_prompt = f"""Parameters for {brand}: {json.dumps(params)}
Rule-based score: {bucketed_score}/100. Correct score (0/25/50/75/100)? Reply with ONLY a number."""
                
                llm_score = self.llm.invoke(verify_prompt, use_fallback=True)
                score_match = re.search(r'\b(0|25|50|75|100)\b', llm_score)
                if score_match:
                    llm_suggested = int(score_match.group())
                    final_score = int((bucketed_score + llm_suggested) / 2)
                    final_score = min(buckets, key=lambda x: abs(x - final_score))
                    self.logger.info(f"  Access Score: Rule={bucketed_score}, LLM={llm_suggested}, Final={final_score}")
                    return str(final_score)
            except Exception as e:
                self.logger.warning(f"  Score LLM verification failed: {e}")
        
        self.logger.info(f"  Access Score: {bucketed_score} (rule-based)")
        return str(bucketed_score)
    
    def save_logs(self, output_dir: str):
        """Save all retrieval logs."""
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_path = os.path.join(log_dir, "retrieval_logs.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.retrieval_logs, f, indent=2, ensure_ascii=False)
        
        # Save judge results
        judge_path = os.path.join(log_dir, "judge_results.json")
        with open(judge_path, "w", encoding="utf-8") as f:
            json.dump(self.judge.get_summary(), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Logs saved to {log_dir}")


# ============================================================================
# LANGGRAPH ORCHESTRATION
# ============================================================================

class PipelineState:
    """State class for the LangGraph pipeline."""
    def __init__(self):
        self.pdf_dir: str = ""
        self.output_dir: str = ""
        self.brands_df: Optional[pd.DataFrame] = None
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.vector_store: Optional[FAISSVectorStore] = None
        self.results: List[Dict] = []
        self.current_file: str = ""
        self.current_brand: str = ""
        self.status: str = "initialized"
        self.errors: List[str] = []


def create_pipeline_graph(state: PipelineState, logger: logging.Logger):
    """Create and execute the LangGraph pipeline."""
    
    # Initialize components
    pdf_parser = PDFParser(logger)
    # Build brand list dynamically from the input Excel
    brand_list = state.brands_df["Brand"].dropna().unique().tolist()
    chunker = HierarchicalChunker(logger, brands=brand_list)
    vector_store = FAISSVectorStore(logger)
    llm_client = LLMClient(logger, api_key=GROQ_API_KEY)
    judge = LLMJudge(llm_client, logger)
    extractor = RAGExtractor(llm_client, vector_store, judge, logger)
    
    # ---- Step 1: Parse all PDFs ----
    logger.info("\n" + "="*80)
    logger.info("STEP 1: PARSING PDFs")
    logger.info("="*80)
    
    index_dir = os.path.join(state.output_dir, "faiss_index")
    
    # Check if index already exists AND matches the current input PDFs
    index_valid = False
    if vector_store.load_index(index_dir):
        # Validate: check that indexed documents cover the PDFs in input_dir
        indexed_sources = set(doc.metadata.get("source", "") for doc in vector_store.documents)
        current_pdfs = set(f.name for f in Path(state.pdf_dir).glob("*.pdf"))
        missing_pdfs = current_pdfs - indexed_sources
        if missing_pdfs:
            logger.warning(f"FAISS index is stale: {len(missing_pdfs)} PDFs not indexed ({list(missing_pdfs)[:5]}...)")
            logger.info("Rebuilding index to include all current PDFs...")
            vector_store.index = None
            vector_store.documents = []
            vector_store.embeddings = None
        else:
            logger.info("Loaded existing FAISS index from disk (validated against input PDFs)")
            index_valid = True
    
    if not index_valid:
        # Parse all PDFs in the directory
        all_documents = []
        pdf_files = list(Path(state.pdf_dir).glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for i, pdf_path in enumerate(pdf_files):
            logger.info(f"  [{i+1}/{len(pdf_files)}] Parsing: {pdf_path.name}")
            try:
                docs = pdf_parser.parse_pdf(str(pdf_path))
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"  Error parsing {pdf_path.name}: {e}")
                state.errors.append(f"Parse error: {pdf_path.name}: {str(e)}")
        
        logger.info(f"Total document sections parsed: {len(all_documents)}")
        
        # ---- Step 2: Chunking ----
        logger.info("\n" + "="*80)
        logger.info("STEP 2: CHUNKING DOCUMENTS")
        logger.info("="*80)
        
        chunks = chunker.chunk_documents(all_documents)
        state.chunks = chunks
        
        # ---- Step 3: Build Vector Store ----
        logger.info("\n" + "="*80)
        logger.info("STEP 3: BUILDING VECTOR STORE")
        logger.info("="*80)
        
        vector_store.build_index(chunks)
        vector_store.save_index(index_dir)
    
    # ---- Step 4: Extract Parameters ----
    logger.info("\n" + "="*80)
    logger.info("STEP 4: EXTRACTING PARAMETERS")
    logger.info("="*80)
    
    results = []
    total_entries = len(state.brands_df)
    
    # Load previously completed results to append to (for resume continuity)
    intermediate_path = os.path.join(state.output_dir, "result_intermediate.csv")
    previous_results = []
    if os.path.exists(intermediate_path):
        try:
            prev_df = pd.read_csv(intermediate_path)
            previous_results = prev_df.to_dict('records')
            logger.info(f"Loaded {len(previous_results)} previous results for merge")
        except Exception:
            pass
    
    for idx, row in state.brands_df.iterrows():
        filename = row["Filename"]
        brand = row["Brand"]
        
        logger.info(f"\n[{idx+1}/{total_entries}] Processing: {filename} | {brand}")
        
        # If all models are exhausted, fill remaining records with NA and stop LLM calls
        if llm_client.all_models_exhausted:
            logger.warning(f"  Skipping LLM extraction (tokens exhausted) — filling with NA")
            result_row = {"Filename": filename, "Brand": brand}
            for param in PARAMETERS:
                result_row[param] = "NA"
            result_row["Access Score"] = "NA"
            results.append(result_row)
            all_results = previous_results + results
            intermediate_df = pd.DataFrame(all_results)
            intermediate_df.to_csv(intermediate_path, index=False, encoding="utf-8", na_rep="NA")
            continue
        
        try:
            params = extractor.extract_parameters(filename, brand)
            result_row = {"Filename": filename, "Brand": brand}
            result_row.update(params)
            results.append(result_row)
            
            # Save intermediate results after each extraction (cumulative: previous + current)
            all_results = previous_results + results
            intermediate_df = pd.DataFrame(all_results)
            intermediate_df.to_csv(
                intermediate_path, 
                index=False, encoding="utf-8", na_rep="NA"
            )
            
        except Exception as e:
            logger.error(f"Error processing {filename}/{brand}: {e}")
            logger.error(traceback.format_exc())
            state.errors.append(f"Extraction error: {filename}/{brand}: {str(e)}")
            
            # Add NA row
            result_row = {"Filename": filename, "Brand": brand}
            for param in PARAMETERS:
                result_row[param] = "NA"
            result_row["Access Score"] = "NA"
            results.append(result_row)
        
        # Respect rate limits between files
        time.sleep(2)
    
    state.results = results
    
    # ---- Step 5: Save Final Results ----
    logger.info("\n" + "="*80)
    logger.info("STEP 5: SAVING RESULTS")
    logger.info("="*80)
    
    # Save retrieval logs
    extractor.save_logs(state.output_dir)
    
    # Save final CSV (merge previous + current session results)
    all_results = previous_results + results
    results_df = pd.DataFrame(all_results)
    
    # Ensure column order matches submission format
    column_order = ["Filename", "Brand"] + PARAMETERS + ["Access Score"]
    for col in column_order:
        if col not in results_df.columns:
            results_df[col] = "NA"
    results_df = results_df[column_order]
    
    output_csv = os.path.join(state.output_dir, "result.csv")
    results_df.to_csv(output_csv, index=False, encoding="utf-8", na_rep="NA")
    logger.info(f"Results saved to: {output_csv}")
    
    # Save as Excel too
    output_xlsx = os.path.join(state.output_dir, "result.xlsx")
    results_df.to_excel(output_xlsx, index=False)
    logger.info(f"Results saved to: {output_xlsx}")
    
    # ---- Step 6: LLM Judge Summary ----
    logger.info("\n" + "="*80)
    logger.info("STEP 6: LLM JUDGE SUMMARY")
    logger.info("="*80)
    
    judge_summary = judge.get_summary()
    logger.info(f"Total judgments: {judge_summary['total']}")
    logger.info(f"Average score: {judge_summary['avg_score']:.2f}")
    
    # Performance summary
    logger.info(f"\n{'='*80}")
    logger.info("PIPELINE SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total LLM calls: {llm_client.call_count}")
    logger.info(f"Total tokens used: {llm_client.total_tokens}")
    logger.info(f"Entries processed: {len(results)}")
    logger.info(f"Errors encountered: {len(state.errors)}")
    
    if state.errors:
        logger.warning("Errors:")
        for err in state.errors:
            logger.warning(f"  - {err}")
    
    return state


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Brand-to-generic mapping for auto-detection
BRAND_GENERIC_MAP = {
    "TREMFYA": "guselkumab",
    "STELARA": "ustekinumab",
    "HUMIRA": "adalimumab",
    "SKYRIZI": "risankizumab",
    "COSENTYX": "secukinumab",
    "ENBREL": "etanercept",
    "OTEZLA": "apremilast",
    "CIMZIA": "certolizumab",
    "SILIQ": "brodalumab",
    "BIMZELX": "bimekizumab",
    "ILUMYA": "tildrakizumab",
    "TALTZ": "ixekizumab",
    "REMICADE": "infliximab",
    "SOTYKTU": "deucravacitinib",
    "RINVOQ": "upadacitinib",
    "XELJANZ": "tofacitinib",
    "AMJEVITA": "adalimumab-atto",
    "HADLIMA": "adalimumab-bwwd",
    "HYRIMOZ": "adalimumab-adaz",
    "CYLTEZO": "adalimumab-adbm",
    "ABRILADA": "adalimumab-afzb",
    "YESINTEK": "ustekinumab-aaly",
    "OTULFI": "ustekinumab-snhy",
    "ACITRETIN": "acitretin",
}


def detect_filename_brand_combinations(input_dir: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Auto-detect (filename, brand) combinations by scanning PDF content.
    Reads first few pages of each PDF and identifies which brands are referenced.
    Also attempts to extract brand from filename patterns.
    Returns a DataFrame with 'Filename' and 'Brand' columns.
    """
    logger.info("Auto-detecting filename-brand combinations from PDF content...")
    
    pdf_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')])
    logger.info(f"Found {len(pdf_files)} PDF files to scan")
    
    combinations = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        try:
            doc = fitz.open(pdf_path)
            # Read first 5 pages (or all if fewer) for brand detection
            max_pages = min(5, len(doc))
            text = ""
            for page_num in range(max_pages):
                text += doc[page_num].get_text("text") + "\n"
            doc.close()
            
            text_upper = text.upper()
            
            # Strategy 1: Detect known brands from BRAND_GENERIC_MAP
            detected_brands = []
            for brand, generic in BRAND_GENERIC_MAP.items():
                if brand.upper() in text_upper or generic.upper() in text_upper:
                    detected_brands.append(brand)
            
            # Strategy 2: Try to extract brand from filename
            # Common patterns: "Payer_BrandName_Policy.pdf", "BrandName PA Policy.pdf"
            if not detected_brands:
                fname_upper = pdf_file.upper().replace('.PDF', '').replace('_', ' ').replace('-', ' ')
                # Check known brands against filename
                for brand in BRAND_GENERIC_MAP.keys():
                    if brand.upper() in fname_upper:
                        detected_brands.append(brand)
                
                # Strategy 3: If still nothing, look for capitalized drug-like words in the document
                # that could be brand names (typically ALL-CAPS or Title-case, 4+ letters)
                if not detected_brands:
                    # Search for patterns like drug brand names in first page
                    # Look for words following "for" or "Policy" or in title position
                    brand_candidates = re.findall(
                        r'\b([A-Z][a-z]+(?:tek|mab|nib|lib|ept|ast|mod|zol|rix|umab|izumab|ximab|zumab|tinib|ciclib|rafenib)\b)',
                        text
                    )
                    # Also look for ALL-CAPS words (5+ chars) that appear near "prior authorization" context
                    allcaps_candidates = re.findall(r'\b([A-Z]{5,})\b', text[:3000])
                    # Filter out common non-drug words
                    non_drug_words = {"PRIOR", "AUTHORIZATION", "POLICY", "CRITERIA", "CLINICAL", 
                                      "COVERAGE", "MEDICAL", "HEALTH", "PHARMACY", "BENEFIT",
                                      "INITIAL", "RENEWAL", "CONTINUATION", "DOCUMENTATION",
                                      "REQUIRED", "REQUIREMENTS", "QUANTITY", "LIMIT", "LIMITS",
                                      "THERAPY", "TREATMENT", "SPECIALIST", "PRESCRIBER",
                                      "SECTION", "TABLE", "CONTENTS", "EFFECTIVE", "REVISED",
                                      "PLAQUE", "PSORIASIS", "PSORIATIC", "ARTHRITIS",
                                      "MODERATE", "SEVERE", "ADULT", "PEDIATRIC", "PATIENT",
                                      "MEMBERS", "MEMBER", "DIAGNOSIS", "MONTH", "MONTHS"}
                    allcaps_filtered = [w for w in allcaps_candidates if w not in non_drug_words and len(w) >= 5]
                    
                    # Combine candidates
                    all_candidates = brand_candidates + allcaps_filtered
                    if all_candidates:
                        # Use the most frequently mentioned candidate as brand
                        from collections import Counter
                        candidate_counts = Counter(all_candidates)
                        best_brand = candidate_counts.most_common(1)[0][0]
                        detected_brands.append(best_brand)
                        logger.info(f"  Auto-discovered new brand '{best_brand}' in {pdf_file}")
            
            if detected_brands:
                for brand in detected_brands:
                    combinations.append({"Filename": pdf_file, "Brand": brand})
            else:
                # If no brand detected, still include file with "UNKNOWN" brand
                logger.warning(f"  No brand detected in {pdf_file}")
                combinations.append({"Filename": pdf_file, "Brand": "UNKNOWN"})
                
        except Exception as e:
            logger.error(f"  Error scanning {pdf_file}: {e}")
            combinations.append({"Filename": pdf_file, "Brand": "UNKNOWN"})
    
    df = pd.DataFrame(combinations)
    logger.info(f"Auto-detected {len(df)} filename-brand combinations from {len(pdf_files)} PDFs")
    logger.info(f"Brands found: {df['Brand'].value_counts().to_dict()}")
    
    return df


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Payer Policy Intelligence - RAG Pipeline")
    # All paths are relative to the project root (where pipeline.py lives)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--input_dir", type=str, 
                        default=os.path.join(BASE_DIR, "data", "input_pdfs"),
                        help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(BASE_DIR, "output"),
                        help="Output directory for results")
    parser.add_argument("--brands_file", type=str, default=None,
                        help="Optional: Excel/CSV file with Filename and Brand columns. If omitted, auto-detects brands from PDFs.")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the primary LLM model name (e.g. llama-4.6-70b)")
    parser.add_argument("--groq_key", type=str, default=None,
                        help="Groq API key (overrides env variable)")
    parser.add_argument("--rebuild_index", action="store_true",
                        help="Force rebuild of FAISS index even if exists")
    parser.add_argument("--no-judge", action="store_true",
                        help="Disable LLM judge to save tokens on full dataset runs")
    parser.add_argument("--enable-score-llm", action="store_true",
                        help="Enable LLM verification of access score (costs 1 extra call per entry)")
    
    args = parser.parse_args()
    
    # Override API key if provided
    global GROQ_API_KEY, ENABLE_LLM_JUDGE, ENABLE_ACCESS_SCORE_LLM
    if args.no_judge:
        ENABLE_LLM_JUDGE = False
    if args.enable_score_llm:
        ENABLE_ACCESS_SCORE_LLM = True
    if args.groq_key:
        GROQ_API_KEY = args.groq_key
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("="*80)
    logger.info("PAYER POLICY INTELLIGENCE - RAG PIPELINE")
    logger.info("="*80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Brands file: {args.brands_file or 'AUTO-DETECT'}")
    
    # Load or auto-detect brands/filename mapping
    if args.brands_file and os.path.exists(args.brands_file):
        if args.brands_file.endswith('.xlsx'):
            brands_df = pd.read_excel(args.brands_file, sheet_name="Submissions")
        else:
            brands_df = pd.read_csv(args.brands_file)
        # Filter to only rows with Filename and Brand
        brands_df = brands_df[["Filename", "Brand"]].dropna()
        logger.info(f"Loaded {len(brands_df)} filename-brand combinations from file")
    else:
        # Auto-detect from PDF content
        logger.info("Brands file not found or not provided - auto-detecting from PDFs...")
        brands_df = detect_filename_brand_combinations(args.input_dir, logger)
    
    logger.info(f"Processing {len(brands_df)} filename-brand combinations")

    # Resume support: if intermediate results exist, skip already-processed combos
    intermediate_path = os.path.join(args.output_dir, "result_intermediate.csv")
    logger.info(f"Checking for intermediate results to support resume: {intermediate_path}")
    if os.path.exists(intermediate_path):
        try:
            done_df = pd.read_csv(intermediate_path)
            if {"Filename","Brand"}.issubset(set(done_df.columns)):
                done_pairs = set(zip(done_df['Filename'].astype(str), done_df['Brand'].astype(str)))
                before = len(brands_df)
                brands_df = brands_df[~brands_df.apply(lambda r: (str(r['Filename']), str(r['Brand'])) in done_pairs, axis=1)]
                after = len(brands_df)
                logger.info(f"Resuming: skipped {before-after} already-processed filename-brand combinations; {after} remaining")
            else:
                logger.info("Intermediate results file found but missing Filename/Brand columns - ignoring for resume")
        except Exception as e:
            logger.warning(f"Could not read intermediate results for resume: {e}")

    if len(brands_df) == 0:
        logger.info("No remaining filename-brand combinations to process. Exiting.")
        return

    # Allow overriding the primary model at runtime
    if args.model:
        old_model = globals().get('PRIMARY_MODEL', None)
        globals()['PRIMARY_MODEL'] = args.model
        logger.info(f"Primary model overridden: {old_model} -> {args.model}")

    # Delete existing index if rebuild requested
    if args.rebuild_index:
        index_dir = os.path.join(args.output_dir, "faiss_index")
        if os.path.exists(index_dir):
            import shutil
            shutil.rmtree(index_dir)
            logger.info("Deleted existing FAISS index for rebuild")

    # Initialize state
    state = PipelineState()
    state.pdf_dir = args.input_dir
    state.output_dir = args.output_dir
    state.brands_df = brands_df
    
    # Run pipeline
    start_time = time.time()
    
    try:
        state = create_pipeline_graph(state, logger)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        raise
    
    elapsed = time.time() - start_time
    logger.info(f"\nPipeline completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Results saved to: {os.path.join(args.output_dir, 'result.csv')}")
    
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Results: {os.path.join(args.output_dir, 'result.csv')}")
    print(f"Logs: {os.path.join(args.output_dir, 'logs')}")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*60}")


# ============================================================================
# LLM JUDGE TEST CASES
# ============================================================================

class LLMJudgeTestSuite:
    """Test suite with LLM Judge evaluation for validating pipeline quality."""
    
    def __init__(self, logger: logging.Logger, output_dir: str):
        self.logger = logger
        self.output_dir = output_dir
        self.test_results = []
        self.llm_client = LLMClient(logger, api_key=GROQ_API_KEY)
        self.judge = LLMJudge(self.llm_client, logger)
    
    def run_all_tests(self, results_df: pd.DataFrame = None, 
                      vector_store: FAISSVectorStore = None):
        """Run all test cases."""
        self.logger.info("\n" + "="*80)
        self.logger.info("RUNNING LLM JUDGE TEST SUITE")
        self.logger.info("="*80)
        
        # Test 1: Retrieval Quality Tests
        if vector_store:
            self._test_retrieval_quality(vector_store)
        
        # Test 2: Extraction Consistency Tests
        if results_df is not None and not results_df.empty:
            self._test_extraction_consistency(results_df)
        
        # Test 3: Access Score Validation Tests
        if results_df is not None and not results_df.empty:
            self._test_access_score_logic(results_df)
        
        # Test 4: Business Rule Compliance Tests
        if results_df is not None and not results_df.empty:
            self._test_business_rule_compliance(results_df)
        
        # Test 5: Output Format Tests
        if results_df is not None and not results_df.empty:
            self._test_output_format(results_df)
        
        # Summary
        self._print_summary()
        self._save_results()
    
    def _test_retrieval_quality(self, vector_store: FAISSVectorStore):
        """Test that retrieval returns relevant chunks for known queries."""
        self.logger.info("\n--- TEST 1: Retrieval Quality ---")
        
        test_queries = [
            {
                "query": "TREMFYA guselkumab prior authorization step therapy plaque psoriasis",
                "brand": "TREMFYA",
                "expected_keywords": ["tremfya", "guselkumab", "psoriasis", "prior authorization"],
                "description": "TREMFYA PA retrieval"
            },
            {
                "query": "STELARA ustekinumab age restriction eligibility psoriasis",
                "brand": "STELARA", 
                "expected_keywords": ["stelara", "ustekinumab", "age", "psoriasis"],
                "description": "STELARA age criteria retrieval"
            },
            {
                "query": "quantity limit supply days units biologic dosing",
                "brand": "TREMFYA",
                "expected_keywords": ["quantity", "limit", "supply", "dose"],
                "description": "Quantity limits retrieval"
            },
            {
                "query": "reauthorization renewal continuation criteria requirements",
                "brand": "STELARA",
                "expected_keywords": ["reauthorization", "renewal", "continuation", "criteria"],
                "description": "Reauthorization criteria retrieval"
            },
            {
                "query": "step therapy biologic generic topical failure inadequate response",
                "brand": "TREMFYA",
                "expected_keywords": ["step", "therapy", "fail", "trial"],
                "description": "Step therapy retrieval"
            },
        ]
        
        for tc in test_queries:
            results = vector_store.search(
                query=tc["query"],
                top_k=10,
                filter_brand=tc["brand"]
            )
            
            # Check that we got results
            if not results:
                self._record_test(tc["description"], "FAIL", "No results returned")
                continue
            
            # Check keyword presence in top chunks
            top_text = " ".join([doc.page_content.lower() for doc, _ in results[:5]])
            hits = sum(1 for kw in tc["expected_keywords"] if kw.lower() in top_text)
            relevance = hits / len(tc["expected_keywords"])
            
            # Use LLM Judge to evaluate retrieval quality
            chunks_text = [doc.page_content for doc, _ in results[:5]]
            judge_result = self.judge.judge_retrieval(
                tc["query"], chunks_text, tc["description"], tc["brand"]
            )
            
            score = judge_result.get("score", 0)
            if score >= 4:
                status = "PASS"
            elif score >= 3:
                status = "WARN"
            else:
                status = "FAIL"
            
            self._record_test(
                tc["description"], status,
                f"Judge score: {score}/5, Keyword hit rate: {relevance:.0%}, "
                f"Feedback: {judge_result.get('feedback', 'N/A')}"
            )
            time.sleep(2)  # Rate limit
    
    def _test_extraction_consistency(self, results_df: pd.DataFrame):
        """Test extraction results for logical consistency."""
        self.logger.info("\n--- TEST 2: Extraction Consistency ---")
        
        for _, row in results_df.iterrows():
            brand = row.get("Brand", "Unknown")
            filename = row.get("Filename", "Unknown")
            test_prefix = f"{filename}/{brand}"
            
            # Test: If Reauthorization Duration is non-NA, Reauthorization Required should be Yes
            reauth_dur = str(row.get("Reauthorization Duration(in-months)", "NA"))
            reauth_req = str(row.get("Reauthorization Required", "NA"))
            
            if reauth_dur.lower() not in ["na", "nan", ""] and reauth_req.lower() != "yes":
                self._record_test(
                    f"{test_prefix} - Reauth consistency",
                    "FAIL",
                    f"Reauth Duration={reauth_dur} but Reauth Required={reauth_req} (should be Yes)"
                )
            else:
                self._record_test(
                    f"{test_prefix} - Reauth consistency", "PASS",
                    "Reauth fields consistent"
                )
            
            # Test: Number of Steps should be numeric or NA
            for step_col in ["Number of Steps through Brands", "Number of Steps through Generic"]:
                val = str(row.get(step_col, "NA"))
                if val.lower() not in ["na", "nan", ""] and not re.match(r'^\d+$', val.strip()):
                    self._record_test(
                        f"{test_prefix} - {step_col} format",
                        "WARN",
                        f"Value '{val}' is not numeric or NA"
                    )
                else:
                    self._record_test(
                        f"{test_prefix} - {step_col} format", "PASS",
                        f"Value '{val}' is valid"
                    )
            
            # Test: Step through Phototherapy should be Yes/No/NA
            photo = str(row.get("Step through-Phototherapy", "NA"))
            if photo.strip().lower() not in ["yes", "no", "na", "n/a", "nan", ""]:
                self._record_test(
                    f"{test_prefix} - Phototherapy format",
                    "FAIL",
                    f"Value '{photo}' should be Yes/No/NA"
                )
            else:
                self._record_test(
                    f"{test_prefix} - Phototherapy format", "PASS",
                    f"Value '{photo}' is valid"
                )
            
            # Test: Access Score should be 0, 25, 50, 75, or 100
            score = str(row.get("Access Score", "NA"))
            if score.strip() not in ["0", "25", "50", "75", "100", "NA"]:
                self._record_test(
                    f"{test_prefix} - Access Score bucket",
                    "FAIL",
                    f"Score '{score}' not in valid buckets [0,25,50,75,100]"
                )
            else:
                self._record_test(
                    f"{test_prefix} - Access Score bucket", "PASS",
                    f"Score '{score}' is valid"
                )
    
    def _test_access_score_logic(self, results_df: pd.DataFrame):
        """Test access score logic using LLM Judge."""
        self.logger.info("\n--- TEST 3: Access Score Validation (LLM Judge) ---")
        
        # Sample up to 3 rows for LLM judge validation
        sample_size = min(3, len(results_df))
        sample_rows = results_df.sample(n=sample_size, random_state=42)
        
        for _, row in sample_rows.iterrows():
            brand = row.get("Brand", "Unknown")
            filename = row.get("Filename", "Unknown")
            score = str(row.get("Access Score", "NA"))
            
            # Collect all parameters for this row
            params = {}
            for param in PARAMETERS:
                params[param] = str(row.get(param, "NA"))
            
            # Ask LLM Judge to independently score
            judge_prompt = f"""Given these extracted parameters from a payer policy for brand {brand}:

{json.dumps(params, indent=2)}

{ACCESS_SCORE_RULES}

Based on ONLY these extracted parameters, what Access Score (0, 25, 50, 75, or 100) would you assign?
Explain your reasoning in 2-3 sentences, then provide the score.

Respond in JSON: {{"reasoning": "<explanation>", "score": <0|25|50|75|100>}}"""
            
            try:
                response = self.llm_client.invoke(judge_prompt, use_fallback=True)
                json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                if json_match:
                    judge_result = json.loads(json_match.group())
                    judge_score = str(judge_result.get("score", "NA"))
                    reasoning = judge_result.get("reasoning", "N/A")
                    
                    # Check if within 1 bucket (25 points) of pipeline score
                    if score != "NA" and judge_score != "NA":
                        diff = abs(int(score) - int(judge_score))
                        if diff <= 25:
                            status = "PASS"
                        else:
                            status = "FAIL"
                        
                        self._record_test(
                            f"{filename}/{brand} - Access Score Judge",
                            status,
                            f"Pipeline={score}, Judge={judge_score}, Diff={diff}. Reasoning: {reasoning}"
                        )
                    else:
                        self._record_test(
                            f"{filename}/{brand} - Access Score Judge",
                            "WARN", f"Could not compare: pipeline={score}, judge={judge_score}"
                        )
                else:
                    self._record_test(
                        f"{filename}/{brand} - Access Score Judge",
                        "WARN", "Could not parse judge response"
                    )
            except Exception as e:
                self._record_test(
                    f"{filename}/{brand} - Access Score Judge",
                    "ERROR", f"Judge failed: {str(e)}"
                )
            
            time.sleep(3)  # Rate limit between judge calls
    
    def _test_business_rule_compliance(self, results_df: pd.DataFrame):
        """Test compliance with business rules using LLM Judge."""
        self.logger.info("\n--- TEST 4: Business Rule Compliance (LLM Judge) ---")
        
        # Pick 2 random rows for deep business rule check
        sample_size = min(2, len(results_df))
        sample_rows = results_df.sample(n=sample_size, random_state=123)
        
        for _, row in sample_rows.iterrows():
            brand = row.get("Brand", "Unknown")
            filename = row.get("Filename", "Unknown")
            
            params = {param: str(row.get(param, "NA")) for param in PARAMETERS}
            
            check_prompt = f"""You are checking whether extracted values comply with business rules.

Brand: {brand}
Extracted Parameters:
{json.dumps(params, indent=2)}

Business Rules:
1. Age: Output "Any" if no age restriction, actual value if specified, or "FDA labelled age"
2. Number of Steps: Must be integer or "NA"
3. Step through-Phototherapy: Must be "Yes", "No", or "N/A"
4. TB Test: Must be "Yes", "No", or "NA"
5. Quantity Limits: Exact text or "NA"
6. Specialist Types: Specific specialties or "NA"
7. Auth Durations: Number (months) or "NA" or "Unspecified"
8. Reauthorization Required: "Yes" if reauth duration/requirements are non-NA, else "No" or "NA"
9. Access Score: Must be 0, 25, 50, 75, or 100

Check each rule. Respond in JSON:
{{"compliant": <true/false>, "violations": ["<list of violations>"], "score": <1-5 overall compliance>}}"""
            
            try:
                response = self.llm_client.invoke(check_prompt, use_fallback=True)
                json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    compliant = result.get("compliant", False)
                    violations = result.get("violations", [])
                    br_score = result.get("score", 3)
                    
                    status = "PASS" if compliant else ("WARN" if br_score >= 3 else "FAIL")
                    self._record_test(
                        f"{filename}/{brand} - Business Rules",
                        status,
                        f"Compliant: {compliant}, Score: {br_score}/5, Violations: {violations}"
                    )
                else:
                    self._record_test(
                        f"{filename}/{brand} - Business Rules",
                        "WARN", "Could not parse response"
                    )
            except Exception as e:
                self._record_test(
                    f"{filename}/{brand} - Business Rules",
                    "ERROR", str(e)
                )
            
            time.sleep(3)
    
    def _test_output_format(self, results_df: pd.DataFrame):
        """Test output format compliance."""
        self.logger.info("\n--- TEST 5: Output Format ---")
        
        # Check all required columns exist
        expected_cols = ["Filename", "Brand"] + PARAMETERS + ["Access Score"]
        missing_cols = [c for c in expected_cols if c not in results_df.columns]
        
        if missing_cols:
            self._record_test("Column completeness", "FAIL", 
                            f"Missing columns: {missing_cols}")
        else:
            self._record_test("Column completeness", "PASS",
                            f"All {len(expected_cols)} columns present")
        
        # Check no completely empty rows
        all_na_rows = results_df[PARAMETERS + ["Access Score"]].apply(
            lambda row: all(str(v).lower() in ["na", "nan", ""] for v in row), axis=1
        )
        na_count = all_na_rows.sum()
        
        if na_count > 0:
            self._record_test("Empty rows check", "WARN",
                            f"{na_count} rows have all NA values")
        else:
            self._record_test("Empty rows check", "PASS", "No fully empty rows")
        
        # Check no null Filename or Brand
        null_keys = results_df[["Filename", "Brand"]].isnull().any().any()
        if null_keys:
            self._record_test("Key columns null check", "FAIL",
                            "Filename or Brand has null values")
        else:
            self._record_test("Key columns null check", "PASS",
                            "No null keys")
    
    def _record_test(self, name: str, status: str, detail: str):
        """Record a test result."""
        self.test_results.append({
            "test": name,
            "status": status,
            "detail": detail
        })
        icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "ERROR": "!"}
        self.logger.info(f"  [{icon.get(status, '?')}] {status}: {name} - {detail[:120]}")
    
    def _print_summary(self):
        """Print test summary."""
        total = len(self.test_results)
        passed = sum(1 for t in self.test_results if t["status"] == "PASS")
        failed = sum(1 for t in self.test_results if t["status"] == "FAIL")
        warned = sum(1 for t in self.test_results if t["status"] == "WARN")
        errors = sum(1 for t in self.test_results if t["status"] == "ERROR")
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"TEST SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total Tests: {total}")
        self.logger.info(f"  PASSED:  {passed}")
        self.logger.info(f"  FAILED:  {failed}")
        self.logger.info(f"  WARNED:  {warned}")
        self.logger.info(f"  ERRORS:  {errors}")
        self.logger.info(f"Pass Rate: {passed/total*100:.1f}%" if total > 0 else "N/A")
        self.logger.info(f"{'='*60}")
    
    def _save_results(self):
        """Save test results to file."""
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        test_path = os.path.join(log_dir, "test_results.json")
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump({
                "test_results": self.test_results,
                "summary": {
                    "total": len(self.test_results),
                    "passed": sum(1 for t in self.test_results if t["status"] == "PASS"),
                    "failed": sum(1 for t in self.test_results if t["status"] == "FAIL"),
                    "warned": sum(1 for t in self.test_results if t["status"] == "WARN"),
                }
            }, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Test results saved to: {test_path}")


# ============================================================================
# TEST EXECUTION ENTRYPOINT
# ============================================================================

def run_tests(input_dir: str = None, output_dir: str = None, brands_file: str = None, 
              limit: int = 3):
    """Run pipeline on a small subset with LLM Judge tests.
    
    Args:
        input_dir: PDF directory
        output_dir: Output directory
        brands_file: Excel with Filename/Brand
        limit: Max rows to process for testing (default 3)
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_dir = input_dir or os.path.join(BASE_DIR, "data", "input_pdfs")
    output_dir = output_dir or os.path.join(BASE_DIR, "test_output")
    brands_file = brands_file or os.path.join(BASE_DIR, "data", "brands.xlsx")
    
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info("="*80)
    logger.info("RUNNING END-TO-END TEST WITH LLM JUDGE")
    logger.info("="*80)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Limit: {limit} entries")
    
    # Load brands
    if brands_file.endswith('.xlsx'):
        brands_df = pd.read_excel(brands_file, sheet_name="Submissions")
    else:
        brands_df = pd.read_csv(brands_file)
    
    brands_df = brands_df[["Filename", "Brand"]].dropna()
    
    # Limit to subset for testing
    brands_df = brands_df.head(limit)
    logger.info(f"Testing with {len(brands_df)} entries: {brands_df[['Filename','Brand']].values.tolist()}")
    
    # Initialize components
    pdf_parser = PDFParser(logger)
    brand_list = brands_df["Brand"].dropna().unique().tolist()
    chunker = HierarchicalChunker(logger, brands=brand_list)
    vector_store = FAISSVectorStore(logger)
    llm_client = LLMClient(logger, api_key=GROQ_API_KEY)
    judge = LLMJudge(llm_client, logger)
    extractor = RAGExtractor(llm_client, vector_store, judge, logger)
    
    # Step 1: Parse PDFs needed for our test subset
    logger.info("\n[TEST] Step 1: Parsing PDFs...")
    needed_files = brands_df["Filename"].unique().tolist()
    all_documents = []
    
    for fname in needed_files:
        pdf_path = os.path.join(input_dir, fname)
        if os.path.exists(pdf_path):
            docs = pdf_parser.parse_pdf(pdf_path)
            all_documents.extend(docs)
            logger.info(f"  Parsed {fname}: {len(docs)} sections")
        else:
            logger.warning(f"  PDF not found: {pdf_path}")
    
    if not all_documents:
        logger.error("No documents parsed! Check input directory.")
        return
    
    # Step 2: Chunking
    logger.info("\n[TEST] Step 2: Chunking...")
    chunks = chunker.chunk_documents(all_documents)
    logger.info(f"  Total chunks: {len(chunks)}")
    
    # Step 3: Build vector store
    logger.info("\n[TEST] Step 3: Building vector store...")
    vector_store.build_index(chunks)
    
    # Step 4: Extract parameters
    logger.info("\n[TEST] Step 4: Extracting parameters...")
    results = []
    
    for idx, row in brands_df.iterrows():
        filename = row["Filename"]
        brand = row["Brand"]
        logger.info(f"\n  [{idx+1}/{len(brands_df)}] {filename} | {brand}")
        
        try:
            params = extractor.extract_parameters(filename, brand)
            result_row = {"Filename": filename, "Brand": brand}
            result_row.update(params)
            results.append(result_row)
        except Exception as e:
            logger.error(f"  Error: {e}")
            result_row = {"Filename": filename, "Brand": brand}
            for param in PARAMETERS:
                result_row[param] = "NA"
            result_row["Access Score"] = "NA"
            results.append(result_row)
        
        time.sleep(2)
    
    # Build results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "test_results.csv"), index=False, na_rep="NA")
    logger.info(f"\n  Results saved: {os.path.join(output_dir, 'test_results.csv')}")
    
    # Print results table
    logger.info("\n" + "="*80)
    logger.info("EXTRACTION RESULTS")
    logger.info("="*80)
    for _, row in results_df.iterrows():
        logger.info(f"\n  {row['Filename']} | {row['Brand']}")
        for param in PARAMETERS:
            val = str(row.get(param, 'NA'))[:80]
            logger.info(f"    {param}: {val}")
        logger.info(f"    Access Score: {row.get('Access Score', 'NA')}")
    
    # Step 5: Run LLM Judge Tests
    logger.info("\n[TEST] Step 5: Running LLM Judge Test Suite...")
    test_suite = LLMJudgeTestSuite(logger, output_dir)
    test_suite.run_all_tests(results_df=results_df, vector_store=vector_store)
    
    # Save extractor logs
    extractor.save_logs(output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("END-TO-END TEST COMPLETE")
    logger.info("="*80)
    
    return results_df


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run test mode with LLM Judge
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        run_tests(limit=limit)
    else:
        main()
