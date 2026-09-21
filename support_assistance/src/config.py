import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "zepto_policies"
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
