# Zepto Support Assistant — RAG + LangGraph + FastAPI

This module implements the required grounded Zepto policy assistant. The graded baseline is fully offline from an LLM-provider perspective: `MOCK_LLM` defaults to `1`, so no LLM API call is made. Embeddings and retrieval run locally using Sentence Transformers and ChromaDB.

## Architecture

```text
8 Zepto policy documents
        |
        v
Ingestion / one chunk per document
        |
        v
SentenceTransformer: all-MiniLM-L6-v2
        |
        v
ChromaDB collection: zepto_policies
        |
        v
LangGraph StateGraph
        |
        +--> classify_intent
              |
       +------+------+
       |             |
policy_question  general_question
       |             |
       v             v
retrieve_and_answer  direct_answer
       |
       v
Pydantic answer/sources/confidence
       |
       v
FastAPI POST /ask
```

### Stage mapping

- **Ingestion:** `src/ingest.py` loads all 8 documents and creates one chunk per document.
- **Embedding:** `src/vector_store.py` uses `all-MiniLM-L6-v2`.
- **Storage/retrieval:** ChromaDB stores the vectors in the `zepto_policies` collection and retrieves the top 3 by cosine similarity.
- **Routing:** `src/graph.py` contains the LangGraph `StateGraph` and its three required nodes.
- **Generation:** `retrieve_and_answer` produces the grounded mock answer or, when `MOCK_LLM=0`, calls the optional real LLM using the structured prompt in `src/prompts.py`.
- **General queries:** `direct_answer` returns the required fixed mock response.
- **MOCK_LLM:** classification and retrieval always run locally. Only answer-generation changes when `MOCK_LLM=0`.

## Structured prompt

`src/prompts.py` contains the role-context-task-format-length skeleton, an explicit negative constraint against unsupported information, and a few-shot example. It is used by the optional real-LLM branch.

## Setup

From the repository root:

```powershell
pip install -r support_assistance/requirements.txt
```

Run from `support_assistance`:

```powershell
python build_index.py
```

This creates/updates the local ChromaDB collection.

## Run API

From the repository root:

```powershell
$env:MOCK_LLM="1"
python -m uvicorn support_assistance.src.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Required example calls

Policy query:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" -Body '{"query":"What is the refund policy?"}'
```

Example response shape:

```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours...",
  "sources": ["doc_02", "doc_06", "doc_01"],
  "confidence": 1.0
}
```

General query:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" -Body '{"query":"Hello, how are you?"}'
```

Example response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Tests

```powershell
pytest tests -v
```

## Docker

From the repository root:

```powershell
docker build -t zepto-support-assistant ./support_assistance
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```

Then call:

```text
POST http://localhost:7860/ask
```

The Dockerfile builds the local embedding index during image construction and starts the FastAPI application on port 7860.

## Optional real LLM

The graded baseline does not require an API key. If experimenting with the optional real-LLM path, set:

```powershell
$env:MOCK_LLM="0"
$env:GROQ_API_KEY="your-key"
```

The code retries structured-output validation up to three total attempts. Never commit an API key.
