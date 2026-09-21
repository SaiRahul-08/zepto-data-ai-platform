from fastapi import FastAPI
from .schemas import AskRequest, AskResponse
from .vector_store import build_index
from .graph import ask

app = FastAPI(title="Zepto Support Assistant RAG API", version="2.0.0")

@app.on_event("startup")
def startup():
    build_index()

@app.get("/")
def root():
    return {"service": "Zepto Support Assistant RAG API", "status": "running", "mock_llm": True}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    return ask(request.query)
