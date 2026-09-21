import json
import os
import requests
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from .config import MOCK_LLM
from .prompts import PROMPT_TEMPLATE
from .schemas import AskResponse
from .vector_store import retrieve

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours"
]

class SupportState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved: list[dict]
    answer: str
    sources: list[str]
    confidence: float

def classify_intent(state: SupportState):
    q = state["query"].lower()
    intent = "policy_question" if any(k in q for k in POLICY_KEYWORDS) else "general_question"
    return {"intent": intent}

def _real_llm(prompt: str):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("MOCK_LLM=0 requires GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    last_error = None
    for _ in range(3):
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages":[{"role":"user","content":prompt}], "temperature":0},
                timeout=30,
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            data = json.loads(raw)
            return AskResponse(**data)
        except Exception as exc:
            last_error = exc
            prompt += "\nCorrective instruction: Return ONLY valid JSON matching answer, sources, confidence."
    raise RuntimeError(f"LLM structured-output validation failed after 3 attempts: {last_error}")

def retrieve_and_answer(state: SupportState):
    chunks = retrieve(state["query"], top_k=3)
    top = chunks[0]
    if MOCK_LLM:
        answer = f"Based on the retrieved context: {top['text'][:200]}"
        response = AskResponse(answer=answer, sources=[c["id"] for c in chunks], confidence=1.0)
    else:
        context = "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
        response = _real_llm(PROMPT_TEMPLATE.format(query=state["query"], context=context))
    return {"retrieved": chunks, "answer": response.answer, "sources": response.sources, "confidence": response.confidence}

def direct_answer(state: SupportState):
    if MOCK_LLM:
        response = AskResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0,
        )
    else:
        response = _real_llm(PROMPT_TEMPLATE.format(query=state["query"], context="No retrieval was required."))
    return {"answer": response.answer, "sources": response.sources, "confidence": response.confidence}

def route(state: SupportState):
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

def build_graph():
    graph = StateGraph(SupportState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()

app_graph = build_graph()

def ask(query: str) -> AskResponse:
    result = app_graph.invoke({"query": query})
    return AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
    )
