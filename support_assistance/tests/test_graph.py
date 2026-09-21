from src.graph import classify_intent, route, direct_answer

def test_policy_route():
    state = {"query": "What is the refund policy?"}
    state.update(classify_intent(state))
    assert state["intent"] == "policy_question"
    assert route(state) == "retrieve_and_answer"

def test_general_route():
    state = {"query": "Hello there"}
    state.update(classify_intent(state))
    assert state["intent"] == "general_question"
    assert route(state) == "direct_answer"

def test_direct_answer():
    result = direct_answer({"query": "Hello"})
    assert result["sources"] == []
    assert "only answer questions about Zepto policies" in result["answer"]
