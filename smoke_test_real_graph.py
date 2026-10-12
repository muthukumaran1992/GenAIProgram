"""Smoke-test the compiled LangGraph with the real project modules.

Run this from the project directory after copying incident_graph.py beside
rag_service.py and installing all project dependencies. A valid OpenAI key may
be required to initialize ChatOpenAI, but this structured-filter smoke test
should not make an LLM request.
"""
from incident_graph import run_incident_graph

result = run_incident_graph("What are the current P1 incidents?")
print(result)

assert result["status"] == "success", result
assert result["query_type"] == "STRUCTURED_FILTER", result
assert result["route"] == "structured", result
assert result["incident_count"] == 2, result
facts = result["analysis"]["confirmed_facts"]
assert any("INC001" in fact for fact in facts), facts
assert any("INC003" in fact for fact in facts), facts
assert not any("INC999" in fact for fact in facts), facts
print("PASS: compiled graph returned canonical P1 incidents INC001 and INC003; INC999 excluded.")
