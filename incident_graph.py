"""First LangGraph orchestration layer for the Incident Intelligence project.

This module deliberately reuses the existing rag_service functions. Imports of
LangGraph and rag_service are delayed until graph construction so the pure
helpers and tests can run without loading vector-store clients or calling APIs.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, TypedDict
from uuid import uuid4
from time import perf_counter

logger = logging.getLogger(__name__)


class IncidentGraphState(TypedDict, total=False):
    query: str
    trace_id: str
    query_type: str
    route: str
    analysis: dict[str, Any]
    status: str
    error_message: str
    incident_count: int


def _has_untrusted_instruction(incident: dict[str, Any], contains_instruction_fn) -> bool:
    """Exclude security-test/malicious records from ordinary business results."""
    comments = incident.get("comments", [])
    return any(contains_instruction_fn(str(comment)) for comment in comments)


def build_structured_analysis(
    query: str,
    matching_incidents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a deterministic, LLM-free response for structured filter queries."""
    facts: list[str] = []
    supporting_evidence: list[dict[str, str]] = []

    for incident in matching_incidents:
        incident_id = str(incident.get("id", "UNKNOWN"))
        facts.append(
            f"{incident_id}: priority {incident.get('priority', 'Unknown')}; "
            f"status {incident.get('status', 'Unknown')}; "
            f"application {incident.get('application', 'Unknown')}; "
            f"SLA breached: {bool(incident.get('sla_breached', False))}."
        )
        comments = incident.get("comments", [])
        for comment in comments:
            supporting_evidence.append({
                "incident_id": incident_id,
                "evidence": str(comment),
            })

    if not facts:
        facts = ["No incidents matched the requested structured filters."]

    return {
        "confirmed_facts": facts,
        "observed_patterns": [
            f"{len(matching_incidents)} incident(s) matched the structured filters."
        ],
        "probable_contributing_factors": [],
        "missing_evidence": [],
        "recommended_investigation_areas": [],
        "supporting_evidence": supporting_evidence,
    }


def _make_nodes(rag_service_module, incident_module, contains_instruction_fn):
    """Build node functions with injected dependencies (also enables unit tests)."""

    def classify_node(state: IncidentGraphState) -> dict[str, Any]:
        query = state.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return {
                "query_type": "INVALID",
                "route": "invalid",
                "status": "error",
                "error_message": "Question must be a non-empty string.",
            }

        query_type = rag_service_module.classify_query(query)
        if query_type == "UNSUPPORTED_FILTER":
            route = "unsupported_filter"
        else:
            route = "structured" if query_type == "STRUCTURED_FILTER" else "investigation"
        return {"query_type": query_type, "route": route, "status": "running"}

    def unsupported_filter_node(state: IncidentGraphState) -> dict[str, Any]:
        filters = rag_service_module.parse_structured_query(
            state.get("query", "")
        )
        return {
            "analysis": {},
            "incident_count": 0,
            "status": "unsupported",
            "unsupported_filters": filters["unsupported_filters"],
            "error_message": (
                "One or more requested filter values could not be matched "
                "to the available incident data. No incidents were returned."
            ),
        }

    def structured_filter_node(state: IncidentGraphState) -> dict[str, Any]:
        query = state.get("query", "")
        filters = rag_service_module.parse_structured_query(query)
        if filters["unsupported_filters"]:
            return unsupported_filter_node(state)
        matches = rag_service_module.filter_incidents(
            priority=filters["priorities"] or None,
            applications=filters["applications"] or None,
            status=filters.get("status"),
            release=filters.get("release"),
        )

        # Defense in depth: security-test records containing instruction-like
        # comments are not included in ordinary business results.
        matches = [
            item for item in matches
            if not _has_untrusted_instruction(item, contains_instruction_fn)
        ]
        analysis = build_structured_analysis(query, matches)
        validated_context = [
            {
                "incident_id": str(item.get("id", "")),
                "application": item.get("application", "Unknown"),
                "priority": item.get("priority", "Unknown"),
                "status": item.get("status", "Unknown"),
                "releases": item.get("deployments", []),
                "related_bugs": item.get("related_bugs", []),
                "comments": item.get("comments", []),
                "validation": {
                    "classification": "INCIDENT",
                    "causation_supported": False,
                    "content_trust": "UNTRUSTED",
                    "instruction_pattern_screen": (
                        "known_pattern_detected"
                        if rag_service_module._has_detected_instruction_pattern(item)
                        else "no_known_pattern_detected_not_certified"
                    ),
                },
                "content_trust": "UNTRUSTED",
            }
            for item in matches
        ]
        try:
            analysis = rag_service_module.validate_final_response(
                analysis,
                validated_context,
            )
        except ValueError:
            logger.exception(
                "Structured response validation failed (trace_id=%s)",
                state.get("trace_id", "unknown"),
            )
            return {
                "analysis": {},
                "incident_count": 0,
                "status": "error",
                "error_message": (
                    "The structured response did not pass output validation. "
                    "No unvalidated analysis was returned."
                ),
            }
        return {
            "analysis": analysis,
            "incident_count": len(matches),
            "status": "success",
            "error_message": "",
        }

    def investigation_node(state: IncidentGraphState) -> dict[str, Any]:
        try:
            # Reuse the existing RAG function, including its current retrieval
            # and output-validation behavior. Do not duplicate that logic here.
            analysis = rag_service_module.ask_incident_question(
                state["query"], trace_id=state.get("trace_id")
            )
            if analysis.get("filter_status") == "unsupported":
                return {
                    "analysis": {},
                    "incident_count": 0,
                    "status": "unsupported",
                    "unsupported_filters": analysis.get(
                        "unsupported_filters", []
                    ),
                    "error_message": (
                        "One or more requested filter values could not be "
                        "matched to the available incident data. No incidents "
                        "were returned."
                    ),
                }
            if analysis.get("security_status") == "excluded_by_policy":
                return {
                    "analysis": analysis,
                    "status": "blocked",
                    "error_message": (
                        "The requested incident was excluded by the security policy. "
                        "No evidence from that record was used."
                    ),
                }
            if analysis.get("security_status") == "not_found":
                return {
                    "analysis": analysis,
                    "status": "not_found",
                    "error_message": "The explicitly requested incident ID was not found.",
                }
            return {
                "analysis": analysis,
                "status": "success",
                "error_message": "",
            }
        except Exception as error:
            logger.error(
                "incident_investigation_failed trace_id=%s error_type=%s",
                state.get("trace_id", "unknown"),
                type(error).__name__,
            )
            return {
                "analysis": {},
                "status": "error",
                "error_message": (
                    "Investigation failed. No unverified analysis was returned; "
                    "check application logs using the trace ID."
                ),
            }

    def invalid_node(state: IncidentGraphState) -> dict[str, Any]:
        return {
            "analysis": {},
            "status": "error",
            "error_message": state.get("error_message") or "Invalid question.",
        }

    return {
        "classify": classify_node,
        "unsupported_filter": unsupported_filter_node,
        "structured_filter": structured_filter_node,
        "investigate": investigation_node,
        "invalid": invalid_node,
    }


def build_incident_graph(rag_service_module=None, incident_module=None):
    """Compile and return the LangGraph workflow.

    Optional dependency injection is intended for tests. In the application,
    call with no arguments to use the project's real rag_service and incidents.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is not installed. Install the project requirements, "
            "including langgraph, before running this workflow."
        ) from exc

    if rag_service_module is None:
        import rag_service as rag_service_module
    if incident_module is None:
        import incidents as incident_module

    from security import contains_instruction

    nodes = _make_nodes(rag_service_module, incident_module, contains_instruction)
    graph = StateGraph(IncidentGraphState)
    graph.add_node("classify", nodes["classify"])
    graph.add_node("unsupported_filter", nodes["unsupported_filter"])
    graph.add_node("structured_filter", nodes["structured_filter"])
    graph.add_node("investigate", nodes["investigate"])
    graph.add_node("invalid", nodes["invalid"])

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        lambda state: state.get("route", "invalid"),
        {
            "structured": "structured_filter",
            "investigation": "investigate",
            "unsupported_filter": "unsupported_filter",
            "invalid": "invalid",
        },
    )
    graph.add_edge("structured_filter", END)
    graph.add_edge("unsupported_filter", END)
    graph.add_edge("investigate", END)
    graph.add_edge("invalid", END)
    return graph.compile()


@lru_cache(maxsize=1)
def get_incident_graph():
    """Build the application graph once per process."""
    return build_incident_graph()


def run_incident_graph(query: str) -> dict[str, Any]:
    """Invoke the compiled workflow with a trace ID and a validated query."""
    if not isinstance(query, str) or not query.strip():
        return {
            "query": query if isinstance(query, str) else "",
            "trace_id": str(uuid4()),
            "query_type": "INVALID",
            "route": "invalid",
            "analysis": {},
            "status": "error",
            "error_message": "Question must be a non-empty string.",
        }

    initial_state: IncidentGraphState = {
        "query": query.strip(),
        "trace_id": str(uuid4()),
    }
    graph_started = perf_counter()
    try:
        result = get_incident_graph().invoke(initial_state)
    except Exception as error:
        graph_latency_ms = round((perf_counter() - graph_started) * 1000, 2)
        logger.error(
            "incident_graph_failed trace_id=%s error_type=%s total_graph_ms=%.2f",
            initial_state["trace_id"],
            type(error).__name__,
            graph_latency_ms,
        )
        return {
            "query": initial_state["query"],
            "trace_id": initial_state["trace_id"],
            "query_type": "UNKNOWN",
            "route": "unknown",
            "analysis": {},
            "status": "error",
            "error_message": (
                "Investigation failed. No unverified analysis was returned; "
                "check application logs using the trace ID."
            ),
            "incident_count": 0,
        }

    graph_latency_ms = round((perf_counter() - graph_started) * 1000, 2)
    logger.info(
        "incident_graph_complete trace_id=%s query_type=%s route=%s status=%s "
        "incident_count=%s total_graph_ms=%.2f",
        result.get("trace_id", initial_state["trace_id"]),
        result.get("query_type", "UNKNOWN"),
        result.get("route", "unknown"),
        result.get("status", "error"),
        result.get("incident_count", 0),
        graph_latency_ms,
    )
    # Expose only the public response fields; keep internal graph state private.
    return {
        "query": result.get("query", query.strip()),
        "trace_id": result.get("trace_id", initial_state["trace_id"]),
        "query_type": result.get("query_type", "UNKNOWN"),
        "route": result.get("route", "unknown"),
        "analysis": result.get("analysis", {}),
        "status": result.get("status", "error"),
        "error_message": result.get("error_message", ""),
        "incident_count": result.get("incident_count"),
    }
