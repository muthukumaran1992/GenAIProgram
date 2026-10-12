import logging
import os
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel



try:
    from azure.monitor.opentelemetry import configure_azure_monitor
except ImportError:  # pragma: no cover - optional dependency in local environments
    def configure_azure_monitor():
        """No-op fallback when Azure Monitor OpenTelemetry is unavailable."""

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
except ImportError:  # pragma: no cover - optional dependency in local environments
    class FastAPIInstrumentor:
        @staticmethod
        def instrument_app(_app):
            return None

import incidents as incident_data
from ai_service import analyze_incident_with_llm
from incident_graph import run_incident_graph

if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor()

logger = logging.getLogger(__name__)

logger = logging.getLogger("incident_intelligence")
logger.setLevel(logging.INFO)
logger.info("Application Insights trace test")

app = FastAPI(title="GenAI ITSM API")
FastAPIInstrumentor.instrument_app(app)

'''
@app.get("/observability-test")
async def observability_test():
    logger.warning("Correlated telemetry test")
    return {"status": "ok"}
'''

logger = logging.getLogger("genai_itsm_api")

@app.get("/")
def read_root():
    logger.warning("Application Insights connectivity test")
    return {"message": "Welcome to the GenAI ITSM API!"}

@app.middleware("http")
async def request_observability(request: Request, call_next):
    """Log request metadata only; never log request bodies or incident content."""
    supplied_request_id = request.headers.get("X-Request-ID", "")
    # Accept only short, safe correlation IDs from callers; otherwise generate one.
    request_id = (
        supplied_request_id
        if supplied_request_id and len(supplied_request_id) <= 128
        and all(ch.isalnum() or ch in "-_." for ch in supplied_request_id)
        else str(uuid4())
    )
    request.state.request_id = request_id
    started = time.perf_counter()
    status_code = 500
    error_type = None

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        # Avoid logging exception text, which may contain sensitive request/data details.
        error_type = type(exc).__name__
        raise
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        route = getattr(request.scope.get("route"), "path", request.url.path)
        logger.info(
            "http_request_completed request_id=%s method=%s route=%s "
            "status_code=%s duration_ms=%.2f error_type=%s",
            request_id,
            request.method,
            route,
            status_code,
            duration_ms,
            error_type or "none",
        )


class Incident(BaseModel):
    id: str
    priority: str
    status: str
    sla_breached: bool
    application: str
    related_bugs: list[str]
    deployments: list[str]
    comments: list[str]

class IncidentQuestion(BaseModel):
    question: str


@app.post("/incidents/ask")
def ask_incident_question_api(payload: IncidentQuestion, http_request: Request):
    graph_started = time.perf_counter()
    result = run_incident_graph(payload.question)
    graph_duration_ms = (time.perf_counter() - graph_started) * 1000
    # Correlate API logs with the graph trace without logging the question or evidence.
    logger.info(
        "incident_analysis_completed request_id=%s graph_trace_id=%s "
        "query_type=%s route=%s result_status=%s duration_ms=%.2f",
        getattr(http_request.state, "request_id", "unknown"),
        result.get("trace_id", "unknown"),
        result.get("query_type", "UNKNOWN"),
        result.get("route", "UNKNOWN"),
        result.get("status", "UNKNOWN"),
        graph_duration_ms,
    )

    if result["status"] == "error":
        raise HTTPException(
            status_code=503,
            detail={
                "message": result["error_message"],
                "trace_id": result["trace_id"],
            },
        )
    if result["status"] == "not_found":
        raise HTTPException(
            status_code=404,
            detail={
                "message": result["error_message"],
                "trace_id": result["trace_id"],
            },
        )
    if result["status"] not in {"success", "blocked"}:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Incident analysis service returned an invalid status.",
                "trace_id": result["trace_id"],
            },
        )

    return {
        "question": payload.question,
        "trace_id": result["trace_id"],
        "query_type": result["query_type"],
        "route": result["route"],
        "status": result["status"],
        "analysis": result["analysis"],
    }

@app.get("/incidents", response_model=list[Incident])
def get_incidents(priority: str | None = None):
    if priority:
        return [
            incident
            for incident in incident_data.incidents
            if incident["priority"] == priority
        ]

    return incident_data.incidents


@app.get("/summary")
def get_summary():
    total_incidents = len(incident_data.incidents)
    p1_incidents = sum(1 for incident in incident_data.incidents if incident["priority"] == "P1")
    p2_incidents = sum(1 for incident in incident_data.incidents if incident["priority"] == "P2")
    sla_breached_count = sum(1 for incident in incident_data.incidents if incident["sla_breached"])

    project_status = (
        "Red"
        if sla_breached_count >= 2
        else "Amber"
        if sla_breached_count >= 1
        else "Green"
    )

    return {
        "total_incidents": total_incidents,
        "p1_incidents": p1_incidents,
        "p2_incidents": p2_incidents,
        "sla_breached_count": sla_breached_count,
        "project_status": project_status,
    }

@app.get("/incidents/{incident_id}/analyze")
def analyze_incident(incident_id: str):

    for incident in incident_data.incidents:

        if incident["id"] == incident_id:

            try:
                analysis = analyze_incident_with_llm(incident)
            except Exception as error:
                raise HTTPException(
                    status_code=503,
                    detail="LLM analysis service is unavailable",
                ) from error

            return {
                "incident_id": incident_id,
                "analysis": analysis,
            }

    return {
        "message": "Incident not found"
    }