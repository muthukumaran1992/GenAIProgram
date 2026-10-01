from urllib import request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import incidents as incident_data
from ai_service import analyze_incident_with_llm
from rag_service import ask_incident_question

app = FastAPI(title="GenAI ITSM API")


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
def ask_incident_question_api(request: IncidentQuestion):
    try:
        result = ask_incident_question(request.question)

        return {
            "question": request.question,
            "analysis": result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail="RAG analysis service is unavailable",
        ) from e

@app.get("/")
def read_root():
    return {"message": "Welcome to the GenAI ITSM API!"}    

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

    for incident in incidents:

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