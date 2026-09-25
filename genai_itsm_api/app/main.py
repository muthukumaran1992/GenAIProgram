from fastapi import FastAPI
from pydantic import BaseModel


class Incident(BaseModel):
    id: str
    priority: str
    status: str
    sla_breached: bool
    application: str


app = FastAPI(title="GenAI ITSM API")

incidents = [
    {
        "id": "INC001",
        "priority": "P1",
        "status": "Open",
        "sla_breached": True,
        "application": "Payment API",
    },
    {
        "id": "INC002",
        "priority": "P2",
        "status": "Investigating",
        "sla_breached": False,
        "application": "Order Services",
    },
    {
        "id": "INC003",
        "priority": "P1",
        "status": "Open",
        "sla_breached": True,
        "application": "Customer Portal",
    },
]

@app.get("/")
def read_root():
    return {"message": "Welcome to the GenAI ITSM API!"}    

@app.get("/incidents", response_model=list[Incident])
def get_incidents(priority: str | None = None):
    if priority:
        return [
            incident
            for incident in incidents
            if incident["priority"] == priority
        ]

    return incidents


@app.get("/summary")
def get_summary():
    total_incidents = len(incidents)
    p1_incidents = sum(1 for incident in incidents if incident["priority"] == "P1")
    p2_incidents = sum(1 for incident in incidents if incident["priority"] == "P2")
    sla_breached_count = sum(1 for incident in incidents if incident["sla_breached"])

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
