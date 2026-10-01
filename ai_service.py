from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent / "Keys" / ".env")


def build_incident_prompt(incident):

    prompt = f"""
You are an IT incident management assistant.

Analyze the following incident.

Incident ID: {incident["id"]}
Priority: {incident["priority"]}
Status: {incident["status"]}
Application: {incident["application"]}
SLA Breached: {incident["sla_breached"]}
Related Bugs: {incident["related_bugs"]}
Deployments: {incident["deployments"]}
Comments: {incident["comments"]}

Provide:

1. Confirmed facts
2. Probable contributing factors
3. Business impact
4. Missing information
5. Recommended next investigation steps

Do not invent information that is not present in the incident.
"""

    return prompt

class IncidentAnalysis(BaseModel):
    confirmed_facts: list[str]
    probable_contributing_factors: list[str]
    business_impact: str
    missing_information: list[str]
    recommended_next_steps: list[str]


client = OpenAI()


def analyze_incident_with_llm(incident):
    prompt = build_incident_prompt(incident)

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_format=IncidentAnalysis,
    )

    analysis = response.choices[0].message.parsed
    return analysis.model_dump()