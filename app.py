import streamlit as st
import requests

st.set_page_config(page_title="GenAI ITSM", page_icon="🤖")

API_URL = "http://127.0.0.1:8000"

st.title("GenAI ITSM")
st.write("Incident Intelligence Dashboard")

col1, col2, col3, col4 = st.columns(4)

st.subheader("Project Status")

try:
    response = requests.get(f"{API_URL}/summary", timeout=5)
    response.raise_for_status()
    summ = response.json()
except (requests.RequestException, ValueError):
    st.error("Unable to retrieve project status")
    summ = {
        "project_status": "Unknown",
        "total_incidents": 0,
        "p1_incidents": 0,
        "p2_incidents": 0,
        "sla_breached_count": 0,
    }
    
project_status = summ["project_status"]
    
if project_status == "Red":
   st.error("🔴 RED")
elif project_status == "Amber":
    st.warning("🟠 AMBER")
else:
    st.success("🟢 GREEN")

col1.metric("Total Incidents", summ["total_incidents"])
col2.metric("P1 Incidents", summ["p1_incidents"])
col3.metric("P2 Incidents", summ["p2_incidents"])
col4.metric("SLA Breaches", summ["sla_breached_count"])

try:
    response = requests.get(f"{API_URL}/incidents", timeout=5)
    response.raise_for_status()
    incidents = response.json()
except (requests.RequestException, ValueError):
    st.error("Unable to retrieve incidents")
    incidents = []

st.subheader("Incidents")
st.dataframe(incidents)

priority = st.selectbox(
    "Select Priority",
    ["All", "P1", "P2", "P3"]
)

if priority == "All":
    filtered_incidents = incidents
else:
    filtered_incidents = [
        incident
        for incident in incidents
        if incident["priority"] == priority
    ]

incident_options = ["All"] + [incident["id"] for incident in filtered_incidents]
selected_incident = st.selectbox(
    "Select Incident",
    incident_options
)

if selected_incident == "All":
    displayed_incidents = filtered_incidents
else:
    displayed_incidents = [
        incident
        for incident in filtered_incidents
        if incident["id"] == selected_incident
    ]

st.dataframe(displayed_incidents)

