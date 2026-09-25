import streamlit as st
import requests

st.set_page_config(page_title="GenAI ITSM", page_icon="🤖")

st.title("GenAI ITSM")
st.write("Incident Intelligence Dashboard")

col1, col2, col3, col4 = st.columns(4)

st.subheader("Project Status")

response = requests.get(
    "http://127.0.0.1:8000/summary"
)

if response.status_code == 200:
    summ = response.json()    
else:
    st.error("Unable to retrieve project status")
    
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

response = requests.get(
    "http://127.0.0.1:8000/incidents"    
)

if response.status_code == 200:
    incidents = response.json()    

    st.subheader("Incidents")
    st.dataframe(incidents)
   
else:
    st.error("Unable to retrieve incidents")

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

st.dataframe(filtered_incidents)
