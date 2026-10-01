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
    st.dataframe(displayed_incidents)
else:
    displayed_incidents = [
        incident
        for incident in filtered_incidents
        if incident["id"] == selected_incident
    ]
    st.subheader("Incident Details")
    st.dataframe(displayed_incidents)

    if st.button("Analyze Incident"):
        try:
            response = requests.get(
                f"{API_URL}/incidents/{selected_incident}/analyze",
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            analysis = result["analysis"]

            if isinstance(analysis, str):
                st.subheader("AI Analysis")
                st.write(analysis)
            else:
                st.subheader("Confirmed Facts")
                for fact in analysis["confirmed_facts"]:
                    st.write("•", fact)

                st.subheader("Probable Contributing Factors")
                for factor in analysis["probable_contributing_factors"]:
                    st.write("•", factor)

                st.subheader("Business Impact")
                st.write(analysis["business_impact"])

                st.subheader("Missing Information")
                for item in analysis["missing_information"]:
                    st.write("•", item)

                st.subheader("Recommended Next Steps")
                for step in analysis["recommended_next_steps"]:
                    st.write("•", step)
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            st.error(f"Unable to analyze incident: {error}")

st.divider()

st.header("Ask Incident Intelligence")

question = st.text_input(
    "Enter your incident question",
    placeholder="What evidence is available about the Payment API incident?"
)

if st.button("Ask AI"):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        response = requests.post(
            f"{API_URL}/incidents/ask",
            json={"question": question},
            timeout=60
        )

        response.raise_for_status()

        result = response.json()["analysis"]

        st.subheader("Confirmed Facts")
        for item in result["confirmed_facts"]:
            st.write(f"- {item}")

        st.subheader("Observed Patterns")
        for item in result["observed_patterns"]:
            st.write(f"- {item}")

        st.subheader("Probable Contributing Factors")
        for item in result["probable_contributing_factors"]:
            st.write(f"- {item}")

        st.subheader("Missing Evidence")
        for item in result["missing_evidence"]:
            st.write(f"- {item}")

        st.subheader("Recommended Investigation Areas")
        for item in result["recommended_investigation_areas"]:
            st.write(f"- {item}")

        st.subheader("Supporting Evidence")
        supporting_evidence = result.get("supporting_evidence", [])
        if supporting_evidence:
            for reference in supporting_evidence:
                st.write(f"**{reference['incident_id']}**: {reference['evidence']}")
        else:
            st.write("No supporting evidence was identified.")

    except requests.exceptions.RequestException as e:
        st.error(f"Unable to connect to Incident Intelligence API: {e}")

    except Exception as e:
        st.error(f"Unexpected error: {e}")

