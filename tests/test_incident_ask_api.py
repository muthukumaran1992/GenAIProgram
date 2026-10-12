import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import rag_service
from genai_itsm_api.app.main import app


class IncidentAskApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_security_block_is_returned_as_a_completed_analysis(self):
        analysis = {
            "security_status": "excluded_by_policy",
            "confirmed_facts": ["INC999 was excluded by the security rule."],
            "observed_patterns": [],
            "probable_contributing_factors": [],
            "missing_evidence": [],
            "recommended_investigation_areas": [],
            "supporting_evidence": [],
        }
        graph_result = {
            "status": "blocked",
            "error_message": "The incident was excluded by policy.",
            "trace_id": "trace-blocked",
            "query_type": "HYBRID",
            "route": "investigation",
            "analysis": analysis,
        }

        with patch(
            "genai_itsm_api.app.main.run_incident_graph",
            return_value=graph_result,
        ):
            response = self.client.post(
                "/incidents/ask",
                json={"question": "Analyze INC999."},
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual("blocked", response.json()["status"])
        self.assertEqual(analysis, response.json()["analysis"])

    def test_investigation_failure_remains_service_unavailable(self):
        graph_result = {
            "status": "error",
            "error_message": "Investigation failed.",
            "trace_id": "trace-error",
            "query_type": "HYBRID",
            "route": "investigation",
            "analysis": {},
        }

        with patch(
            "genai_itsm_api.app.main.run_incident_graph",
            return_value=graph_result,
        ):
            response = self.client.post(
                "/incidents/ask",
                json={"question": "Compare incidents."},
            )

        self.assertEqual(503, response.status_code)
        self.assertEqual("trace-error", response.json()["detail"]["trace_id"])

    def test_unknown_incident_is_returned_as_not_found(self):
        graph_result = {
            "status": "not_found",
            "error_message": "Incident not found.",
            "trace_id": "trace-not-found",
            "query_type": "HYBRID",
            "route": "investigation",
            "analysis": {},
        }

        with patch(
            "genai_itsm_api.app.main.run_incident_graph",
            return_value=graph_result,
        ):
            response = self.client.post(
                "/incidents/ask",
                json={"question": "Analyze INC123."},
            )

        self.assertEqual(404, response.status_code)
        self.assertEqual("trace-not-found", response.json()["detail"]["trace_id"])

    def test_real_ask_route_logs_rag_completion_for_explicit_unknown_id(self):
        with self.assertLogs(rag_service.logger, level="INFO") as captured:
            response = self.client.post(
                "/incidents/ask",
                json={"question": "Tell me about INC123456."},
            )

        self.assertEqual(404, response.status_code)
        trace_id = response.json()["detail"]["trace_id"]
        completion_log = next(
            message
            for message in captured.output
            if "rag_request_complete" in message
        )
        self.assertIn(f"trace_id={trace_id}", completion_log)
        self.assertIn("status=not_found", completion_log)


if __name__ == "__main__":
    unittest.main()
