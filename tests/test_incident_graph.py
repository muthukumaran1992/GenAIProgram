import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import incident_graph


class IncidentGraphTests(unittest.TestCase):
    def test_graph_invocation_failure_is_logged_and_returned_safely(self):
        graph = SimpleNamespace(
            invoke=Mock(side_effect=RuntimeError("sensitive provider details"))
        )

        with (
            patch.object(incident_graph, "get_incident_graph", return_value=graph),
            self.assertLogs(incident_graph.logger, level="ERROR") as captured,
        ):
            result = incident_graph.run_incident_graph("Compare incidents.")

        self.assertEqual("error", result["status"])
        self.assertEqual({}, result["analysis"])
        self.assertTrue(result["trace_id"])
        self.assertIn(f"trace_id={result['trace_id']}", captured.output[0])
        self.assertIn("error_type=RuntimeError", captured.output[0])
        self.assertNotIn("sensitive provider details", captured.output[0])
        self.assertNotIn("sensitive provider details", result["error_message"])

    def test_investigation_errors_are_logged_with_trace_id_and_sanitized(self):
        rag_service = SimpleNamespace(
            ask_incident_question=Mock(side_effect=RuntimeError("provider error"))
        )
        nodes = incident_graph._make_nodes(
            rag_service,
            SimpleNamespace(),
            lambda _text: False,
        )
        state = {"query": "Compare incidents.", "trace_id": "trace-123"}

        with self.assertLogs(incident_graph.logger, level="ERROR") as captured:
            result = nodes["investigate"](state)

        self.assertEqual({}, result["analysis"])
        self.assertEqual("error", result["status"])
        self.assertIn("trace_id=trace-123", captured.output[0])
        self.assertIn("error_type=RuntimeError", captured.output[0])
        self.assertNotIn("provider error", captured.output[0])
        self.assertNotIn("provider error", result["error_message"])


if __name__ == "__main__":
    unittest.main()
