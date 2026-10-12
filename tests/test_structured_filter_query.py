import unittest
from unittest.mock import patch

import incident_graph
import incidents
import rag_service
from security import contains_instruction


class StructuredFilterQueryTests(unittest.TestCase):
    def setUp(self):
        self.nodes = incident_graph._make_nodes(
            rag_service,
            incidents,
            contains_instruction,
        )

    def test_structured_filter_applies_application_with_priority(self):
        result = self.nodes["structured_filter"]({
            "query": "Show P1 incidents in Payment API",
        })

        self.assertEqual(1, result["incident_count"])
        source_comments = next(
            incident["comments"]
            for incident in incidents.incidents
            if incident["id"] == "INC001"
        )
        self.assertEqual(
            [
                {"incident_id": "INC001", "evidence": comment}
                for comment in source_comments
            ],
            result["analysis"]["supporting_evidence"],
        )

    def test_structured_filter_applies_status_and_release(self):
        result = self.nodes["structured_filter"]({
            "query": "Show P1 open incidents for Release 4.3",
        })

        self.assertEqual(1, result["incident_count"])
        self.assertIn("INC001", result["analysis"]["confirmed_facts"][0])
        self.assertEqual(
            "passed",
            result["analysis"]["quality_checks"]["sensitive_data"]["status"],
        )
        self.assertEqual(
            "passed",
            result["analysis"]["quality_checks"]["grounding"]["status"],
        )

    def test_multiple_application_filters_match_either_application(self):
        result = self.nodes["structured_filter"]({
            "query": "Show P1 incidents in Payment API and Customer Portal",
        })

        self.assertEqual(2, result["incident_count"])
        self.assertEqual(
            ["INC001", "INC003"],
            [
                fact.split(":", 1)[0]
                for fact in result["analysis"]["confirmed_facts"]
            ],
        )

    def test_multiple_applications_combine_with_priority_status_and_release(self):
        result = self.nodes["structured_filter"]({
            "query": (
                "Show P1 incidents in Payment API and Customer Portal "
                "with status Open for Release 4.3"
            ),
        })

        self.assertEqual(1, result["incident_count"])
        self.assertIn("INC001", result["analysis"]["confirmed_facts"][0])

    def test_unmatched_filter_values_return_unsupported_without_incidents(self):
        queries = (
            "Show P1 incidents for application Unknown Service",
            "Show incidents with status Pending",
            "Show incidents for Release 99.1",
            "Show P9 incidents",
        )
        for query in queries:
            with self.subTest(query=query):
                classified = self.nodes["classify"]({"query": query})
                self.assertEqual("UNSUPPORTED_FILTER", classified["query_type"])
                self.assertEqual("unsupported_filter", classified["route"])
                result = self.nodes["unsupported_filter"]({"query": query})
                self.assertEqual("unsupported", result["status"])
                self.assertEqual(0, result["incident_count"])
                self.assertEqual({}, result["analysis"])
                self.assertNotIn("INC00", str(result))

    def test_direct_rag_request_does_not_retrieve_for_unsupported_filter(self):
        with (
            patch.object(
                rag_service,
                "sync_incidents",
                side_effect=AssertionError("Unsupported query triggered sync."),
            ),
            patch.object(
                rag_service,
                "build_structured_context",
                side_effect=AssertionError("Unsupported query triggered retrieval."),
            ),
            self.assertLogs(rag_service.logger, level="INFO") as captured,
        ):
            response = rag_service.ask_incident_question(
                "Show incidents for Release 99.1",
                trace_id="trace-unsupported",
            )

        self.assertEqual("unsupported", response["filter_status"])
        self.assertEqual([], response["supporting_evidence"])
        self.assertIn("rag_request_complete", captured.output[0])
        self.assertIn("trace_id=trace-unsupported", captured.output[0])
        self.assertIn("status=unsupported", captured.output[0])

    def test_explicit_unknown_incident_logs_rag_completion_without_retrieval(self):
        with (
            patch.object(
                rag_service,
                "sync_incidents",
                side_effect=AssertionError("Explicit ID query triggered sync."),
            ),
            self.assertLogs(rag_service.logger, level="INFO") as captured,
        ):
            response = rag_service.ask_incident_question(
                "Tell me about INC123456.",
                trace_id="trace-explicit-id",
            )

        self.assertEqual("not_found", response["security_status"])
        self.assertIn("rag_request_complete", captured.output[0])
        self.assertIn("trace_id=trace-explicit-id", captured.output[0])
        self.assertIn("status=not_found", captured.output[0])

    def test_structured_route_blocks_sensitive_evidence(self):
        incident_with_pii = {
            **next(
                incident for incident in incidents.incidents
                if incident["id"] == "INC001"
            ),
            "comments": ["Contact user@example.com about the timeout."],
        }
        with patch.object(
            rag_service,
            "filter_incidents",
            return_value=[incident_with_pii],
        ):
            with self.assertLogs(incident_graph.logger, level="ERROR"):
                result = self.nodes["structured_filter"]({
                    "query": "Show P1 incidents in Payment API",
                    "trace_id": "trace-sensitive",
                })

        self.assertEqual("error", result["status"])
        self.assertEqual({}, result["analysis"])
        self.assertNotIn("user@example.com", str(result))

    def test_final_response_validation_rejects_untraceable_evidence(self):
        result = {
            "confirmed_facts": [],
            "observed_patterns": [],
            "probable_contributing_factors": [],
            "missing_evidence": [],
            "recommended_investigation_areas": [],
            "supporting_evidence": [{
                "incident_id": "INC001",
                "evidence": "Not present in source comments.",
            }],
        }
        context = [{
            "incident_id": "INC001",
            "comments": ["API timeout reported"],
            "validation": {
                "classification": "INCIDENT",
                "usable_for_reasoning": True,
            },
        }]

        with self.assertRaisesRegex(ValueError, "Grounding validation failed"):
            rag_service.validate_final_response(result, context)

    def test_reasoning_and_comparison_override_structured_filter_routing(self):
        for query in (
            "Explain why P1 incidents in Payment API occurred.",
            "Compare P1 incidents in Payment API.",
        ):
            with self.subTest(query=query):
                result = self.nodes["classify"]({"query": query})
                self.assertEqual("investigation", result["route"])

    def test_generic_incident_listing_uses_structured_route_and_returns_records(self):
        for query in (
            "List incidents",
            "Show incidents",
            "What incidents do we have?",
        ):
            with self.subTest(query=query):
                classification = rag_service.classify_query(query)
                self.assertEqual("STRUCTURED_FILTER", classification)

        route = self.nodes["classify"]({"query": "List incidents"})
        self.assertEqual("structured", route["route"])

        result = self.nodes["structured_filter"]({"query": "List incidents"})
        self.assertEqual("success", result["status"])
        self.assertEqual(5, result["incident_count"])
        self.assertTrue(result["analysis"]["supporting_evidence"])

    def test_incident_word_does_not_make_another_query_a_kg_lookup(self):
        self.assertEqual("GENERAL", rag_service.classify_query("What is incident management?"))
        self.assertEqual("KG_HEAVY", rag_service.classify_query("Tell me about INC001"))

    def test_structured_context_applies_every_recognized_filter(self):
        matching_incident = next(
            incident for incident in incidents.incidents
            if incident["id"] == "INC001"
        )
        with (
            patch.object(rag_service, "filter_incidents") as filter_mock,
            patch.object(
                rag_service,
                "retrieve_kg_context",
                side_effect=AssertionError("Structured query used KG retrieval."),
            ),
        ):
            filter_mock.return_value = [matching_incident]
            context = rag_service.build_structured_context(
                "Show P1 open incidents in Payment API for Release 4.3"
            )

        filter_mock.assert_called_once_with(
            priority=["P1"],
            applications=["Payment API"],
            status="Open",
            release="Release 4.3",
        )
        self.assertEqual(["INC001"], context["kg_context"]["related_incidents"])

    def test_filters_constrain_evidence_for_comparison_questions(self):
        query = "Compare P1 incidents in Payment API and Customer Portal"
        with (
            patch.object(
                rag_service,
                "retrieve_kg_context",
                return_value={
                    "bug_id": None,
                    "release_id": None,
                    "related_incidents": [],
                    "context": [],
                },
            ),
            patch.object(
                rag_service,
                "search_incidents",
                return_value={
                    "ids": [["INC001", "INC002", "INC003", "INC004", "INC005"]],
                    "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
                },
            ),
        ):
            context = rag_service.build_structured_context(query)

        self.assertEqual(
            {"INC001", "INC003"},
            {item["incident_id"] for item in context["validated_context"]},
        )

    def test_filters_constrain_evidence_for_causal_questions(self):
        query = "Did Release 4.3 cause P1 incidents in Payment API?"
        with (
            patch.object(
                rag_service,
                "retrieve_kg_context",
                return_value={
                    "bug_id": None,
                    "release_id": "Release 4.3",
                    "related_incidents": ["INC001", "INC002", "INC003"],
                    "context": [],
                },
            ),
            patch.object(
                rag_service,
                "search_incidents",
                return_value={
                    "ids": [["INC001", "INC002", "INC003", "INC004", "INC005"]],
                    "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
                },
            ),
        ):
            context = rag_service.build_structured_context(query)

        self.assertEqual(
            ["INC001"],
            [item["incident_id"] for item in context["validated_context"]],
        )


if __name__ == "__main__":
    unittest.main()
