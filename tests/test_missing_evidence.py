import unittest
from types import SimpleNamespace
from unittest.mock import patch

import rag_service


class MissingEvidenceTests(unittest.TestCase):
    def analyze_with_missing_evidence(self, missing_evidence):
        llm_result = {
            "confirmed_facts": [],
            "observed_patterns": [],
            "probable_contributing_factors": [],
            "missing_evidence": missing_evidence,
            "recommended_investigation_areas": [],
            "supporting_evidence": [],
        }
        structured_context = {
            "query": "Did Release 4.3 cause INC001?",
            "validated_context": [{
                "incident_id": "INC001",
                "validation": {"causation_supported": False},
            }],
            "release_id": "Release 4.3",
        }
        with (
            patch.object(rag_service, "sync_incidents"),
            patch.object(
                rag_service,
                "build_structured_context",
                return_value=structured_context,
            ),
            patch.object(rag_service, "build_llm_context", return_value={}),
            patch.object(
                rag_service,
                "rag_chain",
            ) as rag_chain,
        ):
            rag_chain.invoke.return_value = SimpleNamespace(
                model_dump=lambda: llm_result
            )
            return rag_service.ask_incident_question(structured_context["query"])

    def test_specific_requirements_are_not_followed_by_a_redundant_summary(self):
        specific_requirements = [
            "Before-and-after logs.",
            "Deployment configuration differences.",
            "Metrics and traces.",
            "A confirmed bug-to-release relationship.",
        ]

        result = self.analyze_with_missing_evidence(specific_requirements)

        self.assertEqual(specific_requirements, result["missing_evidence"])

    def test_specific_requirements_are_added_when_model_returns_none(self):
        result = self.analyze_with_missing_evidence([])

        self.assertEqual(4, len(result["missing_evidence"]))
        self.assertTrue(any("logs" in item.lower() for item in result["missing_evidence"]))
        self.assertTrue(any("metrics" in item.lower() for item in result["missing_evidence"]))
        self.assertTrue(any("configuration" in item.lower() for item in result["missing_evidence"]))
        self.assertTrue(any("bug" in item.lower() for item in result["missing_evidence"]))


if __name__ == "__main__":
    unittest.main()
