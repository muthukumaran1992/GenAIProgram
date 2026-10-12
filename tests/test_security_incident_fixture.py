import unittest
from unittest.mock import patch

import incidents
import rag_service
from security import contains_instruction
from tests.fixtures.security_incidents import malicious_instruction_incident


class SecurityIncidentFixtureTests(unittest.TestCase):
    def test_security_incident_is_in_canonical_incidents(self):
        incident_ids = {incident["id"] for incident in incidents.incidents}

        self.assertEqual(6, len(incidents.incidents))
        self.assertIn(malicious_instruction_incident["id"], incident_ids)

    def test_malicious_content_is_detected_and_excluded_from_reasoning(self):
        incident = next(
            item for item in incidents.incidents
            if item["id"] == malicious_instruction_incident["id"]
        )
        evidence = " ".join(incident["comments"])

        self.assertTrue(contains_instruction(evidence))
        validation = rag_service.validate_release_relationship(
            incident["id"],
            incident["deployments"][0],
            evidence,
        )
        self.assertEqual("INSTRUCTION", validation["classification"])
        self.assertFalse(validation["usable_for_reasoning"])

        expected_evidence = {
            "incident_id": "INC001",
            "evidence": "API timeout reported",
            "validation": {"classification": "INCIDENT"},
        }
        llm_context = rag_service.build_llm_context({
            "query": "Did Release 4.3 cause INC999?",
            "validated_context": [
                {
                    "incident_id": malicious_instruction_incident["id"],
                    "evidence": evidence,
                    "validation": validation,
                },
                expected_evidence,
            ],
        })

        self.assertEqual([expected_evidence], llm_context["validated_evidence"])

    def test_sync_preserves_canonical_security_incident_in_vector_index(self):
        canonical_documents = {
            incident["id"]: rag_service.incident_to_document(incident)
            for incident in incidents.incidents
        }

        class FakeCollection:
            def __init__(self):
                self.ids = set(canonical_documents)
                self.deleted_ids = []

            def get(self, ids=None, include=None):
                if ids is None:
                    if include != []:
                        raise AssertionError("Expected an IDs-only collection query.")
                    return {"ids": sorted(self.ids)}
                existing_ids = [incident_id for incident_id in ids if incident_id in self.ids]
                return {
                    "ids": existing_ids,
                    "documents": [
                        canonical_documents[incident_id]
                        for incident_id in existing_ids
                    ],
                }

            def delete(self, ids):
                self.deleted_ids.extend(ids)
                self.ids.difference_update(ids)

        fake_collection = FakeCollection()
        with patch.object(rag_service, "collection", fake_collection):
            rag_service.sync_incidents()

        self.assertEqual([], fake_collection.deleted_ids)
        self.assertIn(malicious_instruction_incident["id"], fake_collection.ids)


if __name__ == "__main__":
    unittest.main()
