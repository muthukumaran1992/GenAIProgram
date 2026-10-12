import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import rag_service


class VectorStoreSecurityTests(unittest.TestCase):
    def setUp(self):
        self.sensitive_incident = {
            "id": "INC123",
            "application": "Payment API",
            "priority": "P1",
            "status": "Open",
            "deployments": ["Release 4.3"],
            "comments": ["Contact user@example.com about the timeout."],
        }

    def test_sensitive_incident_is_rejected_before_document_embedding(self):
        with self.assertRaisesRegex(ValueError, "cannot be embedded"):
            rag_service.incident_to_document(self.sensitive_incident)

    def test_sync_deletes_previously_persisted_sensitive_incident(self):
        class FakeCollection:
            def __init__(self):
                self.deleted_ids = []

            def get(self, ids=None, include=None):
                if ids is None:
                    return {"ids": ["INC123"]}
                return {
                    "ids": ["INC123"],
                    "documents": [
                        "Legacy document containing user@example.com"
                    ],
                    "metadatas": [{"instruction_pattern_detected": False}],
                }

            def delete(self, ids):
                self.deleted_ids.extend(ids)

        fake_collection = FakeCollection()
        with (
            patch.object(
                rag_service,
                "incidents",
                SimpleNamespace(incidents=[self.sensitive_incident]),
            ),
            patch.object(rag_service, "collection", fake_collection),
            patch.object(
                rag_service,
                "create_embedding",
                side_effect=AssertionError("Sensitive document was embedded."),
            ),
        ):
            rag_service.sync_incidents()

        self.assertEqual(["INC123"], fake_collection.deleted_ids)

    def test_sensitive_incident_is_excluded_from_structured_filters(self):
        with patch.object(
            rag_service,
            "incidents",
            SimpleNamespace(incidents=[self.sensitive_incident]),
        ):
            self.assertEqual(
                [],
                rag_service.filter_incidents(priority="P1"),
            )

    def test_chroma_path_is_anchored_to_module_or_absolute_override(self):
        self.assertEqual(
            rag_service.MODULE_DIR / "chroma_db",
            rag_service._resolve_chroma_path("chroma_db"),
        )
        absolute_path = Path(self.id()).resolve()
        self.assertEqual(
            absolute_path,
            rag_service._resolve_chroma_path(str(absolute_path)),
        )


if __name__ == "__main__":
    unittest.main()
