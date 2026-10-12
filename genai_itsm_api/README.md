# GenAI ITSM API

A FastAPI service for browsing sample ITSM incidents, viewing incident summaries,
and requesting incident analysis. The investigation endpoint uses the repository's
LangGraph and retrieval-augmented generation (RAG) workflow.

> **Note:** This service currently reads incident records from the repository's
> root-level `incidents.py`; the records are sample data, not a production ITSM
> connection. The API does not configure authentication. Do not expose it publicly
> or use it with production incident data without adding suitable access controls
> and reviewing data-handling requirements.

## Requirements

- Python 3.10 or newer
- An OpenAI API key for LLM-backed analysis

Install the dependencies from the repository root:

```powershell
python -m pip install -r genai_itsm_api\requirements.txt
```

Set `OPENAI_API_KEY` in your shell before starting the service, or provide it
through the repository's existing `Keys\.env` configuration. Do not commit API
keys or other credentials.

Optional environment variables:

| Variable | Purpose |
| --- | --- |
| `CHROMA_DB_PATH` | Chroma persistence directory. Relative paths resolve from the repository root; use an absolute path for a mounted or persistent volume. |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Enables Azure Monitor telemetry when the optional Azure Monitor package is installed. |

## Run locally

Run these commands from the repository root (`GenAIProgram`):

```powershell
python -m uvicorn genai_itsm_api.app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive API documentation
is at `http://127.0.0.1:8000/docs`; the OpenAPI schema is at
`http://127.0.0.1:8000/openapi.json`.

The repository-root working directory is important: the API imports shared
modules such as `incidents.py`, `incident_graph.py`, and `rag_service.py` from
there.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Basic welcome response. |
| `GET` | `/incidents` | List incidents. |
| `GET` | `/incidents?priority=P1` | List incidents filtered by priority. |
| `GET` | `/summary` | Return incident totals, priority counts, SLA breaches, and overall project status. |
| `POST` | `/incidents/ask` | Ask a question using the incident-analysis graph. |
| `GET` | `/incidents/{incident_id}/analyze` | Analyze one incident with the LLM. |

### Ask a question

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/incidents/ask `
  -ContentType "application/json" `
  -Body '{"question":"Which P1 incidents are open?"}'
```

The response includes a `trace_id`, query classification and route, status, and
an `analysis` object. If the investigation question explicitly identifies an
unknown incident, this endpoint returns `404`; analysis service failures return
`503`. Use the `trace_id` to correlate a failed request with application logs.

### Analyze one incident

```powershell
Invoke-RestMethod http://127.0.0.1:8000/incidents/INC001/analyze
```

## Data and persistence

The sample incident records live in the repository-root `incidents.py`. The RAG
workflow stores its Chroma vector database in the repository-root `chroma_db`
directory by default. Set `CHROMA_DB_PATH` to a persistent location when running
in an environment where local application files are not durable.

The RAG indexing workflow screens incident text for detected sensitive data and
instruction-like content before embedding. These checks are safeguards, not a
guarantee that all sensitive or malicious content will be detected. In particular,
`GET /incidents` returns the source records directly; use trusted, appropriately
sanitized data and add authorization before exposing the API.
