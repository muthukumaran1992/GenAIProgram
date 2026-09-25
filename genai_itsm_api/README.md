# GenAI ITSM API

This project contains a small FastAPI service for ITSM incident data.

## Structure

- `app/` - application source
- `requirements.txt` - Python dependencies

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Example endpoints

- `GET /incidents`
- `GET /incidents?priority=P1`
- `GET /summary`
