# Boeing Insight Chatbot — Google Drive ingestion + free embeddings (prototype)

Overview
This prototype ingests documents from Google Drive (including Google Docs), embeds them using a fast free model (sentence-transformers/all-MiniLM-L6-v2), stores embeddings in a local Chroma vector store, and exposes a FastAPI chat endpoint that returns retrieved contexts (or generates a reply via OpenAI if an API key is provided).

Why this stack
- Google Drive / Docs API: free to read documents you own/share with a service account.
- sentence-transformers (all-MiniLM-L6-v2): very fast, small, free for embeddings.
- Chroma (local duckdb+parquet): fast local vector store with persistence.
- FastAPI: lightweight production-ready API.

Files included
- app/gdrive.py — Drive + Docs download helpers (service account).
- app/embeddings.py — sentence-transformers wrapper.
- app/indexer.py — chunking + Chroma indexing and search.
- app/main.py — FastAPI with /ingest and /chat endpoints.
- requirements.txt, .env.example

Quick start
1. Create a GCP project and enable the Drive API.
2. Create a service account, share the Drive folder(s) with the service account email, and download the JSON key.
3. Set environment variables (or copy `.env.example` to `.env`):
   - SERVICE_ACCOUNT_FILE=/path/to/sa-key.json
   - CHROMA_PERSIST_DIR=./chroma_db
   - DRIVE_DOWNLOAD_DIR=./data/drive
   - (optional) OPENAI_API_KEY=sk-...
4. Install dependencies:
   pip install -r requirements.txt
5. Run the API:
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
6. Ingest a Drive folder:
   POST /ingest JSON {"drive_folder_id":"<folderId>"}
7. Chat:
   POST /chat JSON {"message":"How to submit an incident report?"}

Notes
- If OPENAI_API_KEY is not set, the /chat endpoint returns retrieved contexts only (no paid LLM calls).
- Do NOT commit your service account key to source control.

Security
- Use environment variables or secret managers for credentials.
- Share Drive folders only with the service account email.