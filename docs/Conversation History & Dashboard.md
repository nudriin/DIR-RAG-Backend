# Implementation Plan - Conversation History & Dashboard

This plan outlines the steps to implement persistent conversation history and a statistics dashboard for the RAG backend. We will use SQLite with SQLAlchemy (Async) for data storage.

## User Review Required

> [!IMPORTANT]
> This plan introduces a new database dependency (SQLite + SQLAlchemy).
> New endpoints will be added for accessing history and statistics.
> The `/api/chat` endpoint will be modified to save data asynchronously.

## Proposed Changes

### Dependencies
#### [MODIFY] [requirements.txt](file:///c:/Users/NURDIN/Downloads/Nurdin/00_KULIAH/INFORMATIKA/SMT%209/SKRIPSI/Program/be/DIR-RAG-Backend/requirements.txt)
- Add `sqlalchemy`
- Add `aiosqlite`

### Database Layer
#### [NEW] [app/db/engine.py](file:///c:/Users/NURDIN/Downloads/Nurdin/00_KULIAH/INFORMATIKA/SMT%209/SKRIPSI/Program/be/DIR-RAG-Backend/app/db/engine.py)
- Setup `AsyncEngine` and `AsyncSession`.
- `init_db` function to create tables.

#### [NEW] [app/db/models.py](file:///c:/Users/NURDIN/Downloads/Nurdin/00_KULIAH/INFORMATIKA/SMT%209/SKRIPSI/Program/be/DIR-RAG-Backend/app/db/models.py)
- `Conversation`: `id`, `created_at`, `title`, `metadata` (e.g. user info if any).
- `Message`: `id`, `conversation_id`, `role` (user/assistant), `content`, `created_at`, `tokens` (optional), `confidence`, `rag_iterations`.
- `Feedback` (Optional for now, but good for dashboard): `message_id`, `score`, `comment`.

#### [NEW] [app/db/crud.py](file:///c:/Users/NURDIN/Downloads/Nurdin/00_KULIAH/INFORMATIKA/SMT%209/SKRIPSI/Program/be/DIR-RAG-Backend/app/db/crud.py)
- `create_conversation`
- `add_message`
- `get_conversation_history`
- `get_all_conversations` (with pagination)
- `get_dashboard_stats` (total conversations, total messages, avg confidence, etc.)

### API Layer
#### [NEW] [app/api/dashboard.py](file:///c:/Users/NURDIN/Downloads/Nurdin/00_KULIAH/INFORMATIKA/SMT%209/SKRIPSI/Program/be/DIR-RAG-Backend/app/api/dashboard.py)
- `GET /api/history`: List recent conversations.
- `GET /api/history/{id}`: Get full chat log.
- `GET /api/dashboard/stats`: aggregated metrics.

#### [MODIFY] [app/api/chat.py](file:///c:/Users/NURDIN/Downloads/Nurdin/00_KULIAH/INFORMATIKA/SMT%209/SKRIPSI/Program/be/DIR-RAG-Backend/app/api/chat.py)
- Inject DB session.
- Create `Conversation` on new chat (or use existing if `conversation_id` provided in request).
- Save `User` query and `Assistant` response to DB.

#### [MODIFY] [app/main.py](file:///c:/Users/NURDIN/Downloads/Nurdin/00_KULIAH/INFORMATIKA/SMT%209/SKRIPSI/Program/be/DIR-RAG-Backend/app/main.py)
- Include `dashboard.router`.
- Call `init_db` on startup.

#### [MODIFY] [app/schemas/chat_schema.py](file:///c:/Users/NURDIN/Downloads/Nurdin/00_KULIAH/INFORMATIKA/SMT%209/SKRIPSI/Program/be/DIR-RAG-Backend/app/schemas/chat_schema.py)
- Add `conversation_id` (optional) to `ChatRequest`.
- Add `conversation_id` to `ChatResponse`.

## Verification Plan

### Automated Tests
- We will create a new test file `tests/test_dashboard.py` (if tests exist) or run manual `curl` requests.
- Since there are no existing tests visible in the file list (only `evaluation` folder which seems different), we will rely on manual verification via `curl` or a simple script.

### Manual Verification
1.  **Start the server**: `uvicorn app.main:app --reload`
2.  **Send a chat**: `POST /api/chat` with a query.
3.  **Check History**: `GET /api/history` should show the new conversation.
4.  **Check Details**: `GET /api/history/{id}` should show the messages.
5.  **Check Stats**: `GET /api/dashboard/stats` should show updated counts.
