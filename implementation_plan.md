# Version History and Automatic Rollback Plan

We will implement a structured Version History and Rollback system for the evolutionary algorithm (EA) notebook generator. This system will persist every code modification and user intent, validate changes, and revert the notebook state to the last working version in case of syntax or runtime errors.

## User Review Required

> [!IMPORTANT]
> - **Database Choice**: We propose using **SQLite** for storing structured, relational, and linear version records (version number, cells state, user intent, status, timestamp, and errors). SQLite is lightweight, serverless, and does not require external setup or new package dependencies.
> - **Mem0 & ChromaDB Integration**: Mem0 (with ChromaDB vector backend) will be used to log version history changes and intents semantically. This allows the Learner/Tutor Agent to reference the student's modification and rollback history during chat interactions.
> - **Automatic Rollback Trigger**: An automatic rollback to the last working version will trigger if:
>   1. Refined cells fail static AST syntax validation.
>   2. The debugger agent fails to fix a runtime traceback.

## Open Questions

> [!WARNING]
> 1. Should we add a manual `/rollback` or `/revert` endpoint so a user can explicitly go back to a specific version?
> 2. For the `/generate` endpoint, should we always start a fresh version sequence (Version 1) or append to a single session history? (We propose starting a new session state on `/generate`).

## Proposed Changes

---

### Core Storage & Version Control

#### [NEW] [version_store.py](file:///Users/abhijith/Documents/EvoAcademy-llm-agents/app/core/version_store.py)
Create a SQL-based storage manager for versions and user intents.
- Initialize a SQLite database `version_history.db` under `app/core/` or the project root.
- Define a table `notebook_versions` with columns:
  - `id` (INTEGER PRIMARY KEY)
  - `session_id` (TEXT)
  - `version_number` (INTEGER)
  - `user_intent` (TEXT)
  - `cells` (TEXT - JSON serialized dictionary of cells)
  - `compiled_script` (TEXT)
  - `status` (TEXT: e.g., `'working'`, `'failed'`, `'reverted'`)
  - `error_message` (TEXT, optional)
  - `created_at` (TIMESTAMP)
- Expose methods:
  - `save_version(session_id, user_intent, cells, compiled_script, status, error_message=None)`: saves a new version.
  - `get_latest_working_version(session_id)`: retrieves the last version with status `'working'`.
  - `get_version(session_id, version_number)`: retrieves a specific version.
  - `get_session_history(session_id)`: lists all versions and intents for metadata visualization.
  - `rollback_to_version(session_id, version_number)`: marks subsequent versions as reverted and retrieves the target version cells.

---

### API Endpoints

#### [MODIFY] [main.py](file:///Users/abhijith/Documents/EvoAcademy-llm-agents/app/main.py)
Integrate version history and automatic rollbacks into the FastAPI endpoints:
- **`/generate`**:
  - Run the generation graph.
  - Save the resulting cells as Version 1 with status `'working'` and intent `'Initial generation'`.
- **`/refine`**:
  - Retrieve the current session baseline. If no history exists, save the incoming `current_cells` as Version 0 (`'working'`).
  - Run `refine_graph.ainvoke`.
  - Perform AST syntax verification on the modified cells.
  - **If syntax verification fails**:
    - Save the failed cells as a new version with status `'failed'` and error details.
    - Retrieve the last working cells from the version store.
    - Return the reverted cells with a warning in `tutor_explanation`.
  - **If syntax verification succeeds**:
    - Save the new cells as a new version with status `'working'` and the user prompt as `user_intent`.
    - Log this modification as a memory in `mem0_client` so the tutor agent stays informed.
- **`/debug`**:
  - Run the `refine_graph.ainvoke` with `is_traceback_error=True`.
  - Validate the updated cells.
  - If valid, save as a new working version. If invalid or if the patch fails, automatically rollback to the last working version and return those cells.
- **New Endpoints**:
  - `@app.get("/sessions/{session_id}/history")`: Return a list of all versions and intents.
  - `@app.post("/sessions/{session_id}/rollback")`: Manually rollback to a specific version number.

---

## Verification Plan

### Automated Tests
We will create a test script `app/tests/test_version_control.py` to:
- Test initial generation saves version 1.
- Test refinement success saves version 2.
- Test refinement with syntax error automatically rolls back to version 2 and registers a failed version.
- Test `/rollback` endpoint explicitly reverting to a previous version.

### Manual Verification
- Start the server using `python -m app.main`.
- Send HTTP POST requests to `/generate` and `/refine`.
- Trigger a syntax error via refinement and verify the response returns the last working cells.
- Call the `/history` endpoint to view the list of generated versions and intents.
