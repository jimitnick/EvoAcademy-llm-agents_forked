# Git-like Version History System — Implementation Plan

## Overview

A complete architectural refactor to implement a production-grade, Git-like Version History System for the Evolutionary Algorithm notebook backend. The system cleanly separates concerns across four layers: **metadata (SQLite/SQLAlchemy)**, **file storage (disk/.ipynb)**, **semantic search (ChromaDB)**, and **user memory (Mem0)**. Existing API contracts are preserved.

---

## Proposed New Directory Structure

```
app/
├── api/
│   └── routes/
│       ├── __init__.py
│       ├── generate.py         # POST /generate
│       ├── refine.py           # POST /refine
│       ├── debug.py            # POST /debug
│       └── history.py          # GET/POST /sessions/*/history|rollback|search
├── db/
│   ├── __init__.py
│   ├── database.py             # SQLAlchemy engine + session factory
│   ├── models.py               # ORM models: User, Session, Notebook, NotebookVersion, VersionOperation
│   └── repositories/
│       ├── __init__.py
│       ├── notebook_repo.py    # CRUD for Notebook + active_version management
│       └── version_repo.py     # CRUD for NotebookVersion, history queries
├── services/
│   ├── __init__.py
│   ├── version_service.py      # Orchestrates full version lifecycle
│   ├── storage_service.py      # Read/write .ipynb files to disk
│   ├── chroma_service.py       # Embeddings, indexing, semantic search
│   └── memory_service.py       # Mem0 user preference CRUD
├── agents/                     # Unchanged
├── core/
│   ├── llm.py                  # Unchanged
│   └── memory.py               # Stripped down, delegates to memory_service
├── utils/
│   └── ast_parser.py           # Unchanged
└── main.py                     # Mounts APIRouters, initializes DB

storage/
└── notebooks/
    └── session_{session_id}/
        ├── version_1.ipynb
        ├── version_2.ipynb
        └── version_3.ipynb
```

---

## Database Schema (SQLAlchemy Models)

### `Users`
| Column | Type | Notes |
|---|---|---|
| `user_id` | UUID PK | Auto-generated |
| `username` | String | Optional identifier |
| `created_at` | DateTime | UTC |

### `Sessions`
| Column | Type | Notes |
|---|---|---|
| `session_id` | String PK | Provided by client |
| `user_id` | FK → Users | Optional, nullable |
| `created_at` | DateTime | UTC |
| `last_active_at` | DateTime | Updated on each request |

### `Notebooks`
| Column | Type | Notes |
|---|---|---|
| `notebook_id` | UUID PK | Auto-generated |
| `session_id` | FK → Sessions | |
| `active_version_id` | FK → NotebookVersions | Points to current active version |
| `created_at` | DateTime | UTC |
| `updated_at` | DateTime | UTC |

### `NotebookVersions`
| Column | Type | Notes |
|---|---|---|
| `version_id` | UUID PK | |
| `notebook_id` | FK → Notebooks | |
| `version_number` | Integer | Sequential per notebook |
| `parent_version_id` | FK → NotebookVersions | Self-referential (nullable for v1) |
| `operation_type` | Enum | `generate`, `refine`, `debug`, `rollback` |
| `prompt` | Text | User's original prompt |
| `summary` | Text | LLM-generated summary of changes |
| `file_path` | String | Relative path to `.ipynb` file |
| `checksum` | String | SHA256 of the notebook file |
| `chroma_indexed` | Boolean | Whether indexed in ChromaDB |
| `created_at` | DateTime | UTC |
| `metadata_json` | JSON | Extra fields (cells_modified, target_problem, etc.) |

### `VersionOperations` (Audit Log)
| Column | Type | Notes |
|---|---|---|
| `op_id` | UUID PK | |
| `version_id` | FK → NotebookVersions | |
| `action` | String | e.g. `activated`, `created`, `rolled_back_to` |
| `details` | Text | JSON details |
| `created_at` | DateTime | UTC |

---

## Service Layer

### `StorageService` (`app/services/storage_service.py`)
- `save_notebook(session_id, version_number, cells) -> (file_path, checksum)`: Serializes cells dict → `.ipynb` format, saves to `storage/notebooks/session_{id}/version_{n}.ipynb`, returns path + SHA256 checksum
- `load_notebook(file_path) -> Dict[str, str]`: Reads `.ipynb` file from disk, returns cells dict
- `checksum_matches(file_path, checksum) -> bool`: Integrity verification
- **Future**: Swap disk reads/writes for S3/MinIO without changing callers

### `VersionService` (`app/services/version_service.py`)
- `create_version(session_id, prompt, cells, operation_type) -> NotebookVersion`: Full lifecycle — save file, compute checksum, check for duplicates, insert DB record, index in ChromaDB, update active version
- `get_active_version(session_id) -> NotebookVersion`: Looks up `Notebook.active_version_id`
- `get_history(session_id) -> List[NotebookVersion]`: Ordered history with all metadata
- `rollback_to(session_id, version_number) -> NotebookVersion`: Updates `active_version_id` — **no file operations needed**
- `load_active_cells(session_id) -> Dict[str, str]`: Calls `StorageService.load_notebook`

### `ChromaService` (`app/services/chroma_service.py`)
- `index_version(version_id, session_id, prompt, summary, cells_modified, keywords)`: Creates embedding and stores in ChromaDB collection `notebook_versions`; document = `f"{prompt}\n{summary}\n{keywords}"`; metadata = `{version_id, session_id, version_number, operation_type}`
- `semantic_search(session_id, query, n_results) -> List[SearchResult]`: Returns ranked versions with `version_id`, `version_number`, `summary`, `relevance_score`
- **ChromaDB stores**: Lightweight text summaries only, never full notebook content

### `MemoryService` (`app/services/memory_service.py`)
- `get_user_preferences(session_id) -> str`: Retrieves all Mem0 memories for `user_id=session_id`, returns as formatted string for LLM injection
- `update_preferences(session_id, prompt, response_summary)`: Adds new interaction to Mem0 (Mem0 de-duplicates and extracts preferences automatically)
- **Mem0 stores**: Preferences only — preferred EA library, mutation operators, coding style, verbosity, visualization libraries, etc.

---

## Key Lifecycle: Every Notebook Modification

```
Client Request → API Route → VersionService
                                ├── 1. load active cells via StorageService
                                ├── 2. inject user preferences via MemoryService
                                ├── 3. call LLM workflow (generate_graph / refine_graph)
                                ├── 4. validate cells (AST parse)
                                │     └── on failure → auto-rollback to active (no new version)
                                ├── 5. generate LLM summary of changes
                                ├── 6. save .ipynb file via StorageService → get (path, checksum)
                                ├── 7. check if checksum == active_version.checksum → skip if duplicate
                                ├── 8. insert NotebookVersion row in DB
                                ├── 9. update Notebook.active_version_id
                                ├── 10. index summary in ChromaDB via ChromaService
                                └── 11. update Mem0 preferences via MemoryService
                                     └── return cells + version metadata
```

---

## Rollback Lifecycle

```
POST /sessions/{session_id}/rollback { version_number: 3 }
    → VersionService.rollback_to(session_id, 3)
        → look up NotebookVersion WHERE version_number=3
        → UPDATE Notebook SET active_version_id = <v3.version_id>   ← ONLY operation
        → log to VersionOperations
        → return cells from StorageService.load_notebook(v3.file_path)
```

No file is created, copied, or deleted. History is fully preserved.

---

## API Changes (Backward Compatible)

| Endpoint | Change |
|---|---|
| `POST /generate` | Delegates to `VersionService.create_version(operation="generate")` |
| `POST /refine` | Delegates to `VersionService.create_version(operation="refine")` |
| `POST /debug` | Delegates to `VersionService.create_version(operation="debug")` |
| `GET /sessions/{id}/history` | Returns full version timeline from `version_repo.get_history()` |
| `POST /sessions/{id}/rollback` | Calls `VersionService.rollback_to()` — metadata-only update |
| `GET /sessions/{id}/search?q=...` | Calls `ChromaService.semantic_search()` |

All existing request/response models are unchanged.

---

## Duplicate Version Detection

When a new notebook is generated:
1. Compute SHA256 of the new `.ipynb` file
2. Compare to `active_version.checksum`
3. If equal → return existing version with `status="unchanged"`, no new version created
4. If different → proceed with version creation

---

## Proposed Changes

---

### Database Layer

#### [NEW] [database.py](file:///Users/abhijith/Documents/EvoAcademy-llm-agents/app/db/database.py)
SQLAlchemy engine and session factory. SQLite by default, configurable via `DATABASE_URL` env var for PostgreSQL.

#### [NEW] [models.py](file:///Users/abhijith/Documents/EvoAcademy-llm-agents/app/db/models.py)
ORM models for Users, Sessions, Notebooks, NotebookVersions, VersionOperations.

#### [NEW] [notebook_repo.py](file:///Users/abhijith/Documents/EvoAcademy-llm-agents/app/db/repositories/notebook_repo.py)
CRUD for Notebooks table; manages active_version_id pointer.

#### [NEW] [version_repo.py](file:///Users/abhijith/Documents/EvoAcademy-llm-agents/app/db/repositories/version_repo.py)
CRUD for NotebookVersions; history queries, version lookup.

---

### Service Layer

#### [NEW] [storage_service.py](file:///Users/abhijith/Documents/EvoAcademy-llm-agents/app/services/storage_service.py)
File I/O for `.ipynb` files; returns (path, checksum) on save.

#### [NEW] [chroma_service.py](file:///Users/abhijith/Documents/EvoAcademy-llm-agents/app/services/chroma_service.py)
ChromaDB collection management; lightweight text embedding of summaries and prompts only.

#### [NEW] [memory_service.py](file:///Users/abhijith/Documents/EvoAcademy-llm-agents/app/services/memory_service.py)
Mem0 cloud client wrapper; retrieves/updates user preferences.

#### [NEW] [version_service.py](file:///Users/abhijith/Documents/EvoAcademy-llm-agents/app/services/version_service.py)
Main orchestration service; coordinates all other services through the version lifecycle.

---

### API Routes

#### [NEW] [routes/generate.py](file:///Users/abhijith/Documents/EvoAcademy-llm-agents/app/api/routes/generate.py)
#### [NEW] [routes/refine.py](file:///Users/abhijith/Documents/EvoAcademy-llm-agents/app/api/routes/refine.py)
#### [NEW] [routes/debug.py](file:///Users/abhijith/Documents/EvoAcademy-llm-agents/app/api/routes/debug.py)
#### [NEW] [routes/history.py](file:///Users/abhijith/Documents/EvoAcademy-llm-agents/app/api/routes/history.py)

#### [MODIFY] [main.py](file:///Users/abhijith/Documents/EvoAcademy-llm-agents/app/main.py)
Mount all APIRouters; remove inline business logic.

---

### Deprecated / Replaced

#### [DELETE] `app/core/version_store.py` — replaced by `app/db/repositories/version_repo.py` + `app/services/version_service.py`
#### [DELETE] `app/core/chroma_store.py` — replaced by `app/services/chroma_service.py`
#### [MODIFY] `app/core/memory.py` — simplified to delegate to `memory_service.py`

---

## New Dependencies

```
sqlalchemy>=2.0
alembic               # DB migrations
nbformat              # .ipynb file serialization
```

---

## Open Questions

> [!IMPORTANT]
> 1. **Summary generation**: Should the change summary ("Added tournament selection") be generated by the LLM as part of each modify/refine call? Or generated post-hoc from the diff? We recommend having the LLM generate a one-liner summary alongside the code changes.
> 2. **User identity**: The current system uses `session_id` as `user_id`. Should we add explicit multi-user support (separate Users table) or keep one user per session for now?
> 3. **Notebook format**: Should `.ipynb` files be full Jupyter Notebook format (with `nbformat`) including cell types/outputs, or a simplified JSON of the 12 DEAP cells only?

---

## Verification Plan

### Automated Tests
```bash
python -m unittest app/tests/test_version_control.py
```
Tests will be updated to cover:
- `StorageService`: save + load + checksum verification
- `VersionService`: create version, duplicate detection, rollback metadata-only
- `ChromaService`: index + semantic search round-trip
- `MemoryService`: preference injection into prompts

### Manual Verification
After restarting the server:
1. `POST /generate` → check `storage/notebooks/` for `version_1.ipynb`
2. `GET /sessions/{id}/history` → verify metadata timeline
3. `POST /refine` → confirm new `.ipynb` file created, version incremented
4. `GET /sessions/{id}/search?q=crossover probability` → verify semantic search returns relevant version
5. `POST /sessions/{id}/rollback` → confirm only `active_version_id` changes, no file created
