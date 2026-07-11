# EvoAcademy API Documentation

Welcome to the EvoAcademy backend API documentation. The API serves as the backbone for generating, refining, debugging, and managing the version history of DEAP Evolutionary Algorithm (EA) Jupyter Notebooks.

**Base URL**: `http://localhost:8000` (or your deployed server address)

---

## 1. System Health

### GET `/health`
Check if the API is running and accessible.

**Request Body**: None

**Response**:
```json
{
  "status": "ok",
  "version": "2.0.0",
  "features": ["version_history", "semantic_search", "rollback", "user_preferences"]
}
```
**How to Use**: Call this endpoint when the frontend application first loads to ensure the backend server is reachable before allowing the user to interact with the platform.

---

## 2. Notebook Generation & Refinement

### POST `/generate`
Generate a brand-new evolutionary algorithm notebook from a natural language prompt. This creates `version_1.ipynb` and clears any previous history for the provided session.

**Request Body** (`application/json`):
```json
{
  "session_id": "string (Unique ID for the learning session)",
  "prompt": "string (The student's raw prompt for the EA problem)"
}
```

**Response**:
```json
{
  "status": "string",
  "target_problem": "string",
  "cells": {
    "imports": "...",
    "config": "...",
    "creator": "...",
    "evaluation": "...",
    "crossover": "...",
    "mutation": "...",
    "selection": "...",
    "initialization": "...",
    "toolbox": "...",
    "main_algorithm": "...",
    "stats": "...",
    "visualization": "..."
  },
  "compiled_script": "string (The concatenated python script of all cells)",
  "version_number": 1,
  "version_id": "string (UUID for the generated version)"
}
```
**How to Use**: Use this when a student submits their very first problem prompt (e.g., "Solve the Traveling Salesperson Problem"). The returned `cells` dictionary contains the code for the 12 structural DEAP blocks, which you should map directly to the frontend code editor UI.

### POST `/refine`
Refine an existing notebook based on a follow-up question or modification request. This securely creates a new immutable version without overwriting previous history.

**Request Body** (`application/json`):
```json
{
  "session_id": "string",
  "user_prompt": "string (The student's question or modification request)",
  "current_cells": {
    "imports": "...",
    "config": "..."
  } // Must include the current string state of the 12 DEAP cells
}
```

**Response**:
```json
{
  "status": "string",
  "cells": { 
     // The newly updated 12 DEAP cells
  },
  "cells_modified": ["string (e.g., 'crossover', 'toolbox')"],
  "tutor_explanation": "string (Markdown explanation from the AI tutor)",
  "version_number": 2,
  "version_id": "string"
}
```
**How to Use**: Trigger this endpoint when the user asks a follow-up question in the chat, requests a code change, or asks for theoretical explanations. Display the `tutor_explanation` in the chat UI as a markdown message. Use the `cells_modified` array to visually highlight which code blocks the AI updated in the editor.

### POST `/debug`
Auto-fix a runtime error (traceback) thrown by the Jupyter kernel. Creates a new immutable version containing the fixed code.

**Request Body** (`application/json`):
```json
{
  "session_id": "string",
  "traceback_msg": "string (The raw runtime error thrown by the kernel execution)",
  "current_cells": { 
     // Current state of the 12 DEAP cells 
  } 
}
```

**Response**:
```json
{
  "status": "string",
  "cells": { ... },
  "cells_modified": ["string"],
  "tutor_explanation": "string (Explanation of what caused the bug and how it was fixed)",
  "version_number": 3,
  "version_id": "string"
}
```
**How to Use**: If the frontend attempts to execute the Python code and the compiler/kernel throws a traceback error, automatically capture that traceback string and send it to this endpoint. The AI will diagnose the failure and return patched cells.

---

## 3. Version History & Time Travel

### GET `/sessions/{session_id}/history`
Returns the complete, chronological version timeline for a specific session.

**Path Parameters**:
- `session_id`: Unique identifier for the active session.

**Response**:
```json
[
  {
    "version_number": 1,
    "operation_type": "generate",
    "prompt": "Solve TSP",
    "summary": "Initial generation",
    "is_active": false,
    "file_path": "storage/notebooks/session_id/version_1.ipynb",
    "checksum": "hash_string",
    "cells_modified": [],
    "created_at": "timestamp"
  },
  {
     // ... version 2 ...
  }
]
```
**How to Use**: Fetch this endpoint on load to display a "History Timeline" or "Commit Tree" in the UI. Use the `is_active` boolean to visually indicate which version the user is currently viewing/editing.

### POST `/sessions/{session_id}/rollback`
Rolls back the active state to a previous version. This is a pure metadata operation (it just moves the active pointer) and does not permanently delete any forward history.

**Path Parameters**:
- `session_id`: Unique identifier for the active session.

**Request Body** (`application/json`):
```json
{
  "version_number": 1 // The integer version number to revert back to
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Rolled back to version 1",
  "active_version": {
      "version_number": 1,
      "cells": { 
         // The historical code state of the rolled-back version 
      } 
  }
}
```
**How to Use**: Trigger this when a user clicks a "Revert to this version" button in your History Timeline. Immediately update the UI's code editor with the `cells` dictionary returned in the response object.

### GET `/sessions/{session_id}/search`
Perform a natural language semantic search over the notebook version history using ChromaDB vector embeddings.

**Path Parameters**:
- `session_id`: Unique identifier for the active session.

**Query Parameters**:
- `q`: string (Natural language query, e.g. "version where tournament selection was added")
- `n`: int (Max number of results to return, default is 5)

**Response**:
```json
[
  {
    "version_number": 2,
    "summary": "Added tournament selection logic to toolbox.",
    "similarity_score": 0.89
    // ... other standard history metadata
  }
]
```
**How to Use**: Provide a search bar component in the history/timeline panel. Users can type questions/queries to instantly filter past changes without having to manually read every commit summary.
