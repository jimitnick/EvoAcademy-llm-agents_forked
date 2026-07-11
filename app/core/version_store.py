import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from app.core import chroma_store

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "version_history.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database and creates the versions table if it doesn't exist."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notebook_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                user_intent TEXT NOT NULL,
                cells TEXT NOT NULL, -- JSON serialized dictionary of DEAP cells
                compiled_script TEXT,
                status TEXT NOT NULL, -- 'working', 'failed', 'reverted'
                error_message TEXT,
                created_at TEXT NOT NULL
            )
        """)
        # Create indexes for fast lookup
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session_version ON notebook_versions(session_id, version_number)")
        conn.commit()

# Automatically initialize database when the module is imported
init_db()

def get_next_version_number(session_id: str) -> int:
    """Finds the next version number for a given session."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT MAX(version_number) as max_val FROM notebook_versions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if row and row["max_val"] is not None:
            return row["max_val"] + 1
        return 1

def save_version(
    session_id: str,
    user_intent: str,
    cells: Dict[str, str],
    compiled_script: str,
    status: str,
    error_message: Optional[str] = None
) -> dict:
    """Saves a new notebook state as a new version in SQLite and ChromaDB."""
    version_number = get_next_version_number(session_id)
    created_at = datetime.utcnow().isoformat()
    cells_json = json.dumps(cells)
    
    # 1. Save structured record to SQLite (used for exact rollback and history queries)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO notebook_versions (session_id, version_number, user_intent, cells, compiled_script, status, error_message, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, version_number, user_intent, cells_json, compiled_script, status, error_message, created_at)
        )
        conn.commit()
        last_id = cursor.lastrowid

    # 2. Save to ChromaDB (vector store) for semantic search and dashboard visibility
    chroma_doc_id = f"{session_id}__v{version_number}"
    chroma_store.store_version(
        doc_id=chroma_doc_id,
        session_id=session_id,
        version_number=version_number,
        user_intent=user_intent,
        cells=cells,
        compiled_script=compiled_script,
        status=status,
        created_at=created_at,
        error_message=error_message
    )
    # Also store the user intent in the dedicated intents collection
    chroma_store.store_user_intent(
        doc_id=chroma_doc_id,
        session_id=session_id,
        version_number=version_number,
        user_intent=user_intent,
        cells_modified=list(cells.keys()),
        created_at=created_at
    )

    return {
        "id": last_id,
        "session_id": session_id,
        "version_number": version_number,
        "user_intent": user_intent,
        "cells": cells,
        "compiled_script": compiled_script,
        "status": status,
        "error_message": error_message,
        "created_at": created_at
    }

def get_latest_working_version(session_id: str) -> Optional[dict]:
    """Retrieves the last version that was marked as 'working' for a session."""
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM notebook_versions
            WHERE session_id = ? AND status = 'working'
            ORDER BY version_number DESC LIMIT 1
            """,
            (session_id,)
        ).fetchone()
        
        if row:
            res = dict(row)
            res["cells"] = json.loads(res["cells"])
            return res
        return None

def get_version(session_id: str, version_number: int) -> Optional[dict]:
    """Retrieves a specific version for a session."""
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM notebook_versions
            WHERE session_id = ? AND version_number = ?
            """,
            (session_id, version_number)
        ).fetchone()
        
        if row:
            res = dict(row)
            res["cells"] = json.loads(res["cells"])
            return res
        return None

def get_session_history(session_id: str) -> List[dict]:
    """Lists all versions and intents for a session, ordered by version_number."""
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, session_id, version_number, user_intent, status, error_message, created_at
            FROM notebook_versions
            WHERE session_id = ?
            ORDER BY version_number ASC
            """,
            (session_id,)
        ).fetchall()
        return [dict(row) for row in rows]

def rollback_to_version(session_id: str, version_number: int) -> Optional[dict]:
    """
    Rolls back to a specific target version.
    Appends a new version entry reflecting this rollback, preserving history.
    """
    target = get_version(session_id, version_number)
    if not target:
        return None
        
    return save_version(
        session_id=session_id,
        user_intent=f"Manual rollback to version {version_number}",
        cells=target["cells"],
        compiled_script=target["compiled_script"],
        status="working"
    )
