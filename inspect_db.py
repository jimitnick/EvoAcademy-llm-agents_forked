"""
inspect_db.py — Verify all three storage layers of the version history system.

Usage:
  python inspect_db.py
"""
import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()

SQLITE_DB_PATH = "evo_academy.db"
CHROMA_DB_PATH = ".chroma_version_store"  # Semantic index (chroma_service.py)
STORAGE_DIR = "storage/notebooks"          # Immutable .ipynb files


def inspect_sqlite():
    print("=" * 60)
    print("SQLITE — Structured Metadata (evo_academy.db)")
    print("=" * 60)
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"  [!] No database at '{SQLITE_DB_PATH}'.")
        print("  Tip: Start the server and call POST /generate to initialize.")
        return

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Sessions
    cursor.execute("SELECT session_id, created_at, last_active_at FROM sessions")
    sessions = cursor.fetchall()
    print(f"\nSessions ({len(sessions)} total):")
    for s in sessions:
        print(f"  session_id={s[0]}  created={s[1]}  last_active={s[2]}")

    # Notebooks
    cursor.execute("SELECT notebook_id, session_id, active_version_id, updated_at FROM notebooks")
    notebooks = cursor.fetchall()
    print(f"\nNotebooks ({len(notebooks)} total):")
    for nb in notebooks:
        print(f"  notebook_id={nb[0][:8]}...  session={nb[1]}  active_version={str(nb[2])[:8]}...  updated={nb[3]}")

    # Versions
    cursor.execute("""
        SELECT version_id, notebook_id, version_number, operation_type, summary, file_path, checksum, chroma_indexed, created_at
        FROM notebook_versions
        ORDER BY created_at ASC
    """)
    versions = cursor.fetchall()
    print(f"\nNotebook Versions ({len(versions)} total):")
    for v in versions:
        print(f"  v{v[2]:>3} | op={v[3]:<8} | summary='{(v[4] or '')[:60]}' | chroma={bool(v[7])} | {v[8][:19]}")
        print(f"        file={v[5]}")

    # Audit log
    cursor.execute("SELECT op_id, version_id, action, created_at FROM version_operations ORDER BY created_at DESC LIMIT 10")
    ops = cursor.fetchall()
    print(f"\nVersion Operations (last {len(ops)}):")
    for op in ops:
        print(f"  {op[3][:19]} | {op[2]:<20} | version={str(op[1])[:8]}...")

    conn.close()


def inspect_chromadb():
    print("\n" + "=" * 60)
    print(f"CHROMADB — Semantic Index ({CHROMA_DB_PATH})")
    print("=" * 60)
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"  [!] No ChromaDB store at '{CHROMA_DB_PATH}'.")
        print("  Tip: Start the server and call POST /generate to create the first entry.")
        return

    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collections = client.list_collections()
        if not collections:
            print("  No collections found yet.")
            return
        for col in collections:
            c = client.get_collection(col.name)
            count = c.count()
            print(f"\n  Collection: '{col.name}'  ({count} documents)")
            if count > 0:
                peek = c.peek(limit=5)
                for doc, meta in zip(peek["documents"], peek["metadatas"]):
                    print(f"    v{meta.get('version_number')} [{meta.get('operation_type')}] {meta.get('session_id')[:12]}...")
                    print(f"    summary: {meta.get('summary', '')[:80]}")
                    print(f"    doc snippet: {str(doc)[:100]}")
    except ImportError:
        print("  ChromaDB not installed.")
    except Exception as e:
        print(f"  Error: {e}")


def inspect_storage():
    print("\n" + "=" * 60)
    print(f"FILE STORAGE — Immutable .ipynb files ({STORAGE_DIR})")
    print("=" * 60)
    if not os.path.exists(STORAGE_DIR):
        print(f"  [!] No storage directory at '{STORAGE_DIR}'.")
        return

    total_files = 0
    for session_dir in sorted(os.listdir(STORAGE_DIR)):
        full_path = os.path.join(STORAGE_DIR, session_dir)
        if os.path.isdir(full_path):
            files = sorted([f for f in os.listdir(full_path) if f.endswith(".ipynb")])
            total_files += len(files)
            print(f"\n  {session_dir}/  ({len(files)} versions)")
            for f in files:
                fp = os.path.join(full_path, f)
                size_kb = os.path.getsize(fp) / 1024
                print(f"    {f}  ({size_kb:.1f} KB)")
    print(f"\n  Total .ipynb files: {total_files}")


def inspect_mem0():
    print("\n" + "=" * 60)
    print("MEM0 CLOUD — User Preferences")
    print("=" * 60)
    key = os.getenv("MEM0_API_KEY")
    if not key:
        print("  [!] MEM0_API_KEY not set. Skipping.")
        return
    try:
        from mem0 import MemoryClient
        client = MemoryClient(api_key=key)
        memories = client.get_all()
        if isinstance(memories, dict):
            total = memories.get("count", 0)
            results = memories.get("results", [])
        else:
            total = len(memories)
            results = memories
        print(f"\n  Total preferences stored: {total}")
        for m in results[:10]:
            print(f"  [{m.get('user_id', '?')}] {m.get('memory', str(m))[:100]}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    inspect_sqlite()
    inspect_chromadb()
    inspect_storage()
    inspect_mem0()
    print("\n" + "=" * 60)
    print("Inspection complete.")
