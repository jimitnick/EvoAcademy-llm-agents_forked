import sqlite3
import os
import json

def inspect_sqlite():
    db_path = "version_history.db"
    print("=== INSPECTING SQLITE (Version History) ===")
    if not os.path.exists(db_path):
        print(f"No SQLite database found at '{db_path}'. Make sure you have run the app/tests first.")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, session_id, version_number, user_intent, status, created_at FROM notebook_versions")
        rows = cursor.fetchall()
        print(f"Total version records found: {len(rows)}")
        for row in rows:
            print(f"ID: {row[0]} | Session: {row[1]} | Version: {row[2]} | Intent: '{row[3]}' | Status: {row[4]} | Created: {row[5]}")
        conn.close()
    except Exception as e:
        print(f"Error reading SQLite: {e}")

def inspect_chromadb():
    print("\n=== INSPECTING CHROMADB (Mem0 Vector Store) ===")
    chroma_path = "./.mem0_chromadb"
    if not os.path.exists(chroma_path):
        print(f"No ChromaDB database folder found at '{chroma_path}'.")
        return

    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_path)
        collections = client.list_collections()
        print(f"Collections found: {[c.name for c in collections]}")
        for col in collections:
            collection = client.get_collection(col.name)
            count = collection.count()
            print(f"Collection '{col.name}' has {count} vector entries.")
            if count > 0:
                print("Peeking at first few entries:")
                peek_data = collection.peek(limit=5)
                for doc, meta in zip(peek_data['documents'], peek_data['metadatas']):
                    print(f"  - Document: '{doc}'")
                    print(f"    Metadata: {meta}")
    except ImportError:
        print("ChromaDB library not installed in the current environment.")
    except Exception as e:
        print(f"Error reading ChromaDB: {e}")

if __name__ == "__main__":
    inspect_sqlite()
    inspect_chromadb()
