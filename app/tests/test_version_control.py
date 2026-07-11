import unittest
from unittest.mock import AsyncMock, patch
import os
import shutil
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.db.database import Base, get_db
from app.db.models import Notebook, NotebookVersion
from app.services.version_service import VersionService
from app.services.storage_service import STORAGE_ROOT

# Use file-based SQLite database for testing so all connections share the same tables/data
DB_FILE = "./test_version_history.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_FILE}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db


class TestVersionControl(unittest.TestCase):
    def setUp(self):
        # Create all tables in the test database
        Base.metadata.create_all(bind=engine)
        self.client = TestClient(app)
        self.session_id = "test_session_123"
        self.db = TestingSessionLocal()

    def tearDown(self):
        # Close and clear database
        self.db.close()
        engine.dispose()  # Releases the file locks on test_version_history.db
        
        if os.path.exists(DB_FILE):
            try:
                os.remove(DB_FILE)
            except Exception as e:
                print(f"Warning: Failed to remove test DB file: {e}")

        # Clean up files created in STORAGE_ROOT for the test session
        test_session_dir = os.path.join(STORAGE_ROOT, f"session_{self.session_id}")
        if os.path.exists(test_session_dir):
            try:
                shutil.rmtree(test_session_dir)
            except Exception:
                pass

    def get_history(self) -> list:
        """Helper to fetch history from VersionService."""
        svc = VersionService(self.db)
        history_dict = svc.get_history(self.session_id)
        return history_dict.get("versions", [])

    @patch("app.services.version_service.generate_graph.ainvoke", new_callable=AsyncMock)
    def test_generate_endpoint(self, mock_generate_ainvoke):
        """Test that /generate deletes old session data and stores Version 1."""
        # 1. Insert dummy version/notebook first to check deletion
        svc = VersionService(self.db)
        from app.db.repositories.notebook_repo import NotebookRepository
        from app.db.repositories.version_repo import VersionRepository
        notebook_repo = NotebookRepository(self.db)
        version_repo = VersionRepository(self.db)
        
        notebook = notebook_repo.get_or_create_notebook(self.session_id)
        version_repo.create_version(
            notebook_id=notebook.notebook_id,
            version_number=1,
            operation_type="generate",
            file_path="dummy_path.ipynb",
            checksum="dummy_checksum",
            prompt="Dummy",
            summary="Dummy version to delete"
        )
        self.db.commit()
        
        # Verify dummy exists
        history = self.get_history()
        self.assertEqual(len(history), 1)

        # 2. Mock graph invocation
        mock_generate_ainvoke.return_value = {
            "is_valid_ea_prompt": True,
            "target_problem": "OneMax",
            "notebook_cells": {"imports": "import deap", "config": "rate = 0.5"},
            "compiled_script": "compiled_onemax"
        }

        # 3. Call endpoint
        response = self.client.post(
            "/generate",
            json={"session_id": self.session_id, "prompt": "Solve OneMax"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["target_problem"], "OneMax")
        self.assertEqual(data["cells"]["imports"], "import deap")

        # 4. Check DB status: old history deleted, only new Version 1 exists
        self.db.expire_all()
        history = self.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["version_number"], 1)
        self.assertEqual(history[0]["prompt"], "Solve OneMax")
        self.assertEqual(history[0]["operation_type"], "generate")

    @patch("app.services.version_service.generate_graph.ainvoke", new_callable=AsyncMock)
    def test_generate_endpoint_rejected(self, mock_generate_ainvoke):
        """Test that /generate returns status='rejected' when rejected by the gatekeeper."""
        # 1. Mock graph invocation with rejection
        mock_generate_ainvoke.return_value = {
            "is_valid_ea_prompt": False,
            "rejection_reason": "This platform is only for Evolutionary algorithm problems."
        }

        # 2. Call endpoint
        response = self.client.post(
            "/generate",
            json={"session_id": self.session_id, "prompt": "Build a React todo app"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "rejected")
        self.assertEqual(data["target_problem"], "Invalid Domain")
        self.assertEqual(data["cells"], {})
        self.assertEqual(data["compiled_script"], "# ERROR:This platform is only for Evolutionary algorithm problems.")
        self.assertEqual(data["version_number"], 0)
        self.assertEqual(data["version_id"], "")

        # 3. Verify no versions created in DB
        self.db.expire_all()
        history = self.get_history()
        self.assertEqual(len(history), 0)

    @patch("app.services.version_service.refine_graph.ainvoke", new_callable=AsyncMock)
    def test_refine_endpoint_success(self, mock_refine_ainvoke):
        """Test successful /refine saves a new working version."""
        # 1. Create baseline version 1 using VersionService directly
        svc = VersionService(self.db)
        notebook = svc.notebook_repo.get_or_create_notebook(self.session_id)
        v1 = svc.version_repo.create_version(
            notebook_id=notebook.notebook_id,
            version_number=1,
            operation_type="generate",
            file_path="dummy_path.ipynb",
            checksum="dummy_checksum",
            prompt="Solve OneMax",
            summary="Baseline"
        )
        svc.notebook_repo.set_active_version(notebook.notebook_id, v1.version_id)
        self.db.commit()

        # 2. Mock success response from refinement LLM
        mock_refine_ainvoke.return_value = {
            "notebook_cells": {"imports": "import deap", "config": "rate = 0.7"},
            "cells_to_modify": ["config"],
            "needs_understanding": False,
            "educational_response": "Rate updated successfully."
        }

        # 3. Call refine endpoint
        response = self.client.post(
            "/refine",
            json={
                "session_id": self.session_id,
                "user_prompt": "Increase rate to 0.7",
                "current_cells": {"imports": "import deap", "config": "rate = 0.5"}
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["cells"]["config"], "rate = 0.7")

        # 4. Check DB: Version 2 should exist and be active
        self.db.expire_all()
        history = self.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["version_number"], 2)
        self.assertEqual(history[1]["operation_type"], "refine")

    @patch("app.services.version_service.storage_service.load_notebook")
    @patch("app.services.version_service.refine_graph.ainvoke", new_callable=AsyncMock)
    def test_refine_endpoint_syntax_error(self, mock_refine_ainvoke, mock_load_notebook):
        """Test that /refine with syntax error automatically reverts and returns 200."""
        # 1. Create baseline version 1
        svc = VersionService(self.db)
        notebook = svc.notebook_repo.get_or_create_notebook(self.session_id)
        v1 = svc.version_repo.create_version(
            notebook_id=notebook.notebook_id,
            version_number=1,
            operation_type="generate",
            file_path="dummy_path.ipynb",
            checksum="dummy_checksum",
            prompt="Solve OneMax",
            summary="Baseline"
        )
        svc.notebook_repo.set_active_version(notebook.notebook_id, v1.version_id)
        self.db.commit()

        # Mock disk load to return the active cells
        mock_load_notebook.return_value = {"imports": "import deap", "config": "rate = 0.5"}

        # 2. Mock syntax error response (invalid Python syntax)
        mock_refine_ainvoke.return_value = {
            "notebook_cells": {"imports": "import deap", "config": "rate = 0.7 ("}, # invalid
            "cells_to_modify": ["config"],
            "needs_understanding": False,
            "educational_response": "Trying to update..."
        }

        # 3. Call refine endpoint
        response = self.client.post(
            "/refine",
            json={
                "session_id": self.session_id,
                "user_prompt": "Increase rate with invalid syntax",
                "current_cells": {"imports": "import deap", "config": "rate = 0.5"}
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "reverted")
        self.assertEqual(data["cells"]["config"], "rate = 0.5") # Reverted
        self.assertEqual(data["version_number"], 1)
        self.assertEqual(data["version_id"], v1.version_id)
        
        # 4. Check DB: Should still only have Version 1
        self.db.expire_all()
        history = self.get_history()
        self.assertEqual(len(history), 1)

    @patch("app.services.version_service.storage_service.load_notebook")
    @patch("app.services.version_service.refine_graph.ainvoke", new_callable=AsyncMock)
    def test_refine_endpoint_pipeline_failure(self, mock_refine_ainvoke, mock_load_notebook):
        """Test that /refine with pipeline exception automatically reverts and returns 200."""
        # 1. Create baseline version 1
        svc = VersionService(self.db)
        notebook = svc.notebook_repo.get_or_create_notebook(self.session_id)
        v1 = svc.version_repo.create_version(
            notebook_id=notebook.notebook_id,
            version_number=1,
            operation_type="generate",
            file_path="dummy_path.ipynb",
            checksum="dummy_checksum",
            prompt="Solve OneMax",
            summary="Baseline"
        )
        svc.notebook_repo.set_active_version(notebook.notebook_id, v1.version_id)
        self.db.commit()

        mock_load_notebook.return_value = {"imports": "import deap", "config": "rate = 0.5"}

        # 2. Mock pipeline throwing exception
        mock_refine_ainvoke.side_effect = Exception("LLM connection timed out")

        # 3. Call refine endpoint
        response = self.client.post(
            "/refine",
            json={
                "session_id": self.session_id,
                "user_prompt": "Update rate",
                "current_cells": {"imports": "import deap", "config": "rate = 0.5"}
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "reverted")
        self.assertEqual(data["cells"]["config"], "rate = 0.5")

    @patch("app.services.version_service.refine_graph.ainvoke", new_callable=AsyncMock)
    def test_debug_endpoint_success(self, mock_refine_ainvoke):
        """Test successful /debug saves a new working version."""
        # 1. Create baseline version 1
        svc = VersionService(self.db)
        notebook = svc.notebook_repo.get_or_create_notebook(self.session_id)
        v1 = svc.version_repo.create_version(
            notebook_id=notebook.notebook_id,
            version_number=1,
            operation_type="generate",
            file_path="dummy_path.ipynb",
            checksum="dummy_checksum",
            prompt="Solve OneMax",
            summary="Baseline"
        )
        svc.notebook_repo.set_active_version(notebook.notebook_id, v1.version_id)
        self.db.commit()

        # 2. Mock debug success response
        mock_refine_ainvoke.return_value = {
            "notebook_cells": {"imports": "import deap", "config": "rate = 0.6"},
            "cells_to_modify": ["config"]
        }

        response = self.client.post(
            "/debug",
            json={
                "session_id": self.session_id,
                "traceback_msg": "NameError: name 'rate' is not defined",
                "current_cells": {"imports": "import deap", "config": "rate = 0.5"}
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")

        self.db.expire_all()
        history = self.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["version_number"], 2)
        self.assertEqual(history[1]["operation_type"], "debug")

    @patch("app.services.version_service.storage_service.load_notebook")
    @patch("app.services.version_service.refine_graph.ainvoke", new_callable=AsyncMock)
    def test_debug_endpoint_failure_revert(self, mock_refine_ainvoke, mock_load_notebook):
        """Test that /debug with validation failure automatically reverts and returns 200."""
        # 1. Create baseline version 1
        svc = VersionService(self.db)
        notebook = svc.notebook_repo.get_or_create_notebook(self.session_id)
        v1 = svc.version_repo.create_version(
            notebook_id=notebook.notebook_id,
            version_number=1,
            operation_type="generate",
            file_path="dummy_path.ipynb",
            checksum="dummy_checksum",
            prompt="Solve OneMax",
            summary="Baseline"
        )
        svc.notebook_repo.set_active_version(notebook.notebook_id, v1.version_id)
        self.db.commit()

        mock_load_notebook.return_value = {"imports": "import deap", "config": "rate = 0.5"}

        # 2. Mock debug validation failure (invalid Python syntax)
        mock_refine_ainvoke.return_value = {
            "notebook_cells": {"imports": "import deap", "config": "rate = 0.6 ("}, # invalid
            "cells_to_modify": ["config"]
        }

        response = self.client.post(
            "/debug",
            json={
                "session_id": self.session_id,
                "traceback_msg": "NameError: name 'rate' is not defined",
                "current_cells": {"imports": "import deap", "config": "rate = 0.5"}
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "reverted")
        self.assertEqual(data["cells"]["config"], "rate = 0.5")
        self.assertEqual(data["version_number"], 1)

    def test_manual_rollback_endpoint(self):
        """Test manual rollback endpoint."""
        svc = VersionService(self.db)
        notebook = svc.notebook_repo.get_or_create_notebook(self.session_id)
        
        with patch("app.services.version_service.storage_service.load_notebook") as mock_load:
            mock_load.return_value = {"imports": "import deap", "config": "rate = 0.5"}
            
            v1 = svc.version_repo.create_version(
                notebook_id=notebook.notebook_id,
                version_number=1,
                operation_type="generate",
                file_path="dummy_v1.ipynb",
                checksum="checksum1",
                prompt="V1",
                summary="V1 summary"
            )
            v2 = svc.version_repo.create_version(
                notebook_id=notebook.notebook_id,
                version_number=2,
                operation_type="refine",
                file_path="dummy_v2.ipynb",
                checksum="checksum2",
                prompt="V2",
                summary="V2 summary"
            )
            svc.notebook_repo.set_active_version(notebook.notebook_id, v2.version_id)
            self.db.commit()

            # Call rollback to Version 1
            response = self.client.post(
                f"/sessions/{self.session_id}/rollback",
                json={"version_number": 1}
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["version_number"], 1)

            # Database should point to active version 1
            self.db.expire_all()
            notebook = svc.notebook_repo.get_notebook_by_session(self.session_id)
            self.assertEqual(notebook.active_version_id, v1.version_id)

    def test_get_history_endpoint(self):
        """Test fetching session history via GET endpoint."""
        svc = VersionService(self.db)
        notebook = svc.notebook_repo.get_or_create_notebook(self.session_id)
        
        v1 = svc.version_repo.create_version(
            notebook_id=notebook.notebook_id,
            version_number=1,
            operation_type="generate",
            file_path="dummy.ipynb",
            checksum="hash",
            prompt="V1",
            summary="Initial version"
        )
        svc.notebook_repo.set_active_version(notebook.notebook_id, v1.version_id)
        self.db.commit()
        
        response = self.client.get(f"/sessions/{self.session_id}/history")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["session_id"], self.session_id)
        self.assertEqual(len(data["versions"]), 1)
        self.assertEqual(data["versions"][0]["prompt"], "V1")

    def test_health_check_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")

    @patch("app.services.version_service.chroma_service.semantic_search")
    def test_search_versions_endpoint(self, mock_chroma_search):
        """Test semantic search endpoint."""
        mock_chroma_search.return_value = [
            {
                "version_id": "v1-id",
                "version_number": 1,
                "summary": "Initial version",
                "operation_type": "generate",
                "cells_modified": ["all cells"],
                "relevance_score": 0.95
            }
        ]

        response = self.client.get(f"/sessions/{self.session_id}/search?q=initial")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["session_id"], self.session_id)
        self.assertEqual(data["query"], "initial")
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["version_number"], 1)


if __name__ == "__main__":
    unittest.main()
