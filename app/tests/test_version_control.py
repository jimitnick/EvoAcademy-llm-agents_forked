import unittest
from unittest.mock import AsyncMock, patch
import json
import os
import shutil
from fastapi.testclient import TestClient

from app.main import app, validate_cells, compile_cells
from app.core import version_store

class TestVersionControl(unittest.TestCase):
    def setUp(self):
        # Configure a test database file
        self.test_db_path = os.path.join(os.path.dirname(__file__), "test_version_history.db")
        version_store.DB_PATH = self.test_db_path
        # Force re-initialization of db in the test path
        version_store.init_db()
        self.client = TestClient(app)
        self.session_id = "test_session_123"

    def tearDown(self):
        # Clean up the test database file
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    def test_database_operations(self):
        """Test basic SQLite version store API operations directly."""
        cells = {"imports": "import deap", "config": "rate = 0.5"}
        
        # Test saving a version
        v1 = version_store.save_version(
            session_id=self.session_id,
            user_intent="Initial setup",
            cells=cells,
            compiled_script="script_v1",
            status="working"
        )
        self.assertEqual(v1["version_number"], 1)
        self.assertEqual(v1["status"], "working")
        self.assertEqual(v1["cells"], cells)

        # Test next version number increment
        next_ver = version_store.get_next_version_number(self.session_id)
        self.assertEqual(next_ver, 2)

        # Test latest working version retrieval
        latest = version_store.get_latest_working_version(self.session_id)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["version_number"], 1)

        # Test saving a second version (failed)
        cells_v2 = {"imports": "import deap", "config": "rate = 0.6 ("}
        v2 = version_store.save_version(
            session_id=self.session_id,
            user_intent="Modify config with error",
            cells=cells_v2,
            compiled_script="script_v2",
            status="failed",
            error_message="SyntaxError"
        )
        self.assertEqual(v2["version_number"], 2)
        self.assertEqual(v2["status"], "failed")

        # Latest working version should still be version 1
        latest = version_store.get_latest_working_version(self.session_id)
        self.assertEqual(latest["version_number"], 1)

        # Test rollback to version 1
        v3 = version_store.rollback_to_version(self.session_id, 1)
        self.assertEqual(v3["version_number"], 3)
        self.assertEqual(v3["status"], "working")
        self.assertEqual(v3["cells"], cells)

        # Session history should contain 3 entries
        history = version_store.get_session_history(self.session_id)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["version_number"], 1)
        self.assertEqual(history[1]["version_number"], 2)
        self.assertEqual(history[2]["version_number"], 3)

    @patch("app.main.generate_graph.ainvoke", new_callable=AsyncMock)
    def test_generate_endpoint(self, mock_generate_ainvoke):
        """Test that /generate deletes old session data and stores Version 1."""
        # 1. Insert dummy version first to check deletion
        version_store.save_version(
            session_id=self.session_id,
            user_intent="Dummy version to delete",
            cells={"imports": "import sys"},
            compiled_script="script",
            status="working"
        )
        
        # 2. Mock graph invocation
        mock_generate_ainvoke.return_value = {
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
        history = version_store.get_session_history(self.session_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["version_number"], 1)
        self.assertEqual(history[0]["user_intent"], "Initial generation: Solve OneMax")
        self.assertEqual(history[0]["status"], "working")

    @patch("app.main.refine_graph.ainvoke", new_callable=AsyncMock)
    def test_refine_endpoint_success(self, mock_refine_ainvoke):
        """Test successful /refine saves a new working version."""
        # 1. Save Version 1 baseline
        version_store.save_version(
            session_id=self.session_id,
            user_intent="Baseline",
            cells={"imports": "import deap", "config": "rate = 0.5"},
            compiled_script="script_v1",
            status="working"
        )

        # 2. Mock success response
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

        # 4. Check DB: Version 2 should be 'working'
        history = version_store.get_session_history(self.session_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["version_number"], 2)
        self.assertEqual(history[1]["status"], "working")
        self.assertEqual(history[1]["user_intent"], "Increase rate to 0.7")

    @patch("app.main.refine_graph.ainvoke", new_callable=AsyncMock)
    def test_refine_endpoint_syntax_error_rollback(self, mock_refine_ainvoke):
        """Test that /refine with syntax error automatically rolls back to Version 1."""
        # 1. Save Version 1 baseline
        version_store.save_version(
            session_id=self.session_id,
            user_intent="Baseline",
            cells={"imports": "import deap", "config": "rate = 0.5"},
            compiled_script="script_v1",
            status="working"
        )

        # 2. Mock syntax error response
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
        # Should be reverted
        self.assertEqual(data["status"], "reverted")
        self.assertEqual(data["cells"]["config"], "rate = 0.5") # Reverted cells returned
        self.assertTrue("rolled back" in data["tutor_explanation"])

        # 4. Check DB: Should have Version 2 (failed) and Version 3 (revert working)
        history = version_store.get_session_history(self.session_id)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[1]["version_number"], 2)
        self.assertEqual(history[1]["status"], "failed")
        self.assertEqual(history[2]["version_number"], 3)
        self.assertEqual(history[2]["status"], "working")
        self.assertTrue("Auto-rollback" in history[2]["user_intent"])

    @patch("app.main.refine_graph.ainvoke", new_callable=AsyncMock)
    def test_refine_endpoint_pipeline_failure_rollback(self, mock_refine_ainvoke):
        """Test that /refine with pipeline exception automatically rolls back to Version 1."""
        # 1. Save Version 1 baseline
        version_store.save_version(
            session_id=self.session_id,
            user_intent="Baseline",
            cells={"imports": "import deap", "config": "rate = 0.5"},
            compiled_script="script_v1",
            status="working"
        )

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
        self.assertTrue("Pipeline error" in data["tutor_explanation"])

        # 4. Check DB: Version 2 is 'failed'
        history = version_store.get_session_history(self.session_id)
        self.assertEqual(history[1]["version_number"], 2)
        self.assertEqual(history[1]["status"], "failed")
        self.assertEqual(history[1]["error_message"], "LLM connection timed out")

    @patch("app.main.refine_graph.ainvoke", new_callable=AsyncMock)
    def test_debug_endpoint_success(self, mock_refine_ainvoke):
        """Test successful /debug saves a new working version."""
        version_store.save_version(
            session_id=self.session_id,
            user_intent="Baseline",
            cells={"imports": "import deap", "config": "rate = 0.5"},
            compiled_script="script_v1",
            status="working"
        )

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

        history = version_store.get_session_history(self.session_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["version_number"], 2)
        self.assertEqual(history[1]["status"], "working")
        self.assertTrue("Debug traceback" in history[1]["user_intent"])

    def test_manual_rollback_endpoint(self):
        """Test manual rollback endpoint."""
        version_store.save_version(
            session_id=self.session_id,
            user_intent="V1",
            cells={"imports": "import deap", "config": "rate = 0.5"},
            compiled_script="script_v1",
            status="working"
        )
        version_store.save_version(
            session_id=self.session_id,
            user_intent="V2",
            cells={"imports": "import deap", "config": "rate = 0.8"},
            compiled_script="script_v2",
            status="working"
        )

        # Call rollback to Version 1
        response = self.client.post(
            f"/sessions/{self.session_id}/rollback",
            json={"version_number": 1}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["cells"]["config"], "rate = 0.5")

        # Database should have Version 3 which is a rollback to 1
        history = version_store.get_session_history(self.session_id)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[2]["version_number"], 3)
        self.assertEqual(history[2]["status"], "working")
        self.assertEqual(history[2]["user_intent"], "Manual rollback to version 1")

    def test_get_history_endpoint(self):
        """Test fetching session history via GET endpoint."""
        version_store.save_version(
            session_id=self.session_id,
            user_intent="V1",
            cells={"imports": "import deap"},
            compiled_script="",
            status="working"
        )
        
        response = self.client.get(f"/sessions/{self.session_id}/history")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["session_id"], self.session_id)
        self.assertEqual(len(data["history"]), 1)
        self.assertEqual(data["history"][0]["user_intent"], "V1")

if __name__ == "__main__":
    unittest.main()
