"""
StorageService — Reads and writes notebook .ipynb files to disk.
Each notebook version is saved as an immutable file.
Rollback never touches files; only the active_version_id pointer changes.

Storage layout:
  storage/
    notebooks/
      session_{session_id}/
        version_1.ipynb
        version_2.ipynb
        ...
"""
import hashlib
import json
import logging
import os
from typing import Dict, Tuple

import nbformat
from nbformat.v4 import new_notebook, new_code_cell

logger = logging.getLogger(__name__)

# Root storage directory (relative to project root)
STORAGE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "storage", "notebooks"
)

# Ordered DEAP cell names — preserved when serializing to/from .ipynb
DEAP_CELLS = [
    "imports", "config", "creator", "evaluation", "crossover", "mutation",
    "selection", "initialization", "toolbox", "main_algorithm", "stats", "visualization"
]


class StorageService:
    """File I/O for .ipynb notebook files. Swap this class to add S3/MinIO support."""

    def save_notebook(
        self,
        session_id: str,
        version_number: int,
        cells: Dict[str, str]
    ) -> Tuple[str, str]:
        """
        Serializes cells dict → standard .ipynb, saves to disk.
        Returns (relative_file_path, sha256_checksum).
        File path is relative to project root for portability.
        """
        session_dir = os.path.join(STORAGE_ROOT, f"session_{session_id}")
        os.makedirs(session_dir, exist_ok=True)

        file_name = f"version_{version_number}.ipynb"
        abs_path = os.path.join(session_dir, file_name)

        nb = new_notebook()
        # Preserve DEAP cell order; include any extra cells at the end
        ordered_cells = []
        for cell_name in DEAP_CELLS:
            code = cells.get(cell_name, "")
            if code:
                code_cell = new_code_cell(source=code)
                code_cell.metadata["cell_name"] = cell_name
                ordered_cells.append(code_cell)
        # Add any cells not in DEAP_CELLS
        for cell_name, code in cells.items():
            if cell_name not in DEAP_CELLS and code:
                code_cell = new_code_cell(source=code)
                code_cell.metadata["cell_name"] = cell_name
                ordered_cells.append(code_cell)

        nb.cells = ordered_cells
        nb.metadata["session_id"] = session_id
        nb.metadata["version_number"] = version_number

        with open(abs_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        checksum = self._compute_checksum(abs_path)
        rel_path = os.path.relpath(abs_path, start=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        logger.info(f"[Storage] Saved {rel_path} (checksum={checksum[:12]}...)")
        
        # Keep the active working document up to date
        self.update_active_notebook(session_id, cells)

        return rel_path, checksum

    def update_active_notebook(self, session_id: str, cells: Dict[str, str]) -> str:
        """
        Updates the main 'active.ipynb' file for the session with the given cells.
        This file acts as the user's live working document (refreshable in Jupyter/editors).
        """
        session_dir = os.path.join(STORAGE_ROOT, f"session_{session_id}")
        os.makedirs(session_dir, exist_ok=True)
        abs_path = os.path.join(session_dir, "active.ipynb")

        nb = new_notebook()
        ordered_cells = []
        for cell_name in DEAP_CELLS:
            code = cells.get(cell_name, "")
            if code:
                code_cell = new_code_cell(source=code)
                code_cell.metadata["cell_name"] = cell_name
                ordered_cells.append(code_cell)
        for cell_name, code in cells.items():
            if cell_name not in DEAP_CELLS and code:
                code_cell = new_code_cell(source=code)
                code_cell.metadata["cell_name"] = cell_name
                ordered_cells.append(code_cell)

        nb.cells = ordered_cells
        nb.metadata["session_id"] = session_id

        with open(abs_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        rel_path = os.path.relpath(abs_path, start=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        logger.info(f"[Storage] Updated active working file: {rel_path}")
        return rel_path


    def load_notebook(self, file_path: str) -> Dict[str, str]:
        """
        Reads a .ipynb file and returns a cells dict {cell_name: source_code}.
        file_path can be relative (to project root) or absolute.
        """
        abs_path = self._resolve_path(file_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Notebook not found: {abs_path}")

        with open(abs_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        cells = {}
        for cell in nb.cells:
            cell_name = cell.metadata.get("cell_name")
            if cell_name:
                cells[cell_name] = cell.source
        return cells

    def checksum_matches(self, file_path: str, expected_checksum: str) -> bool:
        abs_path = self._resolve_path(file_path)
        if not os.path.exists(abs_path):
            return False
        return self._compute_checksum(abs_path) == expected_checksum

    def _compute_checksum(self, abs_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(abs_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _resolve_path(self, file_path: str) -> str:
        if os.path.isabs(file_path):
            return file_path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(project_root, file_path)
