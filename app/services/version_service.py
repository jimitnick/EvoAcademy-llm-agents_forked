"""
VersionService — The main orchestrator for the version history system.

Responsibilities:
  1. Load active notebook cells (returns NotebookCells)
  2. Inject user preferences from Mem0 into LLM prompt
  3. Run LLM workflow (generate/refine/debug)
  4. Validate generated cells (AST parse)
  5. Generate a one-line change summary via LLM
  6. Save file via StorageService (snapshot or delta.json depending on version)
  7. Skip if checksum matches current active (no-op / duplicate detection)
  8. Insert NotebookVersion row in DB (with is_snapshot / delta_size)
  9. Update Notebook.active_version_id
  10. Index summary in ChromaDB via ChromaService
  11. Update Mem0 preferences via MemoryService

Rollback is metadata-only: only active_version_id changes, no files touched.

Delta compression
-----------------
StorageService decides whether to write a full .ipynb or a .delta.json based
on the version number.  VersionService supplies the parent NotebookCells so
the diff can be computed.  Reconstruction is injected as a callable to avoid
circular imports between VersionService → StorageService → VersionService.
"""
from __future__ import annotations

import ast
import logging
import os
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session as DBSession

from app.db.models import Notebook, NotebookVersion
from app.db.repositories.notebook_repo import NotebookRepository
from app.db.repositories.version_repo import VersionRepository
from app.schemas.notebook import NotebookCells, DEAP_CELL_ORDER
from app.services.storage_service import StorageService
from app.services.chroma_service import ChromaService
from app.services.memory_service import MemoryService
from app.agents.workflows import generate_graph, refine_graph, DEAP_CELLS

logger = logging.getLogger(__name__)

storage_service = StorageService()
chroma_service = ChromaService()
memory_service = MemoryService()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_cells(cells: NotebookCells) -> Optional[str]:
    """Returns error string if any cell has a SyntaxError, else None."""
    for cell_name, code in cells.to_dict().items():
        if code and not code.startswith("# ERROR"):
            try:
                ast.parse(code)
            except SyntaxError as e:
                return f"SyntaxError in cell '{cell_name}': {str(e)}"
    return None


def _compile_cells(cells: NotebookCells) -> str:
    """Assembles ordered DEAP cells into a single script string."""
    blocks = []
    cells_dict = cells.to_dict()
    for cell_name in DEAP_CELL_ORDER:
        code = cells_dict.get(cell_name)
        if code:
            header = f"\n# {'='*40}\n# CELL: {cell_name.upper()}\n# {'='*40}\n"
            blocks.append(header + code)
        else:
            blocks.append(f"\n# ERROR: Agent failed to generate {cell_name} cell.\n")
    return "\n".join(blocks)


async def _generate_summary(prompt: str, cells_modified: List[str], operation_type: str) -> str:
    """
    Generates a concise one-line summary of the change using the LLM.
    Fallback to a template if the LLM fails.
    """
    try:
        from app.core.llm import architect_llm
        modified_str = ", ".join(cells_modified) if cells_modified else "all cells"
        msg = (
            f"Summarize this notebook modification in one short sentence (max 15 words). "
            f"Operation: {operation_type}. "
            f"User prompt: '{prompt}'. "
            f"Cells modified: {modified_str}. "
            f"Output ONLY the summary sentence, nothing else."
        )
        result = await architect_llm.ainvoke(msg)
        summary = result.content.strip().strip('"').strip("'")
        return summary[:200]
    except Exception as e:
        logger.warning(f"[VersionService] LLM summary generation failed: {e}")
        cells_str = ", ".join(cells_modified) if cells_modified else "notebook"
        return f"{operation_type.capitalize()}: {cells_str} — {prompt[:80]}"


# ---------------------------------------------------------------------------
# VersionService
# ---------------------------------------------------------------------------

class VersionService:
    def __init__(self, db: DBSession):
        self.db = db
        self.notebook_repo = NotebookRepository(db)
        self.version_repo = VersionRepository(db)

    # ------------------------------------------------------------------
    # /generate — Create a brand-new notebook, reset session history
    # ------------------------------------------------------------------
    async def generate(self, session_id: str, prompt: str) -> dict:
        """
        Runs the generate workflow, saves result as version 1.
        If session already has history, it is cleared (new problem).
        """
        preferences = memory_service.get_preferences(session_id)
        enriched_prompt = f"{prompt}\n\n{preferences}" if preferences else prompt

        result_state = await generate_graph.ainvoke({
            "user_prompt": enriched_prompt,
            "session_id": session_id,
        })

        # Check if prompt was rejected by gatekeeper
        if not result_state.get("is_valid_ea_prompt", True):
            reason = result_state.get("rejection_reason", "Prompt is not related to Evolutionary Algorithms.")
            return {
                "status": "rejected",
                "target_problem": "Invalid Domain",
                "cells": NotebookCells().to_dict(),
                "compiled_script": f"# ERROR:{reason}",
                "version_number": 0,
                "version_id": "",
            }

        generated_cells = result_state.get("notebook_cells", {})
        cells = NotebookCells.from_dict({k: generated_cells.get(k, "") for k in DEAP_CELLS})
        target_problem = result_state.get("target_problem", prompt)

        error = _validate_cells(cells)
        if error:
            raise ValueError(f"Generated cells failed validation: {error}")

        # Clear old history for this session (new problem = fresh start)
        self.notebook_repo.clear_session_notebooks(session_id)
        chroma_service.delete_session_versions(session_id)

        version, notebook = await self._create_and_save_version(
            session_id=session_id,
            prompt=prompt,
            cells=cells,
            operation_type="generate",
            cells_modified=list(cells.to_dict().keys()),
            extra_metadata={"target_problem": target_problem}
        )

        return {
            "status": "success",
            "target_problem": target_problem,
            "cells": cells.to_dict(),
            "compiled_script": _compile_cells(cells),
            "version_number": version.version_number,
            "version_id": version.version_id,
        }

    # ------------------------------------------------------------------
    # /refine — Modify existing notebook based on user request
    # ------------------------------------------------------------------
    async def refine(
        self,
        session_id: str,
        user_prompt: str,
        current_cells: Dict[str, str],
    ) -> dict:
        """
        Runs the refine workflow. Never overwrites previous version.
        Falls back to current active version on any failure.
        """
        db_cells = self.get_active_cells(session_id) or NotebookCells()
        merged_cells = db_cells.merge(NotebookCells.from_dict(current_cells))

        preferences = memory_service.get_preferences(session_id)
        enriched_prompt = f"{user_prompt}\n\n{preferences}" if preferences else user_prompt

        result_state = await refine_graph.ainvoke({
            "user_prompt": enriched_prompt,
            "session_id": session_id,
            "notebook_cells": merged_cells.to_dict(),
        })

        refined_cells = NotebookCells.from_dict(
            result_state.get("notebook_cells", merged_cells.to_dict())
        )
        cells_modified = result_state.get("cells_to_modify", [])
        explanation = (
            result_state.get("educational_response")
            or result_state.get("tutor_explanation")
            or ""
        )

        error = _validate_cells(refined_cells)
        if error:
            logger.warning(f"[VersionService] Refined cells failed validation: {error}. Keeping current version.")
            raise ValueError(f"Refine validation failed: {error}")

        version, notebook = await self._create_and_save_version(
            session_id=session_id,
            prompt=user_prompt,
            cells=refined_cells,
            operation_type="refine",
            cells_modified=cells_modified,
        )

        return {
            "status": "success",
            "cells": refined_cells.to_dict(),
            "cells_modified": cells_modified,
            "tutor_explanation": explanation,
            "version_number": version.version_number,
            "version_id": version.version_id,
        }

    # ------------------------------------------------------------------
    # /debug — Auto-fix a traceback in the active notebook
    # ------------------------------------------------------------------
    async def debug(
        self,
        session_id: str,
        traceback_msg: str,
        current_cells: Dict[str, str],
    ) -> dict:
        """
        Runs the refine workflow in debug mode (auto-fix the traceback).
        Same versioning lifecycle as refine.
        """
        db_cells = self.get_active_cells(session_id) or NotebookCells()
        merged_cells = db_cells.merge(NotebookCells.from_dict(current_cells))

        debug_prompt = f"DEBUG: Fix this runtime error:\n{traceback_msg}"
        preferences = memory_service.get_preferences(session_id)
        enriched_prompt = f"{debug_prompt}\n\n{preferences}" if preferences else debug_prompt

        result_state = await refine_graph.ainvoke({
            "user_prompt": enriched_prompt,
            "session_id": session_id,
            "notebook_cells": merged_cells.to_dict(),
            "is_traceback_error": True,
            "traceback_msg": traceback_msg,
        })

        fixed_cells = NotebookCells.from_dict(
            result_state.get("notebook_cells", merged_cells.to_dict())
        )
        cells_modified = result_state.get("cells_to_modify", [])
        explanation = (
            result_state.get("educational_response")
            or result_state.get("tutor_explanation")
            or ""
        )

        error = _validate_cells(fixed_cells)
        if error:
            raise ValueError(f"Debug fix validation failed: {error}")

        version, notebook = await self._create_and_save_version(
            session_id=session_id,
            prompt=debug_prompt,
            cells=fixed_cells,
            operation_type="debug",
            cells_modified=cells_modified,
        )

        return {
            "status": "success",
            "cells": fixed_cells.to_dict(),
            "cells_modified": cells_modified,
            "tutor_explanation": explanation,
            "version_number": version.version_number,
            "version_id": version.version_id,
        }

    # ------------------------------------------------------------------
    # Rollback — metadata-only, no file I/O except for reconstruction
    # ------------------------------------------------------------------
    def rollback_to(self, session_id: str, version_number: int) -> dict:
        """
        Sets active_version_id to the target version.
        For delta versions, cells are reconstructed from the snapshot chain.
        """
        notebook = self.notebook_repo.get_notebook_by_session(session_id)
        if not notebook:
            raise ValueError(f"No notebook found for session '{session_id}'")

        target = self.version_repo.get_version_by_number(notebook.notebook_id, version_number)
        if not target:
            raise ValueError(f"Version {version_number} not found")

        previous_active_id = notebook.active_version_id

        self.notebook_repo.set_active_version(notebook.notebook_id, target.version_id)
        self.version_repo.log_operation(
            version_id=target.version_id,
            action="rolled_back_to",
            details=f"Previous active: {previous_active_id}"
        )
        self.db.commit()

        # Load cells (transparent: handles both snapshot and delta)
        cells = storage_service.load_notebook(
            target.file_path,
            get_ancestor_cells=self._make_ancestor_loader(session_id, notebook.notebook_id),
        )

        # Update the live active working notebook
        storage_service.update_active_notebook(session_id, cells)

        return {
            "status": "success",
            "cells": cells.to_dict(),
            "cells_modified": list(cells.to_dict().keys()),
            "tutor_explanation": f"Rolled back to version {version_number}: {target.summary or ''}",
            "version_number": target.version_number,
            "version_id": target.version_id,
        }

    # ------------------------------------------------------------------
    # History & Search
    # ------------------------------------------------------------------
    def get_history(self, session_id: str) -> dict:
        """Returns complete version timeline for a session."""
        notebook = self.notebook_repo.get_notebook_by_session(session_id)
        if not notebook:
            return {"session_id": session_id, "versions": []}

        versions = self.version_repo.get_history(notebook.notebook_id)
        active_id = notebook.active_version_id

        return {
            "session_id": session_id,
            "notebook_id": notebook.notebook_id,
            "active_version_id": active_id,
            "versions": [v.to_dict(is_active=(v.version_id == active_id)) for v in versions]
        }

    def get_active_cells(self, session_id: str) -> Optional[NotebookCells]:
        """Loads cells from the currently active version file, with disk fallback."""
        notebook = self.notebook_repo.get_notebook_by_session(session_id)
        if notebook and notebook.active_version_id:
            active = self.version_repo.get_version_by_id(notebook.active_version_id)
            if active:
                try:
                    return storage_service.load_notebook(
                        active.file_path,
                        get_ancestor_cells=self._make_ancestor_loader(
                            session_id, notebook.notebook_id
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Failed to load active notebook file: {e}")

        # Fallback: scan disk for latest snapshot version file
        try:
            from app.services.storage_service import STORAGE_ROOT
            session_dir = os.path.join(STORAGE_ROOT, f"session_{session_id}")
            if os.path.exists(session_dir):
                # Prefer .ipynb snapshots for fallback (simpler, no reconstruction needed)
                files = [
                    f for f in os.listdir(session_dir)
                    if f.startswith("version_") and f.endswith(".ipynb")
                ]
                if files:
                    def get_ver_num(filename: str) -> int:
                        try:
                            return int(filename.split("_")[1].split(".")[0])
                        except Exception:
                            return 0
                    files.sort(key=get_ver_num, reverse=True)
                    latest_file = os.path.join(session_dir, files[0])
                    logger.info(f"[VersionService] DB out of sync, loaded fallback from: {latest_file}")
                    return storage_service.load_notebook(latest_file)
        except Exception as e:
            logger.warning(f"Disk fallback for active cells failed: {e}")

        return None

    def semantic_search(self, session_id: str, query: str, n_results: int = 5) -> dict:
        """Natural language search over version summaries via ChromaDB."""
        results = chroma_service.semantic_search(session_id, query, n_results)
        return {
            "session_id": session_id,
            "query": query,
            "results": results
        }

    # ------------------------------------------------------------------
    # Internal — shared version creation lifecycle
    # ------------------------------------------------------------------
    async def _create_and_save_version(
        self,
        session_id: str,
        prompt: str,
        cells: NotebookCells,
        operation_type: str,
        cells_modified: Optional[List[str]] = None,
        extra_metadata: Optional[dict] = None,
    ) -> Tuple[NotebookVersion, Notebook]:
        """
        Steps 6–11 of the version lifecycle:
        load parent → save file (snapshot or delta) → check duplicate →
        insert DB record → update active pointer → index ChromaDB → update Mem0.
        """
        notebook = self.notebook_repo.get_or_create_notebook(session_id)
        version_number = self.version_repo.get_next_version_number(notebook.notebook_id)

        # Load parent cells for delta computation
        parent_cells: Optional[NotebookCells] = None
        if notebook.active_version_id:
            parent_ver = self.version_repo.get_version_by_id(notebook.active_version_id)
            if parent_ver:
                try:
                    parent_cells = storage_service.load_notebook(
                        parent_ver.file_path,
                        get_ancestor_cells=self._make_ancestor_loader(
                            session_id, notebook.notebook_id
                        ),
                    )
                except Exception as e:
                    logger.warning(f"[VersionService] Could not load parent cells for delta: {e}")

        # Save file (StorageService decides snapshot vs delta)
        file_path, checksum, is_snapshot, delta_size = storage_service.save_notebook(
            session_id=session_id,
            version_number=version_number,
            cells=cells,
            parent_cells=parent_cells,
        )

        # Duplicate detection: skip if checksum matches current active
        if notebook.active_version_id:
            current = self.version_repo.get_version_by_id(notebook.active_version_id)
            if current and current.checksum == checksum:
                logger.info("[VersionService] No change detected (checksum match). Skipping version creation.")
                return current, notebook

        # Generate summary (async LLM call)
        summary = await _generate_summary(prompt, cells_modified or [], operation_type)

        # Insert version record
        version = self.version_repo.create_version(
            notebook_id=notebook.notebook_id,
            version_number=version_number,
            operation_type=operation_type,
            file_path=file_path,
            checksum=checksum,
            is_snapshot=is_snapshot,
            delta_size=delta_size,
            prompt=prompt,
            summary=summary,
            parent_version_id=notebook.active_version_id,
            cells_modified=cells_modified or [],
            extra_metadata=extra_metadata or {}
        )

        # Update active version pointer
        self.notebook_repo.set_active_version(notebook.notebook_id, version.version_id)
        self.version_repo.log_operation(
            version_id=version.version_id,
            action="created",
            details=f"op={operation_type} snapshot={is_snapshot} prompt='{prompt[:80]}'"
        )

        self.db.commit()

        # Index in ChromaDB (non-blocking on failure)
        try:
            indexed = chroma_service.index_version(
                version_id=version.version_id,
                session_id=session_id,
                version_number=version_number,
                prompt=prompt,
                summary=summary,
                cells_modified=cells_modified or [],
                operation_type=operation_type,
            )
            if indexed:
                self.version_repo.mark_chroma_indexed(version.version_id)
                self.db.commit()
        except Exception as e:
            logger.warning(f"[VersionService] ChromaDB indexing failed (non-critical): {e}")

        # Update Mem0 preferences (non-blocking on failure)
        try:
            memory_service.update_preferences(session_id, prompt, summary)
        except Exception as e:
            logger.warning(f"[VersionService] Mem0 update failed (non-critical): {e}")

        return version, notebook

    # ------------------------------------------------------------------
    # Delta reconstruction helper
    # ------------------------------------------------------------------
    def _make_ancestor_loader(
        self, session_id: str, notebook_id: str
    ):
        """
        Returns a callable: (base_version_number: int) → NotebookCells | None

        Injected into StorageService.load_notebook() so it can walk the
        snapshot chain without importing VersionService directly.
        """
        def _get_ancestor_cells(base_version_number: int) -> Optional[NotebookCells]:
            ancestor = self.version_repo.get_version_by_number(notebook_id, base_version_number)
            if ancestor is None:
                logger.warning(
                    f"[VersionService] Ancestor v{base_version_number} not found in DB"
                )
                return None
            if not ancestor.is_snapshot:
                logger.error(
                    f"[VersionService] Ancestor v{base_version_number} is not a snapshot; "
                    "cannot use as reconstruction base."
                )
                return None
            try:
                # Snapshots are always .ipynb — no recursion needed
                return storage_service.load_notebook(ancestor.file_path)
            except Exception as e:
                logger.error(f"[VersionService] Failed to load ancestor snapshot: {e}")
                return None

        return _get_ancestor_cells
