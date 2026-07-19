"""
app/schemas/notebook.py
-----------------------
Pydantic models for typed notebook cell handling and delta compression.

NotebookCells  — the 12 named DEAP cells as typed, validated fields.
CellDelta      — a diff between two consecutive versions; used for storage
                 compression (only changed cells are persisted).

These replace the raw Dict[str, str] used throughout the stack and act as
the single source of truth for cell ordering (DEAP_CELL_ORDER).
"""
from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, model_validator


# ---------------------------------------------------------------------------
# Single source of truth for DEAP cell ordering
# ---------------------------------------------------------------------------
DEAP_CELL_ORDER: list[str] = [
    "imports",
    "config",
    "creator",
    "evaluation",
    "crossover",
    "mutation",
    "selection",
    "initialization",
    "toolbox",
    "main_algorithm",
    "stats",
    "visualization",
]


# ---------------------------------------------------------------------------
# NotebookCells — typed model for the 12 DEAP notebook cells
# ---------------------------------------------------------------------------
class NotebookCells(BaseModel):
    """
    Typed representation of the 12 DEAP notebook cells.

    All fields are optional so that a partial update (delta) can be
    expressed by setting only the changed cells and leaving the rest None.
    """

    imports: Optional[str] = None
    config: Optional[str] = None
    creator: Optional[str] = None
    evaluation: Optional[str] = None
    crossover: Optional[str] = None
    mutation: Optional[str] = None
    selection: Optional[str] = None
    initialization: Optional[str] = None
    toolbox: Optional[str] = None
    main_algorithm: Optional[str] = None
    stats: Optional[str] = None
    visualization: Optional[str] = None

    model_config = {"extra": "ignore"}  # silently drop unknown cell names

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, str]:
        """Return only cells that have a non-None value."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def to_ordered_dict(self) -> Dict[str, str]:
        """Return non-None cells in canonical DEAP order."""
        full = self.model_dump()
        return {k: full[k] for k in DEAP_CELL_ORDER if full.get(k) is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "NotebookCells":
        """
        Build a NotebookCells from a plain dict.
        Keys not in DEAP_CELL_ORDER are silently ignored (model_config extra='ignore').
        """
        return cls(**{k: v for k, v in d.items()})

    def diff(self, other: "NotebookCells") -> "NotebookCells":
        """
        Compute a partial NotebookCells that contains only the cells
        where `other` differs from `self`.  Used to create a CellDelta.

        Returns a NotebookCells where:
          - A field is set  → the cell changed (new value from `other`)
          - A field is None → the cell is unchanged
        """
        self_dict = self.model_dump()
        other_dict = other.model_dump()
        changed: Dict[str, Optional[str]] = {}
        for cell_name in DEAP_CELL_ORDER:
            old_val = self_dict.get(cell_name)
            new_val = other_dict.get(cell_name)
            if old_val != new_val:
                changed[cell_name] = new_val
            else:
                changed[cell_name] = None  # unchanged
        return NotebookCells(**changed)

    def is_empty(self) -> bool:
        """True when no cells have content (all None)."""
        return all(v is None for v in self.model_dump().values())

    def merge(self, overlay: "NotebookCells") -> "NotebookCells":
        """
        Returns a new NotebookCells with all non-None fields from `overlay`
        applied on top of `self`.  Used during delta reconstruction.
        """
        base = self.model_dump()
        for k, v in overlay.model_dump().items():
            if v is not None:
                base[k] = v
        return NotebookCells(**base)


# ---------------------------------------------------------------------------
# CellDelta — a compressed diff between two consecutive versions
# ---------------------------------------------------------------------------
class CellDelta(BaseModel):
    """
    Represents the difference between a version and its nearest full-snapshot
    ancestor.  Only `changed_cells` (a partial NotebookCells) is persisted.

    Fields
    ------
    base_version_number : int
        The version_number of the nearest ancestor that is a full snapshot.
        Used during reconstruction to locate the base .ipynb file.
    is_snapshot : bool
        Always False for CellDelta objects (True means this is a full snapshot,
        stored as .ipynb, not a .delta.json file).
    changed_cells : NotebookCells
        Partial cells where only changed fields are set; unchanged fields are None.
    """

    base_version_number: int
    is_snapshot: bool = False
    changed_cells: NotebookCells

    def apply_to(self, base: NotebookCells) -> NotebookCells:
        """
        Reconstruct a full NotebookCells by merging this delta onto `base`.

        base     — the full snapshot (is_snapshot=True) this delta is relative to
        returns  — a complete NotebookCells with all 12 cells populated
        """
        return base.merge(self.changed_cells)

    @classmethod
    def compute(
        cls,
        parent_cells: NotebookCells,
        new_cells: NotebookCells,
        base_version_number: int,
    ) -> "CellDelta":
        """
        Factory: compute the delta between parent_cells and new_cells.

        base_version_number — the snapshot version number that will be used
                              as the reconstruction anchor.
        """
        changed = parent_cells.diff(new_cells)
        return cls(base_version_number=base_version_number, changed_cells=changed)

    @property
    def changed_cell_names(self) -> list[str]:
        """List of cell names that are different in this delta."""
        return [k for k, v in self.changed_cells.model_dump().items() if v is not None]

    @property
    def delta_size_bytes(self) -> int:
        """Approximate storage size of this delta when serialized to JSON."""
        return len(self.model_dump_json().encode("utf-8"))
