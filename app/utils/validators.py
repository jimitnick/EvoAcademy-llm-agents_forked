"""
Shared input validation constants and utilities for EvoAcademy API.

Centralises all sanitisation rules so every endpoint enforces the same
constraints without duplicating logic.
"""
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# ── Valid DEAP notebook cell names ──────────────────────────────
VALID_CELL_NAMES = frozenset([
    "imports", "config", "creator", "evaluation", "crossover", "mutation",
    "selection", "initialization", "toolbox", "main_algorithm", "stats",
    "visualization",
])

# ── Constraints (shared across Pydantic models & Path/Query) ───
SESSION_ID_PATTERN = r"^[a-zA-Z0-9_\-]+$"
SESSION_ID_MAX_LENGTH = 128

PROMPT_MIN_LENGTH = 3
PROMPT_MAX_LENGTH = 5000

TRACEBACK_MAX_LENGTH = 10000

SEARCH_QUERY_MAX_LENGTH = 500
SEARCH_MAX_RESULTS = 50


def validate_cell_keys(cells: Dict[str, str]) -> Dict[str, str]:
    """Raise ValueError if any dict key is not a recognised DEAP cell name."""
    invalid = set(cells.keys()) - VALID_CELL_NAMES
    if invalid:
        raise ValueError(
            f"Invalid cell names: {sorted(invalid)}. "
            f"Valid names: {sorted(VALID_CELL_NAMES)}"
        )
    return cells


async def check_ea_domain_relevance(prompt: str) -> Tuple[bool, str]:
    """
    LLM-based domain guardrail — checks whether *prompt* is related to
    Evolutionary Algorithms / DEAP.

    Returns ``(is_valid, reason)``.  On LLM failure the function raises so
    callers can decide whether to fail-open or fail-closed.
    """
    from pydantic import BaseModel, Field as PydanticField
    from app.core.llm import coder_llm

    class _PromptValidation(BaseModel):
        is_valid_ea: bool = PydanticField(
            description=(
                "True if the prompt is about Evolutionary Algorithms, "
                "Genetic Algorithms, optimisation, or related topics."
            )
        )
        reason: str = PydanticField(
            description=(
                "If false, a brief polite message explaining that "
                "this platform is only for Evolutionary Algorithm topics."
            )
        )

    validator = coder_llm.with_structured_output(_PromptValidation)

    system_prompt = (
        "You are the domain gatekeeper for an educational platform "
        "teaching Evolutionary Algorithms (DEAP).\n"
        f'Evaluate this user prompt: "{prompt}"\n'
        "Return True if the prompt relates to: optimisation, genetic algorithms, "
        "evolutionary strategies, DEAP parameters, fitness functions, crossover / "
        "mutation operators, selection schemes, math, or educational questions "
        "about the student's existing EA notebook.\n"
        "Return False ONLY if the prompt is completely unrelated to EA / DEAP "
        "(e.g., Todo lists, web apps, databases, games)."
    )

    decision = await validator.ainvoke(system_prompt)
    return decision.is_valid_ea, decision.reason
