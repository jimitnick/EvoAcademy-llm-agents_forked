"""
EvoAcademy FastAPI Application
Version History System — Modular Architecture

Storage layout:
  storage/notebooks/session_{id}/version_{n}.ipynb  — immutable .ipynb files
  evo_academy.db                                      — SQLite metadata (SQLAlchemy)
  .chroma_version_store/                              — ChromaDB semantic index
  Mem0 cloud                                          — user preferences

API Endpoints:
  POST /generate                         — generate new EA notebook
  POST /refine                           — refine existing notebook
  POST /debug                            — auto-fix runtime traceback
  GET  /sessions/{id}/history            — full version timeline
  POST /sessions/{id}/rollback           — metadata-only rollback
  GET  /sessions/{id}/search?q=...       — semantic search via ChromaDB
  GET  /health                           — connectivity check
"""
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import init_db
from app.api.routes import generate, refine, debug, history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB tables on startup."""
    logger.info("Initializing database...")
    init_db()
    logger.info("EvoAcademy API v2.0 ready.")
    yield

app = FastAPI(
    title="EvoAcademy — Evolutionary Algorithm Notebook API",
    description=(
        "Git-like version history for AI-generated DEAP notebooks. "
        "Every modification creates an immutable .ipynb version. "
        "Rollback changes only the active_version_id pointer."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(generate.router, tags=["Notebooks"])
app.include_router(refine.router, tags=["Notebooks"])
app.include_router(debug.router, tags=["Notebooks"])
app.include_router(history.router, tags=["Version History"])


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "ok",
        "version": "2.0.0",
        "features": ["version_history", "semantic_search", "rollback", "user_preferences"]
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
