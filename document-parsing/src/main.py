"""FastAPI entrypoint for the document-parsing-service.

Routes:
    POST /api/v1/jobs         — submit async parse job (Celery)
    POST /api/v1/parse-image  — sync image OCR with per-stage timing
    POST /api/v1/parse        — sync parse for small files
    GET  /api/v1/jobs/...     — poll status / fetch markdown
    GET  /api/v1/health       — registry summary

Lifespan:
    startup → priming parser registry; if the parsing pipeline is available,
    eagerly instantiate VNDocParser so /parse-image's first request has no
    model-load latency.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from api.routes import router
from settings import settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup: priming parser registry...")
    try:
        from core.registry import registry
        reg = registry()
        log.info("registry has %d extensions", len(reg))
    except Exception:
        log.exception("registry build failed during startup")

    # Eagerly load VNDocParser if the parsing pipeline is available — this makes
    # the sync /parse-image endpoint respond without a 2-3 minute first-request stall.
    if not settings.pdf_force_plain:
        try:
            from parsers.pdf_layout import _get_parser
            _get_parser()
            log.info("VNDocParser ready (preloaded for sync /parse-image).")
        except Exception:
            log.warning("VNDocParser preload skipped — sync image OCR unavailable",
                        exc_info=True)

    yield
    log.info("shutdown.")


def create_app() -> FastAPI:
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = FastAPI(
        title="document-parsing-service",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
