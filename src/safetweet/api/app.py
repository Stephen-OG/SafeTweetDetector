from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException

from safetweet.inference.service import HuggingFaceProvider, MockProvider, ModerationService
from safetweet.schemas import ModerateRequest, ModerateResponse, QueueItem, ReviewRequest
from safetweet.storage import ModerationStore


def create_app(
    *,
    store: ModerationStore | None = None,
    provider=None,
) -> FastAPI:
    db_path = Path(os.getenv("SAFETWEET_DB_PATH", "var/moderation.db"))
    model_dir = os.getenv("SAFETWEET_MODEL_DIR")
    resolved_store = store or ModerationStore(db_path)
    resolved_store.initialize()
    resolved_provider = provider or (
        HuggingFaceProvider(model_dir) if model_dir else MockProvider()
    )
    service = ModerationService(provider=resolved_provider, store=resolved_store)

    app = FastAPI(title="Safe Tweet Detector", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ready", "model_version": resolved_provider.model_version}

    @app.post("/moderate", response_model=ModerateResponse)
    def moderate(request: ModerateRequest):
        return service.moderate(request.text)

    @app.get("/queue", response_model=list[QueueItem])
    def queue():
        return resolved_store.list_queue()

    @app.patch("/review/{item_id}", response_model=QueueItem)
    def review(item_id: int, request: ReviewRequest):
        try:
            return resolved_store.review(
                item_id,
                status=request.status,
                reviewer_label=request.reviewer_label,
                notes=request.notes,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/analytics")
    def analytics():
        return resolved_store.analytics()

    return app


app = create_app()


def main() -> None:
    uvicorn.run("safetweet.api.app:app", host="127.0.0.1", port=8000, reload=True)
