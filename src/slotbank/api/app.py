from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from slotbank.api.chat import register_chat
from slotbank.api.messages import register_messages
from slotbank.api.responses import register_responses


def create_app(engine, *, api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="slotbank", docs_url=None, redoc_url=None)
    app.state.engine = engine
    app.state.api_key = api_key

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        if request.url.path in {"/health", "/"}:
            return await call_next(request)
        expected = app.state.api_key
        if not expected:
            return await call_next(request)
        got = (
            request.headers.get("x-api-key")
            or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        )
        if got != expected:
            return JSONResponse({"error": {"message": "invalid api key", "type": "auth"}}, 401)
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": engine.model_id}

    @app.get("/")
    def root():
        return {
            "name": "slotbank",
            "model": engine.model_id,
            "endpoints": [
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/models",
                "/v1/messages",
                "/v1/messages/count_tokens",
                "/v1/responses",
            ],
        }

    register_chat(app, engine)
    register_messages(app, engine)
    register_responses(app, engine)
    return app
