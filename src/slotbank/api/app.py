from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from slotbank.api.chat import llama_props_payload, models_payload, register_chat
from slotbank.api.messages import register_messages
from slotbank.api.responses import register_responses

_OPEN_PATHS = frozenset({
    "/health", "/", "/models", "/props", "/v1/models",
})


class LoadingEngine:
    """Serves discovery while the real Engine thread loads weights."""

    def __init__(self, model_id: str, *, error: str | None = None,
                 context_window: int = 16384):
        self.model_id = model_id
        self.context_window = context_window
        self.loading = error is None
        self.load_error = error

    def _busy(self) -> str:
        if self.load_error:
            return self.load_error
        return f"slotbank is still loading {self.model_id}"

    def tokenize_chat(self, messages, tools):
        raise ValueError(self._busy())

    def tokenize_text(self, text):
        raise ValueError(self._busy())

    def generate(self, ids, sampling, on_token=None):
        raise ValueError(self._busy())

    def stream(self, ids, sampling):
        raise ValueError(self._busy())
        yield from ()  # pragma: no cover — generator for type checkers


class EngineProxy:
    """Swap the inner engine after load without rewriting route closures."""

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def replace(self, inner) -> None:
        object.__setattr__(self, "_inner", inner)

    @property
    def inner(self):
        return object.__getattribute__(self, "_inner")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)


def create_app(engine, *, api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="slotbank", docs_url=None, redoc_url=None)
    app.state.engine = engine
    app.state.api_key = api_key

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        if request.url.path in _OPEN_PATHS:
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
        eng = app.state.engine
        err = getattr(eng, "load_error", None)
        if err:
            return JSONResponse(
                {"status": "error", "model": eng.model_id, "error": err}, 503,
            )
        if getattr(eng, "loading", False):
            return {"status": "loading", "model": eng.model_id}
        return {"status": "ok", "model": eng.model_id}

    @app.get("/")
    def root():
        return {
            "name": "slotbank",
            "model": app.state.engine.model_id,
            "endpoints": [
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/models",
                "/models",
                "/props",
                "/v1/messages",
                "/v1/messages/count_tokens",
                "/v1/responses",
            ],
        }

    @app.get("/models")
    def native_models():
        return models_payload(app.state.engine)

    @app.get("/props")
    def native_props():
        return llama_props_payload(app.state.engine)

    register_chat(app, engine)
    register_messages(app, engine)
    register_responses(app, engine)
    return app
