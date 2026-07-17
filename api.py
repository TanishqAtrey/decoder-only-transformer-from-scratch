"""
api.py
FastAPI REST server for KANHA.

Endpoints:
    POST /chat      — single-turn chat
    POST /reset     — clear memory
    GET  /health    — health check

Run:
    python main.py api --model models/finetuned/sft_final.pt --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from kanha.inference.engine import InferenceEngine
from kanha.utils.logging import get_logger

log = get_logger(__name__)

# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    stream: Optional[bool] = False


class ChatResponse(BaseModel):
    response: str
    model: str = "kanha"


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(model_path: str, index_dir: str = None) -> FastAPI:
    app = FastAPI(
        title="KANHA AI API",
        description="Knowledge-Augmented Neural Heuristic Assistant",
        version="0.1.0",
    )

    # Load engine once at startup
    engine = InferenceEngine.from_pretrained(
        model_path=model_path,
        index_dir=index_dir,
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_path}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        if not req.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
        try:
            response = engine.chat(req.message, stream=False)
            return ChatResponse(response=response)
        except Exception as e:
            log.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/reset")
    def reset():
        engine.reset_memory()
        return {"status": "memory cleared"}

    return app