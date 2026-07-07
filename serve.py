"""FastAPI server for next-action prediction.

Usage:
    python serve.py
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"events": [{"app_name": "Outlook", "window_title": "Inbox", "event_type": "focus_in"}]}'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agentic_workflow_discovery.model.lstm_model import SequenceLSTM
from agentic_workflow_discovery.model.tokenizer import EventTokenizer

app = FastAPI(title="Agentic Workflow Discovery API", version="0.1.0")

# ── Request / Response schemas ──────────────────────────────────────────

class EventInput(BaseModel):
    app_name: str
    window_title: str
    event_type: str


class PredictRequest(BaseModel):
    events: list[EventInput]


class TopKItem(BaseModel):
    token_id: int
    score: float
    state: str


class PredictResponse(BaseModel):
    token_id: int
    confidence: float
    top_k: list[TopKItem]


# ── Model initialisation ───────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "models/checkpoint.pt")
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", "1000"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

model = SequenceLSTM(vocab_size=VOCAB_SIZE, embedding_dim=128, hidden_dim=256, n_layers=2)
tokenizer = EventTokenizer(max_vocab_size=VOCAB_SIZE)

checkpoint = Path(MODEL_PATH)
if checkpoint.exists():
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    print(f"Loaded checkpoint from {MODEL_PATH}")
else:
    print(f"Warning: no checkpoint found at {MODEL_PATH}. Using untrained model.", file=sys.stderr)

model.to(DEVICE)
model.eval()


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model_loaded": checkpoint.exists()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.events:
        raise HTTPException(status_code=400, detail="events list is empty")

    # Convert Pydantic models to token IDs
    # We construct lightweight objects matching the tokenizer's expected attrs
    class _Event:
        def __init__(self, e: EventInput):
            self.app_name = e.app_name
            self.window_title = e.window_title
            self.event_type = e.event_type

    dummy_events = [_Event(e) for e in req.events]

    try:
        token_ids = tokenizer.encode(dummy_events)
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Tokenizer not fitted. Train a model first.")

    if len(token_ids) == 0:
        raise HTTPException(status_code=400, detail="All events mapped to unknown tokens")

    tensor = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits[0], dim=-1)

    top_scores, top_indices = probs.topk(3)

    top_k = [
        TopKItem(
            token_id=int(idx),
            score=float(score),
            state=tokenizer.decode(int(idx)),
        )
        for idx, score in zip(top_indices, top_scores)
    ]

    return PredictResponse(
        token_id=int(top_indices[0]),
        confidence=float(top_scores[0]),
        top_k=top_k,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
