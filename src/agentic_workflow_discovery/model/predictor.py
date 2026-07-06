"""Inference wrapper: next-token prediction with confidence scores.

The Predictor takes a trained model and a tokenizer, accepts raw events
as input, and returns the predicted next token with its softmax confidence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from agentic_workflow_discovery.ingestion.schemas import EventRecord
from agentic_workflow_discovery.model.lstm_model import SequenceLSTM
from agentic_workflow_discovery.model.tokenizer import EventTokenizer


class Predictor:
    """Wraps a trained model for next-event prediction.

    Parameters
    ----------
    model : SequenceLSTM
        Trained model (in eval mode).
    tokenizer : EventTokenizer
        Fitted tokenizer.
    device : torch.device | str
    """

    def __init__(
        self,
        model: SequenceLSTM,
        tokenizer: EventTokenizer,
        device: torch.device | str = "auto",
    ):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer

        if isinstance(device, str) and device == "auto":
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        self.model.to(self.device)

    @torch.inference_mode()
    def predict_next(
        self,
        context_events: list[EventRecord],
        top_k: int = 3,
    ) -> dict:
        """Predict the next event token given a context window.

        Parameters
        ----------
        context_events : list[EventRecord]
            Exactly ``context_length`` events (must match model training).
        top_k : int
            Number of top predictions to return.

        Returns
        -------
        dict with keys:
            "token_id" : int — predicted token ID
            "confidence" : float — softmax score of predicted token
            "top_k" : list[dict] — top-k (token_id, score, state_string)
        """
        token_ids = self.tokenizer.encode(context_events)
        if len(token_ids) == 0:
            return {"token_id": 0, "confidence": 0.0, "top_k": []}

        tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        logits = self.model(tensor)  # (1, V)
        probs = F.softmax(logits[0], dim=-1)  # (V,)

        top_scores, top_indices = probs.topk(top_k)

        predicted_id = int(top_indices[0])
        predicted_score = float(top_scores[0])

        top_k_list = [
            {
                "token_id": int(idx),
                "score": float(score),
                "state": self.tokenizer.decode(int(idx)),
            }
            for idx, score in zip(top_indices, top_scores)
        ]

        return {
            "token_id": predicted_id,
            "confidence": predicted_score,
            "top_k": top_k_list,
        }
