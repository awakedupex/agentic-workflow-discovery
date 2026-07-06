"""A compact GRU/LSTM sequence model for next-event prediction.

Architecture choices:
---------------------
- Embedding layer: maps discrete token IDs to dense vectors.
- GRU (or LSTM): processes the embedding sequence.  GRU is the default
  because it has fewer parameters (3 gates vs 4) and lower latency, with
  comparable accuracy on short sequences (context_length=5).
- Linear head: projects the final hidden state to vocab-size logits.
- Dropout between layers for regularisation.

Why GRU over Transformer for this task?
----------------------------------------
- The context window is short (5 tokens).  Self-attention's advantage
  over RNNs grows with sequence length; at length 5 there is essentially
  no gap.
- GRU inference is O(context_length · d²) with no KV-cache overhead.
  A Transformer at this scale has comparable FLOPs but higher memory
  bandwidth pressure from the attention matrix.
- For longer contexts (>32 tokens) we would switch to Transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SequenceLSTM(nn.Module):
    """GRU-based next-event prediction model.

    Parameters
    ----------
    vocab_size : int
        Number of unique tokens (including [PAD] and [UNK]).
    embedding_dim : int
        Dimensionality of token embeddings (default 128).
    hidden_dim : int
        Dimensionality of the GRU hidden state (default 256).
    n_layers : int
        Number of stacked GRU layers (default 2).
    dropout : float
        Dropout probability between layers (default 0.2).
    rnn_type : str
        "gru" or "lstm" (default "gru").
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "gru",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        rnn_class = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch_size, seq_len) token IDs.

        Returns:
            logits: (batch_size, vocab_size) — prediction for the *last*
                    position in each sequence.
        """
        # (B, S) → (B, S, D)
        emb = self.embedding(x)

        # (B, S, D) → (B, S, H)
        rnn_out, _ = self.rnn(emb)

        # Take the final time step's output
        # rnn_out shape: (B, S, H) → last_out: (B, H)
        last_out = rnn_out[:, -1, :]
        last_out = self.dropout(last_out)

        # (B, H) → (B, V)
        logits = self.head(last_out)
        return logits
