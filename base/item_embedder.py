import torch
import torch.nn as nn
import numpy as np


# ═══════════════════════════════════════════════════════════════
# Item Embedding Layer
# ═══════════════════════════════════════════════════════════════

class ItemEmbedder(nn.Module):
    """
    Learnable embedding for stock identifiers.

    Maps integer stock IDs to real-valued vectors via nn.Embedding.

    Args:
        num_items    : total number of distinct stocks 
        embedding_dim: output vector dimension 

    Usage:
        embedder = ItemEmbedder(num_items=3, embedding_dim=1)
        ids      = torch.tensor([0, 0, 1, 2])   # stock IDs
        out      = embedder(ids)                 # shape: (4, 1)
    """

    # Default stock ID registry
    STOCK_MAP = {"SPY": 0, "DIA": 1, "QQQ": 2}

    def __init__(self, num_items: int, embedding_dim: int):
        super().__init__()
        self.num_items     = num_items
        self.embedding_dim = embedding_dim
        self.embedding     = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim,
        )

    # ─────────────────────────────────────────
    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_ids: torch.LongTensor of shape (N,) or (N, T)

        Returns:
            torch.Tensor of shape (N, embedding_dim) or (N, T, embedding_dim)
        """
        return self.embedding(item_ids)

    # ─────────────────────────────────────────
    @classmethod
    def id_from_ticker(cls, ticker: str) -> int:
        """
        Convert ticker string to integer ID.

        Args:
            ticker: e.g. 'SPY', 'DIA', 'QQQ'

        Returns:
            integer stock ID

        Example:
            ItemEmbedder.id_from_ticker('SPY')  # → 0
        """
        ticker = ticker.upper()
        if ticker not in cls.STOCK_MAP:
            raise ValueError(
                f"Unknown ticker '{ticker}'. "
                f"Registered: {list(cls.STOCK_MAP.keys())}"
            )
        return cls.STOCK_MAP[ticker]

    # ─────────────────────────────────────────
    @classmethod
    def register_ticker(cls, ticker: str, stock_id: int) -> None:
        """
        Register a new ticker → ID mapping.

        Args:
            ticker  : e.g. 'TSLA'
            stock_id: integer ID (must be < num_items)

        Example:
            ItemEmbedder.register_ticker('TSLA', 3)
        """
        cls.STOCK_MAP[ticker.upper()] = stock_id