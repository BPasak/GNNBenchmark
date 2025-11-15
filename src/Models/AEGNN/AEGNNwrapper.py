# AEGNNwrapper.py
from __future__ import annotations

import inspect
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

# from .base import BaseModel  # if using packages
from base import BaseModel     # adjust to your layout

def _infer_r_from_model(model: nn.Module) -> int:
    """Infer receptive-field hops as the count of MessagePassing layers."""
    return max(1, sum(1 for m in model.modules() if isinstance(m, MessagePassing)))

class AEGNNAsyncWrapper(nn.Module):
    """
    Wrap any BaseModel with AEGNN async execution.
    - Auto-fills required args like `r` if the function expects them.
    - Records why async failed in `why_not_async`.
    """
    def __init__(self, model: BaseModel, **aegnn_kwargs):
        super().__init__()
        self.model = model
        self._async_impl: nn.Module = self.model
        self._is_async: bool = False
        self._async_error: str | None = None
        self._accepted_args: list[str] = []

        try:
            from aegnn.asyncronous import make_model_asynchronous
            sig = inspect.signature(make_model_asynchronous)
            params = sig.parameters
            self._accepted_args = [n for n,p in params.items()
                                   if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]

            # If `r` is required and not provided, infer it.
            if "r" in params:
                p = params["r"]
                needs_r = (p.default is inspect._empty) and ("r" not in aegnn_kwargs)
                if needs_r:
                    aegnn_kwargs["r"] = _infer_r_from_model(self.model)

            # Filter unknown kwargs
            filtered = {k: v for k, v in aegnn_kwargs.items() if k in self._accepted_args}
            dropped  = [k for k in aegnn_kwargs.keys() if k not in filtered]
            if dropped:
                print(f"[AEGNN] Dropping unsupported kwargs: {dropped}")

            self._async_impl = make_model_asynchronous(self.model, **filtered)
            self._is_async = True
        except Exception as e:
            self._async_error = f"{type(e).__name__}: {e}"

    @property
    def is_async(self) -> bool:
        return self._is_async

    @property
    def why_not_async(self) -> str | None:
        return self._async_error

    @property
    def accepted_async_args(self) -> list[str]:
        return list(self._accepted_args)

    def forward(self, data: Data) -> torch.Tensor:
        return self._async_impl(data)

    def base_model(self) -> BaseModel:
        return self.model
