# ---------------- Async wrapper (separate class) ----------------
from sympy.printing.pytorch import torch
from torch_geometric import nn
from torch_geometric.data import Data

from src.Models.base import BaseModel


class AEGNNAsyncWrapper(nn.Module):
    """
    Wrapper class to convert a BaseModel into its asynchronous version using AEGNN"""
    def __init__(self, model: BaseModel, **aegnn_kwargs):
        super().__init__()
        self.model = model  # registered as a submodule (important for params/optimizers)
        self._is_async = False

        #try building the async version
        try:
            from asyncronous import make_model_asynchronous
            self._async_impl = make_model_asynchronous(self.model, **aegnn_kwargs)
            self._is_async = True
        except Exception as e:
            #on failure, fall back to the base model
            self._async_impl = self.model

    @property
    def is_async(self) -> bool:
        return self._is_async

    def forward(self, data: Data) -> torch.Tensor:
        return self._async_impl(data)

    def base_model(self) -> BaseModel:
        return self.model