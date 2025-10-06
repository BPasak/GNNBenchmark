from torch import nn, Tensor
import torch_geometric.data

from Models.base import BaseModel


class ExampleModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(10, 10)

    def forward(self, x: torch_geometric.data.Data) -> Tensor:
        x = Tensor([0,1,2,3,4,5,6,7,8,9])
        x = self.linear(x)
        return x