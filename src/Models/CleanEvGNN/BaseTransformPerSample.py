import torch
from torch_geometric.data import Data, Batch


import random
from torch_geometric.transforms import BaseTransform, Compose
from abc import abstractmethod
class BaseTransformPerSample(BaseTransform):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def transform_per_sample(self, data: Data):
        return data

    def __call__(self, batch: Batch):
        data_list = batch.to_data_list()
        transformed_data_list = []
        for data in data_list:
            transformed_data = self.transform_per_sample(data)
            transformed_data_list.append(transformed_data)
        transformed_batch = Batch.from_data_list(transformed_data_list)
        return transformed_batch


class RandomXFlip(BaseTransform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            pos = data.pos.clone()
            max,_ = pos.max(dim=0)
            max_x = max[0]
            pos[:,0] = -pos[:,0] + max_x
            data.pos = pos
        return data


class RandomShiftPerSample(BaseTransform):
    def __init__(self, p=0.5):
        self.p = p

    def sample_pos_trans(self, ori_pos):
        pos = ori_pos.clone()
        if random.random() < self.p:
            if pos.dim() == 1:
                pos = pos.unsqueeze(0)
            max,_ = pos.max(dim=0)
            x_max, y_max = max[0], max[1]
            dx = 119 - x_max
            dy = 99 - y_max
            rx = random.random()
            ry = random.random()
            sx = torch.floor(rx * dx)
            sy = torch.floor(ry * dy)
            pos[:,0] += sx
            pos[:,1] += sy
            pos = pos.squeeze()
        return pos

    def __call__(self, data):
        unique_samples = torch.unique(data.batch)
        for sample in unique_samples:
            # find which rows belong to which samples
            sample_indices = (data.batch == sample).nonzero().squeeze()

            # extract from data pos
            sample_pos = data.pos[sample_indices].clone()

            # bigmax,_ = sample_pos.max(dim=0)
            # x_bigmax, y_bigmax = bigmax[0], bigmax[1]

            tpos = self.sample_pos_trans(sample_pos)
            data.pos[sample_indices] = tpos
        return data


class RandomSubgraph(BaseTransformPerSample):
    def __init__(self, num_samples: int, p: float = 0.5):
        self.num_samples = num_samples
        self.p = p

    def transform_per_sample(self, data: Data):
        if (random.random() < self.p) and (data.label[0] == 'car'):
            real_num_samples = max(1, min(data.num_nodes, self.num_samples))  # real_num_samples = min(num_nodes, num_samples), and >= 1
            subset = random.sample(range(data.num_nodes), real_num_samples)
            sorted_unique_subset = torch.tensor(subset).sort().values
            data_subgraph = data.subgraph(sorted_unique_subset).clone()
        else:
            data_subgraph = data
        return data_subgraph

class RandomRangeSubgraph(BaseTransformPerSample):
    def __init__(self, range_start: int, range_end: int, p: float = 0.5):
        self.range_start = range_start
        self.range_end = range_end
        self.p = p

    def transform_per_sample(self, data: Data):
        if (random.random() < self.p) \
            and (data.label[0] == 'car') \
            and (data.num_nodes >= self.range_start) \
            and (data.num_nodes < self.range_end):

            bias = 500  # no sample has nodes less than 500
            scale = self.range_start - bias
            num_samples = int(random.random() * scale) + bias  # choose a random num samples that < range_start
            subset = random.sample(range(data.num_nodes), num_samples)
            sorted_unique_subset = torch.tensor(subset).sort().values
            data_subgraph = data.subgraph(sorted_unique_subset).clone()
        else:
            data_subgraph = data
        return data_subgraph
