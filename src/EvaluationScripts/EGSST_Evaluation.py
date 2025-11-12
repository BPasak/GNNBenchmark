import os

from Datasets.base import Dataset
from Datasets.ncars import NCars

dataset: Dataset = NCars(root=os.getenv("NCARS_ROOT"))