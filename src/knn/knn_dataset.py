import pandas as pd
from torch.utils.data import Dataset


class KNNDataset(Dataset):
    def __init__(self, path):
        self._df = pd.read_csv(path, dtype=str, header="infer")

    def __len__(self):
        return self._df.shape[0]

    def __getitem__(self, index):
        return self._df.iloc[index][0], self._df.iloc[index][1]
