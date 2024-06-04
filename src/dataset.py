import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class WaferDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        waferMap = self.dataframe.iloc[idx]["waferMap"]
        waferMap = np.expand_dims(waferMap, axis=2)
        waferMap = np.repeat(waferMap, 3, axis=2)
        label = self.dataframe.iloc[idx]["encoded_labels"]

        if self.transform:
            waferMap = self.transform(waferMap)

        return waferMap, label
