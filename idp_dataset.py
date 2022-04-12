import os
from enum import Enum
import os.path as osp

from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import pandas as pd


# Amino acid list based on: https://www.mathworks.com/help/bioinfo/ref/aa2int.html
AAs = "?ARNDCQEGHILKMFPSTWYVBZX*-"


class DatasetMode(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class IdpDataset(Dataset):

    def __init__(self, mode=DatasetMode.TRAIN):
        if mode == DatasetMode.TRAIN:
            self.directory = osp.join('dataset', 'train')
        elif mode == DatasetMode.VALIDATION:
            self.directory = osp.join('dataset', 'validation')
        elif mode == DatasetMode.TEST:
            self.directory = osp.join('dataset', 'test')
        else:
            raise ValueError('Invalid mode')
        self.proteins = []
        for file in tqdm(os.listdir(self.directory), desc="Loading data from {}".format(self.directory)):
            if not file.endswith('.parquet'):
                continue
            df = pd.read_parquet(osp.join(self.directory, file))
            df['sequence'] = df['sequence'].apply(AAs.index)
            self.proteins.append(df)
        self.label_names = ['is_disordered', 'is_binding', 'is_linker', 'is_protein_binding', 'is_nucleic_acid_binding']
        self.feat_names = self.proteins[0].loc[:, ~self.proteins[0].columns.isin(self.label_names)].columns.values.tolist()

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        df = self.proteins[idx]

        labels = torch.from_numpy(df.loc[:, self.label_names].values)
        feats = torch.from_numpy(df.loc[:, self.feat_names].values)
        return feats, labels


if __name__ == '__main__':
    dataset = IdpDataset(DatasetMode.TRAIN)
    print(dataset[0])
