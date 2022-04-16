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
    # Training dataset
    TRAIN = 0
    # Validation dataset for parameters optimization
    VALIDATION = 1
    # Test dataset for final evaluation
    TEST = 2


class IdpDataset(Dataset):

    def __init__(self, mode=DatasetMode.TRAIN, only_binary_labels=False,
                 normalize_features=True, normalization_means=None, normalization_stds=None):
        """
        Create the dataset assuming you have the dataset directory.
        NOTE: This currently does not work well with minibatches. So you must currently treat
        each sequence as its own minibatch.

        Example:
          >>> train_dataset = IdpDataset(only_binary_labels=True)
          >>> train_mean = train_dataset.normalization_means
          >>> train_std = train_dataset.normalization_stds
          >>> train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
          >>> validation_dataset = IdpDataset(DatasetMode.VALIDATION, only_binary_labels=True,
          >>>                                 normalization_means=train_mean, normalization_stds=train_std)
          >>> validation_loader = DataLoader(validation_dataset, batch_size=1)

        See also: basic_mlp.py as an example of how to use this dataset.

        :param mode: The mode of the dataset (train, validation or test). Test should only be used for the final evaluation!
        :param only_binary_labels: If true, only return binary labels for the set (0 for ordered and 1 for disordered).
            if false, return a vector of labels for each protein (i.e. disordered binding domain, linker, etc).
        :param normalize_features: If true, make sure that features are normalized between 0 and 1.
        :param normalization_means: If the features are normalized and this is not the training set, provide a vector of
            means to use for normalization that reflect the normalized training set.
        :param normalization_stds: If the features are normalized and this is not the training set, provide a vector of
            standard deviations to use for normalization that reflect the normalized training set.
        """
        if mode == DatasetMode.TRAIN:
            self.directory = osp.join('dataset', 'train')
        elif mode == DatasetMode.VALIDATION:
            self.directory = osp.join('dataset', 'validation')
        elif mode == DatasetMode.TEST:
            self.directory = osp.join('dataset', 'test')
        else:
            raise ValueError('Invalid mode')
        self.mode = mode
        self.only_binary_labels = only_binary_labels
        self.proteins = []
        self.label_names = ['is_disordered', 'is_binding', 'is_linker', 'is_protein_binding', 'is_nucleic_acid_binding']
        self.feat_names = None
        self.normalize_features = normalize_features
        self.normalization_means = normalization_means
        self.normalization_stds = normalization_stds

        all_feat_tracker = None
        feats_to_norm = None

        for file in tqdm(os.listdir(self.directory), desc="Loading data from {}".format(self.directory)):
            if not file.endswith('.parquet'):
                continue
            df = pd.read_parquet(osp.join(self.directory, file))
            df['sequence'] = df['sequence'].apply(AAs.index)
            # Move letter to be first column for conveinence
            df.insert(0, 'sequence', df.pop('sequence'))
            # Normalize position to be between 0 and 1
            df['position'] = df['position'] / df['position'].max()

            if self.feat_names is None:
                self.feat_names = df.loc[:, ~df.columns.isin(self.label_names)].columns.values.tolist()
                feats_to_norm = [feat for feat in self.feat_names if feat not in ('position', 'sequence')]

            if normalize_features:
                # Track the features to compute the mean and std after normalization
                if all_feat_tracker is None:
                    all_feat_tracker = self.df_to_tensor(df, feats_to_norm)
                else:
                    all_feat_tracker = torch.cat((all_feat_tracker, self.df_to_tensor(df, feats_to_norm)), dim=0)

            self.proteins.append(df)

        if all_feat_tracker is not None:
            # Min and max of each feature
            self._min = torch.min(all_feat_tracker, dim=0)[0]
            self._max = torch.max(all_feat_tracker, dim=0)[0]
            if self.normalization_means is None or self.normalization_stds is None:
                transformed_features = (all_feat_tracker - self._min) / self._max
                self.normalization_means = transformed_features.mean(dim=0)
                self.normalization_stds = transformed_features.std(dim=0)
                del transformed_features
            del all_feat_tracker

    def df_to_tensor(self, df, cols):
        return torch.from_numpy(df.loc[:, cols].values)

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        df = self.proteins[idx]

        labels = self.df_to_tensor(df, 'is_disordered' if self.only_binary_labels else self.label_names)
        feats = self.df_to_tensor(df, self.feat_names)

        # First two columns are the sequence and position, so normalize only the rest
        if self.normalize_features:
            feats[:, 2:] = (feats[:, 2:] - self._min) / self._max
            # Norm by mean and std only if it is not the training set
            if self.mode != DatasetMode.TRAIN:
                feats[:, 2:] = (feats[:, 2:] - self.normalization_means) / self.normalization_stds

        return feats, labels


if __name__ == '__main__':
    dataset = IdpDataset(DatasetMode.TRAIN)
    print(dataset[0])
