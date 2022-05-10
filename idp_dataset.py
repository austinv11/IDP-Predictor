import gc
import os
import re
from enum import Enum
import os.path as osp
from typing import Tuple, List

import h5py
import wget
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel

from utilities import df_to_tensor, sliding_window

_data_path = "dataset"

# Amino acid list based on: https://www.mathworks.com/help/bioinfo/ref/aa2int.html
AAs = {aa: index for index, aa in enumerate("-ARNDCQEGHILKMFPSTWYVBZX")}
INDEX_TO_AA = {index: aa for index, aa in enumerate("-ARNDCQEGHILKMFPSTWYVBZX")}


class DatasetMode(Enum):
    # Training dataset
    TRAIN = 0
    # Validation dataset for parameters optimization
    VALIDATION = 1
    # Test dataset for final evaluation
    TEST = 2


def get_sequence_loader(mode: DatasetMode = DatasetMode.TRAIN, batch_size: int = 450) -> DataLoader:
    """
    Returns a dataset with only sequence data.
    :param mode: The mode of the dataset.
    """
    dataset = T5SequenceDataset.load_remote_embeddings(mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=mode == DatasetMode.TRAIN)


TRAIN_EMBEDDINGS = "https://wcm.box.com/shared/static/uov7y3zhyz35n1b0s5u20wjpukk3epav.bin"
TEST_EMBEDDINGS = "https://wcm.box.com/shared/static/bgdr859zs7xktbhey7gc4taptdmoskpv.bin"
VALIDATION_EMBEDDINGS = "https://wcm.box.com/shared/static/zt1vkljzce4x63fl0fypy1ig2nehpb75.bin"


class T5SequenceDataset(Dataset):

    def __init__(self, mode: DatasetMode = DatasetMode.TRAIN, sequence_length: int = 512, device: str = "cpu",
                 load_data: bool = True, checkpoint_file: str = None):
        if mode == DatasetMode.TRAIN:
            self.directory = osp.join(_data_path, 'train')
        elif mode == DatasetMode.VALIDATION:
            self.directory = osp.join(_data_path, 'validation')
        elif mode == DatasetMode.TEST:
            self.directory = osp.join(_data_path, 'test')
        else:
            raise ValueError('Invalid mode')
        self.mode = mode
        self.sequence_length = sequence_length
        self.filled_keys = []
        self.embeddings_and_labels = []
        self._collapsed_cache = None
        self._size = None
        self._stride = 256  # Stride for sliding window

        if load_data:
            print("Loading tokenizer...")
            self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)

            print("Loading encoder...")
            os.makedirs("Rostlab/prot_t5_xl_uniref50_local", exist_ok=True)
            # Download this version since its only the encoder
            if not osp.exists("Rostlab/prot_t5_xl_uniref50_local/config.json"):
                wget.download('https://rostlab.org/~deepppi/protT5_xl_u50_encOnly_fp16_checkpoint/pytorch_model.bin', "Rostlab/prot_t5_xl_uniref50_local/pytorch_model.bin")
                wget.download('https://rostlab.org/~deepppi/protT5_xl_u50_encOnly_fp16_checkpoint/config.json', "Rostlab/prot_t5_xl_uniref50_local/config.json")
            self.encoder = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50_local', torch_dtype=torch.float16).eval().to(device)
            gc.collect()

            sequences_and_labels = dict()

            if checkpoint_file is not None and osp.exists(checkpoint_file):
                print("Loading checkpoint...")
                T5SequenceDataset.load_embeddings(checkpoint_file, self)

            # Read Files
            for file in tqdm(os.listdir(self.directory), desc="Loading data from {}".format(self.directory)):
                if not file.endswith('.parquet'):
                    continue
                name = osp.basename(file).split('.')[0]
                df = pd.read_parquet(osp.join(self.directory, file))
                # Replace ambiguous amino acids with 'X'
                # Follows https://huggingface.co/Rostlab/prot_t5_xl_bfd preprocessing
                sequence = df['sequence'].apply(lambda x: re.sub(r"[UZOB]", "X", x.upper())).values.tolist()
                label = df_to_tensor(df, 'is_disordered').detach()

                seq_len = len(sequence)
                if seq_len <= self.sequence_length:
                    sequences_and_labels[name] = (sequence, label)
                else:
                    # Split into chunks of size self.sequence_length
                    for i in range(0, seq_len, self.sequence_length):
                        end = min(i + self.sequence_length, seq_len)
                        sequences_and_labels[name + '_' + str(i)] = (sequence[i:end], label[i:end])

            # Make embeddings
            sorted_keys = list(sorted(sequences_and_labels.keys(), key=lambda x: len(sequences_and_labels[x][0]), reverse=True))
            # Batch and make embeddings
            batch_size = 32
            for i in tqdm(list(range(0, len(sorted_keys), batch_size)), desc="Generating embeddings"):
                batch_keys = sorted_keys[i:i + batch_size]
                if any([key in self.filled_keys for key in batch_keys]):
                    continue

                sequences_and_labels_batch = [sequences_and_labels[x] for x in batch_keys]
                # Format sequences according to expectation for the tokenizer
                sequence_batch = [' '.join(x[0]) for x in sequences_and_labels_batch]
                label_batch = [x[1] for x in sequences_and_labels_batch]
                batch_encoding = self.tokenizer.batch_encode_plus(sequence_batch, add_special_tokens=True, padding="longest")
                token_ids = torch.tensor(batch_encoding['input_ids']).to(device)
                attention_mask = torch.tensor(batch_encoding['attention_mask']).to(device)

                with torch.no_grad():
                    embeddings = self.encoder(input_ids=token_ids, attention_mask=attention_mask)
                embeddings = embeddings.last_hidden_state.detach().cpu()

                # Remove padding
                for (key, seq_i, label) in zip(batch_keys, range(len(embeddings)), label_batch):
                    seq_len = (attention_mask[seq_i] == 1).sum()
                    seq_emd = embeddings[seq_i][:seq_len - 1]
                    self.embeddings_and_labels.append((key, seq_emd, label))

            if checkpoint_file is not None:
                print("Saving embeddings...")
                self.save_embeddings(checkpoint_file)

    def _embedding_sliding_windows(self, embeddings: torch.Tensor, labels: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        embedding_windows = list(sliding_window(embeddings.to(torch.float32), self.sequence_length, dimension=0, flatten=False, centered=False, stride=self._stride, no_padding=True))
        labels = list(sliding_window(labels, self.sequence_length, dimension=0, flatten=False, centered=False, fill=2, stride=self._stride, no_padding=True))
        return list(zip(embedding_windows, labels))

    def _collapse_split_embeddings(self) -> Tuple[int, List[Tuple[str, torch.Tensor, torch.Tensor]]]:
        if self._collapsed_cache is not None:
            return self._size, self._collapsed_cache

        collapsed_embeddings_and_labels = []

        curr_split_key = None
        curr_split_data = []
        for (key, embedding, label) in self.embeddings_and_labels:
            if '_' in key:
                orig_key = key.split('_')[0]
                if curr_split_key is not None and curr_split_key != orig_key:
                    concat_embeddings = torch.cat([x[0] for x in curr_split_data])
                    concat_labels = torch.cat([x[1] for x in curr_split_data])
                    collapsed_embeddings_and_labels.append((curr_split_key, concat_embeddings, concat_labels))
                    curr_split_data = []

                curr_split_data.append((embedding, label))
                curr_split_key = orig_key

            else:
                collapsed_embeddings_and_labels.append((key, embedding, label))

        if len(curr_split_data) > 0:
            concat_embeddings = torch.cat([x[0] for x in curr_split_data])
            concat_labels = torch.cat([x[1] for x in curr_split_data])
            collapsed_embeddings_and_labels.append((curr_split_key, concat_embeddings, concat_labels))

        self._collapsed_cache = collapsed_embeddings_and_labels
        del self.embeddings_and_labels
        gc.collect()

        self._size = sum([x[1].shape[0]+(self.sequence_length // self._stride)-1 for x in collapsed_embeddings_and_labels])

        return self._size, collapsed_embeddings_and_labels

    def __len__(self):
        return self._collapse_split_embeddings()[0]

    def __getitem__(self, idx):
        data_size, embeds = self._collapse_split_embeddings()

        seq_index = idx // data_size
        window_index = idx % (self.sequence_length // self._stride)

        key, embedding, label = embeds[seq_index]

        windows = self._embedding_sliding_windows(embedding, label)
        embedding = windows[window_index][0]
        label = windows[window_index][1]
        return embedding, label

    def save_embeddings(self, path: str):
        exists = osp.exists(path)
        with h5py.File(path, "a") as hf:
            if not exists:
                hf.create_dataset("mode", data=self.mode.value)
                hf.create_dataset("sequence_length", data=self.sequence_length)
            for (key, embeddings, labels) in self.embeddings_and_labels:
                if exists and (key + "/labels") in hf:
                    continue
                hf.create_dataset(key + "/embeddings", data=embeddings.numpy(), compression="gzip", chunks=True, compression_opts=9, shuffle=True)
                hf.create_dataset(key + "/labels", data=labels.numpy(), compression="gzip", chunks=True, compression_opts=9, shuffle=True)

    @staticmethod
    def load_embeddings(path: str, partial_dataset: 'T5SequenceDataset' = None) -> 'T5SequenceDataset':
        keys = []
        with h5py.File(path, "r") as hf:
            embeddings_and_labels = []
            mode = DatasetMode(int(hf['mode'][()]))
            sequence_length = int(hf['sequence_length'][()])
            for key in hf.keys():
                if key == "mode" or key == "sequence_length":
                    continue
                embeddings = torch.from_numpy(hf[key + "/embeddings"][:]).detach()
                labels = torch.from_numpy(hf[key + "/labels"][:]).detach()
                embeddings_and_labels.append((key, embeddings, labels))
                keys.append(key)
        if partial_dataset is not None:
            assert partial_dataset.mode == mode and partial_dataset.sequence_length == sequence_length
            dataset = partial_dataset
        else:
            dataset = T5SequenceDataset(mode=mode, sequence_length=sequence_length, load_data=False)
        dataset.embeddings_and_labels = embeddings_and_labels
        dataset.filled_keys = keys
        return dataset

    @staticmethod
    def load_remote_embeddings(mode: DatasetMode) -> 'T5SequenceDataset':
        if mode == DatasetMode.TRAIN:
            url = TRAIN_EMBEDDINGS
            filename = "train_embeddings.h5"
        elif mode == DatasetMode.VALIDATION:
            url = VALIDATION_EMBEDDINGS
            filename = "validation_embeddings.h5"
        elif mode == DatasetMode.TEST:
            url = TEST_EMBEDDINGS
            filename = "test_embeddings.h5"

        if not osp.exists(filename):
            print(f"Downloading {filename} from {url}....")
            wget.download(url, filename)

        print("Reading embeddings...")
        return T5SequenceDataset.load_embeddings(filename)


class IdpDataset(Dataset):

    def __init__(self, mode=DatasetMode.TRAIN, only_binary_labels=True,
                 only_sequences=False,
                 normalize_features=True, normalization_means=None,
                 normalization_stds=None):
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
        :param only_sequences: If true, only return the sequences. If false, return the features as well.
        :param sequences_for_protT5: If true, return just sequences, preprocessed as for protT5.
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
        self.only_sequences = only_sequences

        all_feat_tracker = None
        feats_to_norm = None

        for file in tqdm(os.listdir(self.directory), desc="Loading data from {}".format(self.directory)):
            if not file.endswith('.parquet'):
                continue
            df = pd.read_parquet(osp.join(self.directory, file))
            df['sequence'] = df['sequence'].apply(lambda x: AAs.get(x.upper(), 0))
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
                    all_feat_tracker = df_to_tensor(df, feats_to_norm)
                else:
                    all_feat_tracker = torch.cat((all_feat_tracker, df_to_tensor(df, feats_to_norm)), dim=0)

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

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        df = self.proteins[idx]

        labels = df_to_tensor(df, 'is_disordered' if self.only_binary_labels else self.label_names)

        if self.only_sequences:
            return df_to_tensor(df, 'sequence'), labels

        feats = df_to_tensor(df, self.feat_names)

        # First two columns are the sequence and position, so normalize only the rest
        if self.normalize_features:
            feats[:, 2:] = (feats[:, 2:] - self._min) / self._max
            # Norm by mean and std only if it is not the training set
            if self.mode != DatasetMode.TRAIN:
                feats[:, 2:] = (feats[:, 2:] - self.normalization_means) / self.normalization_stds

        return feats, labels
