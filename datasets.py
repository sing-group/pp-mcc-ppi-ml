import h5py
import pickle

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, StratifiedKFold

import functions as fn
from print import pr_cyan, pr_orange

def compute_counts(df, target):
    proteins = set(df['prot1']).union(set(df['prot2']))

    counts_y_true_1 = {protein: 0 for protein in proteins}
    counts_y_true_0 = {protein: 0 for protein in proteins}

    for i, (_, row) in enumerate(df.iterrows()):
        if target[i] == 1:
            counts_y_true_1[row['prot1']] += 1
            counts_y_true_1[row['prot2']] += 1
        else:
            counts_y_true_0[row['prot1']] += 1
            counts_y_true_0[row['prot2']] += 1

    return counts_y_true_1, counts_y_true_0


def create_ppi_dataset(prot1, emb_prot1, prot2, emb_prot2, target):
    df = pd.DataFrame({
        'prot1': prot1,
        'emb_prot1': emb_prot1,
        'prot2': prot2,
        'emb_prot2': emb_prot2,
        'target': target
    })
    df.to_numpy()

    if df['emb_prot1'].shape[0] != df['emb_prot2'].shape[0]:
        raise ValueError("Arrays emb_prot1 and emb_prot2 should have the same length")

    X = df[['emb_prot1', 'emb_prot2', 'prot1', 'prot2']]

    if df['target'].dtype == 'object':
        y = df['target'].map({'True': True, 'False': False}).astype(int)
    else:
        y = df['target'].astype(int)


    return X, np.array(y)


def load_h5_as_df(input_file):
    with h5py.File(input_file, 'r') as h5:
        serialized = h5['dataset'][()]
        dataset = pickle.loads(serialized.tostring())

        return create_ppi_dataset(
            [row[0] for row in dataset],
            [np.array(row[1]) for row in dataset], # Assuming row[1] is an ndarray
            [row[2] for row in dataset],
            [np.array(row[3]) for row in dataset], # Assuming row[3] is an ndarray
            [row[4] for row in dataset] # Assuming row[4] is a boolean
        )

class AbstractDataset(ABC):

    @abstractmethod
    def load(self):
        """Loads the dataset."""
        pass

    @abstractmethod
    def outer_cv(self):
        """Returns an array with the train/test indexes."""
        pass

    @abstractmethod
    def outer_splits(self):
        """Returns the number of train/tests splits returned by outer_cv."""
        pass

    @abstractmethod
    def name(self):
        """Returns the dataset name."""
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def filename(file_path):
        return file_path.split('/')[-1]

    def get_X_train_df(self):
        return self.X_train

    def get_y_train(self):
        return self.y_train


class Dataset(AbstractDataset):

    def __init__(self, input_file, test_size=0.0, outer_cv_splits=3, random_state=2024, make_protein_level_splits=False, print_debug_messages=True):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.test_size = test_size
        self.input_file = input_file
        self.outer_cv_splits = outer_cv_splits
        self.random_state = random_state
        self.make_protein_level_splits = make_protein_level_splits
        self.print_debug_messages = print_debug_messages

    def load(self):
        self.X, self.y = load_h5_as_df(self.input_file)

        if self.test_size > 0.0:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y)

            if self.make_protein_level_splits:
                self.X_train, self.y_train = fn.remove_from_train(self.X_train, self.y_train, self.X_test)

            if self.print_debug_messages:
                pr_orange(f'Size after initial train/test split: {self.X_train.shape[0]} with test_size = {self.test_size}')
        else:
            pr_cyan('INFO: Using all data for external cross-validation as test_size = 0.0')
            self.X_train = self.X
            self.y_train = self.y
    
    def outer_cv(self):
        cv = StratifiedKFold(n_splits=self.outer_cv_splits)

        return cv.split(self.X_train, self.y_train)

    def outer_splits(self):
        return self.outer_cv_splits

    def name(self):
        return self.filename(self.input_file)

    def __str__(self):
        if self.X is None or self.y is None:
            return f"Dataset(input_file = {self.input_file}, test_size = {self.test_size})"
        else:
            return f"Dataset(X = {self.X.shape}, y = {self.y.shape}, input_file = {self.input_file}, test_size = {self.test_size})"


class PartitionedDataset(AbstractDataset):

    def __init__(self, train_file, val_file, debug_size=0.0):
        self.X_train_original = None
        self.y_train_original = None
        self.X_val_original = None
        self.y_val_original = None
        self.X_train = None
        self.y_train = None
        self.train_indices = None
        self.val_indices = None

        self.train_file = train_file
        self.val_file = val_file
        self.debug_size = debug_size

    def load(self):
        self.X_train_original, self.y_train_original = load_h5_as_df(self.train_file)
        self.X_val_original, self.y_val_original = load_h5_as_df(self.val_file)

        if self.debug_size > 0.0:
            debug_test_size =  1 - self.debug_size
            self.X_train_original, _, self.y_train_original, _ = train_test_split(
                self.X_train_original, self.y_train_original, test_size=debug_test_size, random_state=2024, stratify=self.y_train_original)
            self.X_val_original, _, self.y_val_original, _ = train_test_split(
                self.X_val_original, self.y_val_original, test_size=debug_test_size, random_state=2024, stratify=self.y_val_original)

        self.X_train = pd.concat([self.X_train_original, self.X_val_original], axis=0).reset_index(drop=True)
        self.y_train = np.concatenate([self.y_train_original, self.y_val_original], axis=0)

        self.train_indices = np.arange(len(self.X_train_original))
        self.val_indices = np.arange(len(self.X_train_original), len(self.X_train))

    def outer_cv(self):
        return [(self.train_indices, self.val_indices)]
    
    def outer_splits(self):
        return 1
    
    def name(self):
        return f'{self.filename(self.train_file)}_{self.filename(self.val_file)}'
    
    def __str__(self):
        if self.X_train is None or self.y_train is None:
            return f"PartitionedDataset(train_file = {self.train_file}, val_file = {self.val_file})"
        else:
            return f"PartitionedDataset(Train = {self.X_train_original.shape}, Validation = {self.X_val_original.shape}, train_file = {self.train_file}, val_file = {self.val_file})"
