import h5py
import pickle

import numpy as np
import pandas as pd


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
