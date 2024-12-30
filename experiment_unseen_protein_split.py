import embeddings as em
import models.models as md

from datasets import Dataset

random_state = 2024
test_size = 0 # Use 0.0 to skip the initial train/test split
n_jobs = 1 # None means 1; -1 means all processors
shuffle = False # Use True to run the sanity check
use_GPU = True # Use True to use the GPU
nested_cv_outer_splits = 50
nested_cv_inner_splits = 40
make_protein_level_splits = True
per_fold = False # True means that metris are computed for each fold separately and then averaged
print_debug_messages = False

datasets = [
    Dataset('./Datasets/dataset_esm_yeast.h5', test_size, nested_cv_outer_splits, random_state, make_protein_level_splits, print_debug_messages),
    Dataset('./Datasets/dataset_protbert_yeast.h5', test_size, nested_cv_outer_splits, random_state, make_protein_level_splits, print_debug_messages)
]
models_to_exec = ['KNN', 'LR', 'RF']
models = md.prepare_models(models_to_exec, random_state, use_GPU)
embeddings_combinators = [
    em.ConcatEmbeddings(),
    em.AddEmbeddings(),
    em.MultiplyEmbeddings()
]
