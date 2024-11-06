# Project Overview
This project runs machine learning experiments for protein analysis using classical models and advanced configurations.

# Configuration Parameters
The configuration file allows customization of various aspects of the experiment:

- **random_state**: Controls the random seed.
- **test_size**: Defines the test set size. Use `0.0` to skip the initial train/test split.
- **n_jobs**: Number of processors to use (1 for a single processor, -1 for all processors).
- **use_GPU**: Enables GPU usage if available.
- **nested_cv_outer_splits / nested_cv_inner_splits**: Sets the number of splits for nested cross-validation.
- **make_protein_level_splits**: Specifies if splits should be made at the protein level.
- **per_fold**: If `True`, computes metrics for each fold separately and then averages them.
- **print_debug_messages**: Enables detailed debugging messages.


There are two example configuration files that are used to obtain the article results, `experiment_random_split.py` and `experiment_unseen_protein_split.py`.

## Available Models
You can select classic machine learning models via the `models_to_exec` parameter:

- `KNN`: k-nearest neighbors
- `LR`: logistic regression classifier
- `RF`: random forest classifier


## Embedding Combinations
To handle different embedding representations, the following combinations are offered via `embeddings_combinators`:

- **ConcatEmbeddings**: Concatenates embeddings.
- **AddEmbeddings**: Sums the values of embeddings.
- **MultiplyEmbeddings**: Multiplies the values of embeddings.

# Example Usage
1. **Set up the experiment**: Edit the `EXPERIMENT CONFIGURATION` section in `analysis.py`.
2. **Run the experiment**: Use the following command to run the analysis with the selected machine learning model and, optionally, a name for the experiment log:

   ```bash
   python analysis.py experiment_unseen_protein_split [experiment_name]
    ```

    The first argument is the name of the file with the selected configuration, and the second argument, optional, is an additional name for the experiment logs folder


    If you do not want that python uses buffered output, which is useful when you want to see stdout logs as soon as they are produced, especially when the stdout is written to a file (e.g. nohup), where large buffers are used that may retain the output for a while, run python with "-u" option (unbuffered). For example:

    ```bash
    python -u analysis.py experiment_unseen_protein_split unseen_protein_split
    ```


# Creating the virtual environment

## Python virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Conda

```
conda create -n rapids-24.02 -c rapidsai -c conda-forge -c nvidia cuml=24.02 python=3.10 cuda-version=11.8
```


### Running with GPU
To run on GPU, activate the Conda environment and ensure that the `use_GPU = True` setting is enabled in the configuration file.