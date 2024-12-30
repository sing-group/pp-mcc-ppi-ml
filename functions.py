import os
import math
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from slugify import slugify

from print import pr_orange
from datasets import compute_counts


def duplicate_labels(arr):
    result = []
    result.extend(arr)
    result.extend(arr)

    return np.array(result)


def results_csv_row_header(metrics, sep=';'):
    return 'dataset;combination;model;' + sep.join(metrics)


def exclude_std_keys(keys):
    return list(filter(lambda k: not k.endswith('_std'), keys))


def results_csv_row(metrics, dataset_name, dataset_combination, model, sep=';'):
    row = [dataset_name, dataset_combination, model]
    for metric in exclude_std_keys(metrics.keys()):
        mean = metrics[metric]
        row.append(f'{mean:.3f}')

    return sep.join(row)


def show_metrics(results, indent=1):
    for metric in exclude_std_keys(results.keys()):
        mean = results[metric]
        indent_str = '\t' * indent
        
        metric_std = f'{metric}_std'
        if metric_std in results:
            std = results[metric_std]
            print(f'{indent_str}{metric} = {mean:.3f} ±{std:.3f}')
        else:
            print(f'{indent_str}{metric} = {mean:.3f}')


def cat(path):
    with open(path, 'r', encoding = 'utf-8') as file:
        for line in file:
            print(line[:-1])


def write_or_append_file(filename, content, add_new_line=True):
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode, encoding='utf-8') as file:
        file.write(content)
        if add_new_line:
            file.write('\n')


def show_intermediate_results(dataset, model, results, log_file, prompt):
    print(prompt)
    show_metrics(results, indent=2)
    print('*' * 80)

    dataset_name = dataset.split('__')[0]
    dataset_combination = dataset.split('__')[1]
    write_or_append_file(log_file, results_csv_row(results, dataset_name, dataset_combination, model))


def extract_protein_set(df):
    prot1 = df['prot1'].values
    prot2 = df['prot2'].values

    return set(prot1) | set(prot2)


def remove_from_train(train_partition, train_partition_y, test_partition):
    """
    Removes protein interactions from the training set if any protein interactor appear in the test set.

    Parameters:
    - train_partition (pandas.DataFrame): A DataFrame containing the protein interactions for the training set, where 
        each row represents an interaction, and includes columns for the proteins involved, e.g., 'prot1', 'prot2', 
        and other interaction-specific features.
    - train_partition_y (pandas.Series or pandas.DataFrame): The labels or target values associated with each interaction 
    in `train_partition`. The order of `train_partition_y` must correspond to the rows in `train_partition`.
    - test_partition (pandas.DataFrame): A DataFrame containing the protein interactions for the test set, structured 
    similarly to `train_partition`. This set is used to identify proteins that should be excluded from the training set 
    to prevent data leakage.

    Returns:
    - (tuple): A tuple containing two elements:
        - filtered_dataframe_2 (pandas.DataFrame): The filtered `train_partition` DataFrame after removing interactions involving proteins found in `test_partition`.
        - filtered_dataframe_y_2 (pandas.Series or pandas.DataFrame): The corresponding labels for `filtered_dataframe_2`, maintaining the original order and indexing.

    Note:
    - It is assumed that the `test_partition` DataFrame contains at least one column that lists proteins involved in interactions, which is used to identify the proteins to be excluded from the training set.
    - This function does not modify the input DataFrames in-place; instead, it returns new, filtered DataFrames.
    """
    test_proteins = extract_protein_set(test_partition)

    mask = ~train_partition['prot1'].isin(test_proteins)
    filtered_dataframe = train_partition[mask]
    filtered_dataframe_y = train_partition_y[mask]

    mask = ~filtered_dataframe['prot2'].isin(test_proteins)
    filtered_dataframe_2 = filtered_dataframe[mask]
    filtered_dataframe_y_2 = filtered_dataframe_y[mask]

    return filtered_dataframe_2, filtered_dataframe_y_2


def filter_train_indexes(train_idx, X_tr, fold_proteins):
    """
    Filters indices of training samples by excluding those involving specified proteins.

    This function iterates over a given list of indices (`train_idx`) that correspond to samples in a 
    DataFrame (`X_tr`). It excludes any index corresponding to a sample where either interacting 
    protein is found in the specified `fold_proteins` list. This is used to create training folds that
    do not contain interactions involving certain proteins.

    Parameters:
    - train_idx (list or array-like): A list or array-like object containing indices of samples from 
    `X_tr` that are considered for filtering.
    - X_tr (pandas.DataFrame): A DataFrame where each row corresponds to a protein interaction, and 
    includes at least the columns 'prot1' and 'prot2' to denote the interacting proteins.
    - fold_proteins (set or list): A collection of protein identifiers that are to be excluded from 
    the training set. Any interaction involving at least one protein from this collection will be excluded.

    Returns:
    - filtered_indexes (list): A list of indices from `train_idx` that passed the filtering criteria, i.e., 
    interactions that do not involve any proteins listed in `fold_proteins`.

    Note:
    - This function adjusts indices based on the size of `X_tr` to handle scenarios where `train_idx` might 
    contain indices that loop over `X_tr` more than once (e.g., in augmented datasets).
    - It is important to ensure that the protein identifiers in `fold_proteins` match those used in the 
    'prot1' and 'prot2' columns of `X_tr`.
    """
    filtered_indexes = []
    for index in train_idx:
        original_index = index
        if original_index >= X_tr.shape[0]:
            index = index - X_tr.shape[0]
        
        if not X_tr.iloc[index]['prot1'] in fold_proteins and not X_tr.iloc[index]['prot2'] in fold_proteins:
            if original_index >= X_tr.shape[0]:
                index = index + X_tr.shape[0]

            filtered_indexes.append(index)
    
    return filtered_indexes


def do_nested_cv(
    dataset,
    model,
    param_grid,
    combinator,
    make_protein_level_splits,
    inner_splits=5,
    inner_refit_scoring='f1',
    n_jobs=-1,
    print_debug_messages=False
):
    """
    Perform nested cross-validation for model selection and evaluation, with special considerations 
    for protein-protein interaction data.

    This function facilitates nested cross-validation, allowing for both model selection via hyperparameter
    tuning within the inner loop and model evaluation in the outer loop. It is specifically tailored for 
    datasets involving protein-protein interactions, offering options to make protein-level splits and to handle 
    duplicated labels resulting from data augmentation techniques like adding inverted interactions.

    Parameters
    - dataset (AbstractDataset): The datsaet with the protein interaction data.
    - model: The machine learning estimator object from scikit-learn or a compatible library.
    - param_grid (dict or list of dicts): Dictionary with parameters names (str) as keys and lists of parameter
    settings to try as values, or a list of such dictionaries, each corresponding to a different search space.
    - combinator: An object responsible for preprocessing the feature matrix X and perform embedding extraction 
    and data augmentation.
    - make_protein_level_splits (bool): Flag indicating whether to ensure that proteins are not split across training
    and testing sets, to mimic a more realistic scenario where the model is tested on entirely unseen proteins.
    - inner_splits (int, optional): Number of folds for the inner cross-validation loop, used for hyperparameter
    tuning. Defaults to 5.
    - inner_refit_scoring (str, optional): Scoring metric for refitting the model on the entire training set within
    the inner cross-validation loop. Defaults to 'f1'.
    - n_jobs (int, optional): Number of jobs to run in parallel during the GridSearchCV phase. -1 means using all 
    processors. Defaults to -1.
    - print_debug_messages (bool, optional): Flag to enable printing of debug messages during execution. Useful for 
    tracking progress and debugging.

    Returns:
    - tuple: A tuple containing three lists:
        - pred_y_folds (list of numpy arrays): Predictions for each outer fold.
        - true_y_folds (list of numpy arrays): True target values for each outer fold.
        - test_indexes (list of array-like): Indices of the test sets for each outer fold.

    Notes:
    - This function is particularly suitable for datasets where it's crucial to maintain the integrity of biological
    entities (e.g., proteins) across folds, thereby preventing data leakage and ensuring that the model's performance
    is evaluated on entirely unseen entities.
    - The function utilizes a custom process for creating training and testing splits, especially when 
    `make_protein_level_splits` is True, to accommodate the unique requirements of protein-protein interaction data.
    - The `combinator` object plays a critical role in preprocessing the data, potentially handling tasks such as 
    embedding extraction and the management of duplicated labels due to data augmentation techniques.

    """
    inner_CV = StratifiedGroupKFold(n_splits=inner_splits)

    pred_y_folds = []
    true_y_folds = []
    test_indexes = []
    X_train_dataframes = []
    y_train_folds = []

    iteration = 1
    
    for train_index, test_index in dataset.outer_cv():
        print(f'\tStarting outer CV iteration {iteration} out of {dataset.outer_splits()}')
        iteration = iteration + 1

        X = dataset.get_X_train_df()
        y = dataset.get_y_train()

        X_tr, X_tt = X.iloc[train_index], X.iloc[test_index]
        y_tr, y_tt = y[train_index], y[test_index]

        if make_protein_level_splits:
            X_tr, y_tr = remove_from_train(X_tr, y_tr, X_tt)

        if print_debug_messages:
            pr_orange(f'\t\tOuter CV train interactions: {X_tr.shape[0]} with cv = {dataset.outer_splits()}')

        X_tr_emb, groups = combinator.unpack_embeddings(X_tr)

        if combinator.should_duplicate_labels():
            y_tr = duplicate_labels(y_tr)

        inner_cv_split_indices = list(inner_CV.split(X_tr_emb, y_tr, groups=groups))

        inner_cv_split_indices_filtered = []

        for train_idx, test_idx in inner_cv_split_indices:
            # 1. Fix testing folds
            if combinator.should_duplicate_labels():
                # When reversed interactions are added by a combinator, those interactions are
                # removed from the test folds to make those tests like the original data.
                filtered_test_idx = [idx for idx in test_idx if idx < X_tr.shape[0]]
            else:
                filtered_test_idx = test_idx
            
            # 2. Update training partitions
            if make_protein_level_splits:
                # When making splits at protein level, inner CV folds should be also modified so
                # that each training set (train_idx) does not contain proteins from the corresponding
                # test fold (test_idx)

                fold_proteins = extract_protein_set(X_tr.iloc[filtered_test_idx])
                train_idx_filtered = filter_train_indexes(train_idx, X_tr, fold_proteins)

                if print_debug_messages:
                    pr_orange(f'\t\t\tInner CV fold train interactions: {len(train_idx_filtered)} with cv = {inner_splits}')
            else:
                train_idx_filtered = train_idx

            inner_cv_split_indices_filtered.append((train_idx_filtered, filtered_test_idx))

        clf = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=inner_cv_split_indices_filtered, refit=inner_refit_scoring, n_jobs=n_jobs)
        clf.fit(X_tr_emb, y_tr)

        X_tt_emb, _ = combinator.unpack_embeddings_test(X_tt)
        pred = clf.predict(X_tt_emb)
        
        X_train_dataframes.append(X_tr)
        y_train_folds.append(y_tr)
        pred_y_folds.append(pred)
        true_y_folds.append(y_tt)
        test_indexes.append(test_index)

    return pred_y_folds, true_y_folds, test_indexes, X_train_dataframes, y_train_folds


def compute_metrics(
    pred_y_folds,
    true_y_folds,
    scoring,
    per_fold=True
):
    """
    Compute evaluation metrics for model predictions against true values, optionally on a per-fold basis.

    This function calculates various evaluation metrics for given predicted and true values,
    which are organized by cross-validation folds. It supports computing metrics for each fold
    individually and then aggregating them, or computing metrics across all data ignoring fold divisions.

    Args:
        pred_y_folds (list of lists): Predicted values for each fold in cross-validation.
        true_y_folds (list of lists): True values for each fold in cross-validation.
        scoring (dict): A dictionary where keys are metric names and values are callable functions that
                        calculate the metric given true and predicted values.
        per_fold (bool, optional): If True, metrics are computed for each fold and then aggregated. 
                                   If False, metrics are computed across all data. Defaults to True.

    Returns:
        dict: A dictionary of computed evaluation metrics. If `per_fold` is True, each metric will also
              include its standard deviation across folds with the key format `{metric_name}_std`.

    Notes:
        - The length of each fold in `pred_y_folds` and `true_y_folds` must match.
        - The scoring functions in the `scoring` dictionary should accept two arguments: the true values
          and the predicted values, in that order.

    Raises:
        ValueError: If there's a mismatch in the length of folds or other input inconsistencies.
        TypeError: If scoring functions are not callable or return invalid values.

    Examples:
        >>> from sklearn.metrics import accuracy_score, precision_score
        >>> pred_y_folds = [[0, 1, 0], [1, 0, 0]]
        >>> true_y_folds = [[1, 0, 0], [0, 1, 0]]
        >>> scoring = {'accuracy': accuracy_score, 'precision': precision_score}
        >>> compute_metrics(pred_y_folds, true_y_folds, scoring)
        {'accuracy': 0.5, 'accuracy_std': 0.05, 'precision': 0.5, 'precision_std': 0.05}
    """
    results = {}

    if per_fold:
        results_folds = {}
        for y_pred, y_true in zip(pred_y_folds, true_y_folds):
            for metric in scoring:
                if not metric in results_folds:
                    results_folds[metric] = []

                results_folds[metric].append(scoring[metric](y_true, y_pred))
        
        for metric, values in results_folds.items():
            results[metric] = np.mean(values)
            results[f'{metric}_std'] = np.std(values)

    else:
        y_pred = [item for sublist in pred_y_folds for item in sublist]
        y_true = [item for sublist in true_y_folds for item in sublist]

        for metric in scoring:
            try:
                results[metric] = scoring[metric](y_true, y_pred)
            except (ValueError, ZeroDivisionError):
                results[metric] = math.nan
    
    return results


def extract_proteins(interaction_index, X):
    prot1 = X['prot1'].values[interaction_index]
    prot2 = X['prot2'].values[interaction_index]

    return prot1, prot2


def log_folds(
    pred_y_folds,
    true_y_folds,
    test_indexes,
    X_train,
    X_train_dataframes,
    y_train,
    logs_dir,
    log_filename
):
    log_filename = slugify(log_filename.replace('__', 'PLACEHOLDER'), separator='_', lowercase=False)
    log_filename = log_filename.replace('PLACEHOLDER', '__')

    y_pred = [item for sublist in pred_y_folds for item in sublist]
    y_true = [item for sublist in true_y_folds for item in sublist]
    indexes = [item for sublist in test_indexes for item in sublist]
    folds = []

    # Precompute protein counts in each train partition
    counts_y_true = {}
    fold = 0
    for x_tr, y_tr in zip(X_train_dataframes, y_train):
        counts_y_true[fold] = compute_counts(x_tr, y_tr)
        fold = fold + 1

    for i, _ in enumerate(pred_y_folds):
        folds.extend([i] * len(pred_y_folds[i]))

    with open(f'{logs_dir}/{log_filename}', 'w', encoding='utf-8') as log_file:
        log_file.write('index,fold,prot_a,prot_b,prot_a_total_int,prot_a_pos_int,prot_b_total_int,prot_b_pos_int,y_true,y_pred\n')
        for i, current_index in enumerate(indexes):
            prot1, prot2 = extract_proteins(current_index, X_train)
            counts_y_true_1, counts_y_true_0 = counts_y_true[folds[i]]


            # prot1 and prot2 are the test proteíns and thus it is possible that a given protein that only appears in one interaction
            # will not be present in any train set and therefore the following counts will not be available for it

            prot_a_pos_int = counts_y_true_1.get(prot1, math.nan)
            prot_b_pos_int = counts_y_true_1.get(prot2, math.nan)
            prot_a_total_int = prot_a_pos_int + counts_y_true_0.get(prot1, math.nan)
            prot_b_total_int = prot_b_pos_int + counts_y_true_0.get(prot2, math.nan)

            log_file.write(f'{current_index},{folds[i]},{prot1},{prot2},{prot_a_total_int},{prot_a_pos_int},{prot_b_total_int},{prot_b_pos_int},{y_true[i]},{y_pred[i]}\n')
