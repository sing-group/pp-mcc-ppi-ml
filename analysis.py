import os
import time 
import numpy as np

from datetime import datetime
from sklearn.model_selection import train_test_split

import functions as fn
import scoring as sc
import importlib
import sys

from datasets import load_h5_as_df
from print import pr_cyan, pr_green, pr_red, pr_yellow, pr_orange

def dump_configuration(file, vars):
    with open(file, 'w', encoding='utf-8') as output_file:
        for var in vars:
            output_file.write(f'{var}={eval(var)}\n')

def import_module(module_name):
    try:
        # Import the module dynamically
        module = importlib.import_module(module_name)
        pr_green(f"Successfully imported module: {module_name}")

        # Add module attributes to global namespace
        for attr_name in dir(module):
            if not attr_name.startswith('__'):
                globals()[attr_name] = getattr(module, attr_name)
    except ImportError:
        pr_red(f"Failed to import module: {module_name}")

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Usage: python analysis.py <configuration_module_name> [experiment_name]")
    exit(1)

module_name = os.path.splitext(sys.argv[1])[0]
import_module(module_name)
required_variables = [
    'random_state', 'test_size', 'n_jobs', 'use_GPU', 'datasets', 'models', 
    'embeddings_combinators', 'nested_cv_outer_splits', 'nested_cv_inner_splits', 
    'make_protein_level_splits', 'per_fold', 'print_debug_messages'
]
for var_name in required_variables:
    if var_name not in globals():
        raise ImportError(f"Variable '{var_name}' is not imported from module '{module_name}'")

experiment_name=''
if len(sys.argv) > 2:
    experiment_name = '_' + sys.argv[2]

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logs_dir = f'logs/{timestamp}{experiment_name}'
os.makedirs(logs_dir)

config_log = f'{logs_dir}/_CONFIG.txt'
dump_configuration(config_log, required_variables)

#
# The results* dictionaries will contain dataset names in the first level of keys.
# The second level will contain model names. They can be processed with "fn.results_report".
#
results = {}
results_log = f'{logs_dir}/_RESULTS_ALL.csv'

metrics = sc.DEFAULT_SCORING_DICT
protein_level_metrics = sc.PROTEIN_LEVEL_METRICS

metric_names = list(metrics.keys())
metric_names.extend(protein_level_metrics.keys())
metric_names.extend(['time'])

fn.write_or_append_file(results_log, fn.results_csv_row_header(metric_names))

for dataset in datasets:
    X, y = load_h5_as_df(dataset)
    print(f'Loaded {dataset} dataset: X = {X.shape}, y = {y.shape}')

    for combinator in embeddings_combinators:
        dataset_name = f'{dataset}__{combinator}'
        duplicated_labels = combinator.should_duplicate_labels()

        if test_size > 0.0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y)

            if make_protein_level_splits:
                X_train, y_train = fn.remove_from_train(X_train, y_train, X_test)

            if print_debug_messages:
                pr_orange(f'Size after initial train/test split: {X_train.shape[0]} with test_size = {test_size}')
        else:
            pr_cyan('INFO: Using all data for external cross-validation as test_size = 0.0')
            X_train = X
            y_train = y

        results[dataset_name] = {}

        for model in models:
            model_name = model.name
            print(f'Start new nested CV: {model_name} with {dataset_name}')
            start_time = time.time()

            pred_y_folds, true_y_folds, test_indexes, X_train_dataframes, y_train_folds = fn.do_nested_cv(
                    X_train, y_train, model.clf, model.param_grid, 
                    combinator, make_protein_level_splits,
                    outer_splits=nested_cv_outer_splits,
                    inner_splits=nested_cv_inner_splits,
                    n_jobs=n_jobs,
                    print_debug_messages=print_debug_messages
                )

            total_time = time.time() - start_time
            print(f'End nested CV: {model_name} with {dataset_name}. Execution time: {total_time:.3f}')
            
            results[dataset_name][model_name] = fn.compute_metrics(pred_y_folds, true_y_folds, metrics, per_fold=per_fold)

            protein_level_metrics_calculator = sc.ProteinLevelMetrics(protein_level_metrics, per_fold=per_fold, print_debug_messages=print_debug_messages)
            protein_level_metrics_results = protein_level_metrics_calculator.score(true_y_folds, pred_y_folds, test_indexes, X_train)
            results[dataset_name][model_name].update(protein_level_metrics_results)
            results[dataset_name][model_name].update({'time': total_time})

            fn.show_intermediate_results(
                dataset_name, model_name, results[dataset_name][model_name], results_log, '\tCV results:')
            
            fn.log_folds(pred_y_folds, true_y_folds, test_indexes, X_train, X_train_dataframes, y_train_folds, logs_dir, f'{dataset_name}__{model_name}')


print('\n', '# All interactions #')
fn.cat(results_log)
