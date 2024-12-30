import math
import warnings
import pandas as pd
import numpy as np

from collections import defaultdict
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, matthews_corrcoef
from functions import extract_proteins, extract_protein_set
from print import pr_orange

# balanced_accuracy With adjusted=True = Youden
def youden(y_true, y_pred):
     return balanced_accuracy_score(y_true, y_pred, adjusted=True)

def specificity(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

DEFAULT_SCORING_DICT = {
    'recall': recall_score,
    'precision': precision_score,
    'f1': f1_score,
    'accuracy': accuracy_score, 
    'specificity': specificity,
    'matthews_corrcoef': matthews_corrcoef,
    'youden': youden
}

PROTEIN_LEVEL_METRICS = {
    'pp_matthews_corrcoef': matthews_corrcoef,
    'pp_youden': youden
}

def weighted_avg_std(values, weights):
    """
    Compute the weighted average and standard deviation of a given set of values and weights.
    
    Parameters:
    values (array-like): The data values.
    weights (array-like): The weights corresponding to the data values.
    
    Returns:
        tuple(float, float): The weighted average and standard deviation.
    """
    if len(values) != 0:
        average = np.average(values, weights=weights)
        variance = np.average((values-average)**2, weights=weights)
        return average, np.sqrt(variance)
    else:
        return np.nan, np.nan


class ProteinLevelMetrics:
    """
    A class for evaluating protein interaction metrics based on prediction and true data. 

    This class takes scoring functions as an input to evaluate various metrics on protein interactions,
    considering true values and predictions. It provides functionalities to score individual proteins,
    calculate weighted averages for metrics, and aggregate results for all proteins.

    Intances are created passing a scoring dictionary where keys are metric names and values are sklearn 
    functions to compute these metrics based on y_true and y_pred.
    """
     
    def __init__(self, scoring, per_fold=True, print_debug_messages=False):
        self.scoring = scoring
        self.per_fold = per_fold
        self.print_debug_messages = print_debug_messages
    
    @staticmethod
    def _create_dataframe(y_true, y_pred, test_indexes, X_train):
        prot1_list = []
        prot2_list = []
        for current_index in test_indexes:
            prot1, prot2 = extract_proteins(current_index, X_train)
            prot1_list.append(prot1)
            prot2_list.append(prot2)
        
        df = pd.DataFrame({
            'indexes': test_indexes,
            'y_true': y_true,
            'y_pred': y_pred,
            'prot1': prot1_list,
            'prot2': prot2_list
        })

        return df

    def _score_protein(self, protein_name, df, column_protein='prot1'):
        if column_protein == 'prot1':
            interactions = df[(df['prot1'] == protein_name)]
        elif column_protein == 'both':
            interactions = df[(df['prot1'] == protein_name) | (df['prot2'] == protein_name)]
            if self.print_debug_messages:
                prot1_interactions = df[df['prot1'] == protein_name].shape[0]
                prot2_interactions = df[df['prot2'] == protein_name].shape[0]
                pr_orange(f'[ProteinLevelMetrics] Protein {protein_name}: prot1 interactions = {prot1_interactions}, prot2 interactions = {prot2_interactions}')
        else:
            raise ValueError("Invalid value for column_protein. Use 'prot1' or 'both'.")
        
        protein_metrics = {"num_interactions": interactions.shape[0]}

        y_true = interactions['y_true']
        y_pred = interactions['y_pred']

        if len(y_pred) > 0:
            for metric in self.scoring.keys():
                try:
                    if sum(y_true == 1) > 0 and sum(y_true == 0) > 0:
                        # sklearn returns 0 when MCC cannot be calculated, thus we need to explicitly
                        # skip those cases to avoid adding false 0's to the computations.
                        # FIXME: other metrics may work even if this condition is not true and thus
                        # legitimate values will be ommitted. Currently, we accept it as we are only
                        # using MCC and Youden, for which this condition is neccessary.

                        score = self.scoring[metric](y_true, y_pred)
                        if math.isfinite(score):
                            protein_metrics[metric] = score
                except ZeroDivisionError:
                    pass

        return protein_metrics

    def _score_all_proteins(self, df):
        protein_metrics = {}
        protein_set = sorted(extract_protein_set(df))

        for protein in protein_set:
            protein_metrics[protein] = self._score_protein(protein, df, column_protein='both')

        return protein_metrics

    def _get_weighted_average(self, protein_metrics, metric):
        total_interactions = sum(protein_metrics[protein_name]['num_interactions'] for protein_name in protein_metrics if metric in protein_metrics[protein_name])

        count = 0
        values = []
        weights = []
        for protein_name in protein_metrics:
            if metric in protein_metrics[protein_name]:
                count = count + 1
                values.append(protein_metrics[protein_name][metric])
                weights.append(protein_metrics[protein_name]['num_interactions'] / total_interactions)

        if self.print_debug_messages:
            pr_orange(f'\t[ProteinLevelMetrics] {metric} computed for {count} proteins')

        return weighted_avg_std(values, weights)

    def _get_weighted_averages(self, weighted_averages, protein_metrics):
        for metric in self.scoring.keys():
            avg, std = self._get_weighted_average(protein_metrics, metric)

            if not avg is np.nan and not std is np.nan:
                weighted_averages[metric].append(avg)
                weighted_averages[f'{metric}_std'].append(std)

    def _average_folds(self, weighted_averages):
        for key in weighted_averages:
            if self.per_fold:
                weighted_averages[key] = np.mean(weighted_averages[key])
            else:
                weighted_averages[key] = weighted_averages[key][0]

    def score(self, y_true, y_pred, test_indexes, X_train):
        """
        Scores the protein interactions based on the provided true and predicted interaction data. The 
        weighted averages across proteins are computed on a fold basis and then averaged again to obtain
        the final average.

        Args:
            y_true (list of lists): The true labels for the protein interactions for each fold.
            y_pred (list of lists): The predicted labels for the protein interactions for each fold.
            test_indexes (list of lists): The indexes of the test data for each fold.
            X_train (DataFrame): The training dataset, used for extracting protein names.

        Returns:
            - A dictionary with weighted averages for each metric across all proteins and across folds.
          """

        if not self.per_fold:
            y_pred = [[item for sublist in y_pred for item in sublist]]
            y_true = [[item for sublist in y_true for item in sublist]]
            test_indexes = [[item for sublist in test_indexes for item in sublist]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            weighted_averages = defaultdict(list)
            for i in range(len(y_true)):
                df = ProteinLevelMetrics._create_dataframe(y_true[i], y_pred[i], test_indexes[i], X_train)
                protein_metrics = self._score_all_proteins(df)
                self._get_weighted_averages(weighted_averages, protein_metrics)

            self._average_folds(weighted_averages)

        return weighted_averages
