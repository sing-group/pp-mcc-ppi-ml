from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from collections import namedtuple
from enum import Enum
from importlib import import_module

from print import pr_red

class Classifier(Enum):
    KNN = "KNN"
    RF = "Random Forest Classifier"
    LR = "Logist Regression"
    
Model = namedtuple('Model', 'clf param_grid name')

def _prepare_models(models_to_exec, use_GPU, model_configurations):
    models = []
    # Iterate over the models to execute
    for model_name in models_to_exec:
        try:
            config = model_configurations[model_name]
        except KeyError:
            pr_red(f"No configuration found for model {model_name}")
            continue

        param_grid = config['param_grid']
        classifier_config = config['cuML'] if use_GPU else config['sklearn']

        module_name = classifier_config['module']
        class_name = classifier_config['class']
        try:
            module = import_module(module_name)
        except ImportError as e:
            pr_red(f"Failed to import module {module_name}: {e}")
            continue
        except AttributeError as e:
            pr_red(f"Failed to get class {class_name} from module {module_name}: {e}")
            continue

        classifier_instance = getattr(module, classifier_config['class'])(**classifier_config['params'])

        gpu_or_cpu = "GPU" if use_GPU else "CPU"
        print(f"Using {model_name} Classifier for {gpu_or_cpu}")
        models.append(Model(classifier_instance, param_grid, model_name))

    return models

def prepare_models(models_to_exec, random_state, use_GPU):
    # Define a dictionary with the configurations of the models
    model_configurations = {
        # Taller 4. Nested CV: kNN + GridSearchCV (baish-line)
        Classifier.KNN.name: {
            'param_grid': {'n_neighbors': [25, 75, 125]},
            'cuML': {'module': 'models.modelsWrapper', 'class': 'cuMLKNNWrapper', 'params': {}},
            'sklearn': {'module': 'sklearn.neighbors', 'class': 'KNeighborsClassifier', 'params': {}}
        },
        # Taller 2. Nested CV: Logistic regression pipeline + GridSearchCV
        Classifier.LR.name: {
            'param_grid': {'C': [0.0001, 1, 10], 'penalty': ['l1', 'l2']},
            'cuML': {'module': 'cuml.linear_model', 'class': 'LogisticRegression', 'params': {'max_iter': 1000, 'solver': 'qn'}},
            'sklearn': {'module': 'sklearn.linear_model', 'class': 'LogisticRegression', 'params': {'max_iter': 1000, 'solver': 'liblinear'}}
        },
        # Taller 3. Nested CV: Random Forest + GridSearchCV
        Classifier.RF.name: {
            'param_grid': {'n_estimators': [100, 200], 'min_samples_leaf': [1, 10, 50], 'max_samples': [0.75, 1.0]},
            'cuML': {'module': 'cuml.ensemble', 'class': 'RandomForestClassifier', 'params': {'random_state': random_state, 'n_streams': 1}},
            'sklearn': {'module': 'sklearn.ensemble', 'class': 'RandomForestClassifier', 'params': {'random_state': random_state}}
        }
    }

    return _prepare_models(models_to_exec, use_GPU, model_configurations)
