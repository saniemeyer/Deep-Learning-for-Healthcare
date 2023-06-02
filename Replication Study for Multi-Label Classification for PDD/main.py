import time
import warnings
from preprocessing import Preprocessing
from helper_functions import print_and_save_results
from sklearn_multilabel_experiment import SklearnMultiLabelExperiment
from sklearn_multiclass_experiment import SklearnMultiClassExperiment
from sklearn_singlelabel_experiment import SklearnSingleLabelExperiment


def warn(*args, **kwargs):
    pass

warnings.filterwarnings('ignore')
warnings.warn = warn

Preprocessing.create_dataset()

n_splits = 100
test_size = 0.2

# Run Imbalanced Multi-Label Experiment for ML Models
start_time = time.time()
results = SklearnMultiLabelExperiment.evaluate_models(balanced=False, n_splits=n_splits, test_size=test_size)
print_and_save_results(experiment="Imbalanced Multi-Label", results=results, start_time=start_time, overwrite=True)


# Run Imbalanced Multi-Class Experiment for ML Models
start_time = time.time()
results = SklearnMultiClassExperiment.evaluate_models(balanced=False, n_splits=n_splits, test_size=test_size)
print_and_save_results(experiment="Imbalanced Multi-Class", results=results, start_time=start_time)

# Run Balanced Multi-Label Experiment for ML Models
start_time = time.time()
results = SklearnMultiLabelExperiment.evaluate_models(balanced=True, n_splits=n_splits, test_size=test_size)
print_and_save_results(experiment="Balanced Multi-Label", results=results, start_time=start_time)


# Run Balanced Multi-Class Experiment for ML Models
start_time = time.time()
results = SklearnMultiClassExperiment.evaluate_models(balanced=True, n_splits=n_splits, test_size=test_size)
print_and_save_results(experiment="Balanced Multi-Class", results=results, start_time=start_time)

# Run Imbalanced Single-Label Experiment for ML Models
start_time = time.time()
results = SklearnSingleLabelExperiment.evaluate_models(balanced=False, n_splits=n_splits, test_size=test_size)
print_and_save_results(experiment="Imbalanced Single-Label", results=results, start_time=start_time)

# Run Balanced Single-Label Experiment for ML Models
start_time = time.time()
results = SklearnSingleLabelExperiment.evaluate_models(balanced=True, n_splits=n_splits, test_size=test_size)
print_and_save_results(experiment="Balanced Single-Label", results=results, start_time=start_time)




