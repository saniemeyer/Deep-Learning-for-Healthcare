import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# set seed
SEED = 100
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


class SklearnSingleLabelExperiment:

    @staticmethod
    def get_models():

        models = [
            ('MLP', MLPClassifier(random_state=SEED)),
            ('SVM', SVC(random_state=SEED)),
            ('RF', RandomForestClassifier(random_state=SEED)),
            ('DT', DecisionTreeClassifier(random_state=SEED)),
            ("GBM", GradientBoostingClassifier(random_state=SEED))]

        return models

    @staticmethod
    def score_model(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def summarize_scores(accuracy):
        return accuracy.groupby(["Model", "Target"]).mean()

    @staticmethod
    @ignore_warnings(category=ConvergenceWarning)
    def evaluate_models(balanced=True, n_splits=10, test_size=0.2, dropout=0.0):

        target_cols = ["Insomnia", "Schizophrenia", "Vascular Dementia", "ADHD", "Bipolar"]
        models = SklearnSingleLabelExperiment.get_models()

        # load data
        if balanced:
            x = pd.read_csv(os.path.join("data", "x_balanced.csv"))
            y = pd.read_csv(os.path.join("data", "y_balanced.csv"))
        else:
            x = pd.read_csv(os.path.join("data", "x_imbalanced.csv"))
            y = pd.read_csv(os.path.join("data", "y_imbalanced.csv"))

        # create a cross validator
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)

        accuracy_df = pd.DataFrame({"Model": pd.Series(dtype="str"),
                                    "Target": pd.Series(dtype="str"),
                                    "Accuracy": pd.Series(dtype="float")})

        for (train_index, test_index) in cv.split(x, y):

            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            for target in target_cols:

                y_train_single = y_train[target]
                y_test_single = y_test[target]

                for mod_name, mod in models:
                    mod.fit(x_train, y_train_single)
                    y_pred = mod.predict(x_test)
                    acc_score = SklearnSingleLabelExperiment.score_model(y_test_single, y_pred)
                    accuracy_df = accuracy_df.append({"Model": mod_name,
                                                      "Target": target,
                                                      "Accuracy": acc_score}, ignore_index=True)

        return SklearnSingleLabelExperiment.summarize_scores(accuracy_df)
