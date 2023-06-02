import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# set seed
SEED = 100
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


class SklearnMultiClassExperiment:

    @staticmethod
    def get_models():

        models = [
            ("MLP", MLPClassifier(random_state=SEED)),
            ("SVM", SVC(kernel="linear", random_state=SEED)),
            ("RF", RandomForestClassifier(random_state=SEED)),
            ("DT", DecisionTreeClassifier(random_state=SEED)),
            ("GBM", GradientBoostingClassifier(random_state=SEED))]

        return models

    @staticmethod
    def score_model(y_true, y_pred):

        subset_acc = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        return subset_acc, balanced_acc

    @staticmethod
    def summarize_scores(accuracy, balanced):

        if not balanced:
            accuracy.drop(["Balanced Accuracy"], axis=1, inplace=True)

        return accuracy.groupby(["Model"]).mean()

    @staticmethod
    @ignore_warnings(category=ConvergenceWarning)
    def evaluate_models(balanced=True, n_splits=10, test_size=0.2):

        models = SklearnMultiClassExperiment.get_models()

        # load data
        if balanced:
            x = pd.read_csv(os.path.join("data", "x_balanced.csv"))
            y = pd.read_csv(os.path.join("data", "y_balanced.csv"))
        else:
            x = pd.read_csv(os.path.join("data", "x_imbalanced.csv"))
            y = pd.read_csv(os.path.join("data", "y_imbalanced.csv"))

        # Set y to our concatenated target variable; the "class" target, rather than the 5 individual labels
        y = y.target

        # create a cross validator
        # We need to use a Stratified Shuffle Split to ensure that the test and train sets have the same set of classes
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)

        accuracy_df = pd.DataFrame({"Model": pd.Series(dtype="str"),
                                    "Accuracy": pd.Series(dtype="float"),
                                    "Balanced Accuracy": pd.Series(dtype="float")})

        for (train_index, test_index) in cv.split(x, y):

            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            for mod_name, mod in models:
                mod.fit(x_train, y_train)
                y_pred = mod.predict(x_test)
                subset_acc, balanced_acc = SklearnMultiClassExperiment.score_model(y_test, y_pred)

                accuracy_df = accuracy_df.append({"Model": mod_name,
                                                  "Accuracy": subset_acc,
                                                  "Balanced Accuracy": balanced_acc}, ignore_index=True)

        return SklearnMultiClassExperiment.summarize_scores(accuracy_df, balanced)
