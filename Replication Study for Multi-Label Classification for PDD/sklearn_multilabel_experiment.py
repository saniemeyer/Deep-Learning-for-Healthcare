import pandas as pd
import numpy as np
import random
import os
from helper_functions import hamming_score
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from tensorflow import keras

# set seed
SEED = 100
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


class SklearnMultiLabelExperiment:

    @staticmethod
    def get_models():

        svm = MultiOutputClassifier(SVC(random_state=SEED))
        gbm = MultiOutputClassifier(GradientBoostingClassifier(random_state=SEED))

        models = [
            ("MLP", MLPClassifier(random_state=SEED)),
            ("SVM", svm),
            ("RF", RandomForestClassifier(random_state=SEED)),
            ("DT", DecisionTreeClassifier(random_state=SEED)),
            ("GBM", gbm)]

        return models

    @staticmethod
    def score_model(y_true, y_pred):

        binary_acc = np.mean(keras.metrics.binary_accuracy(np.array(y_true), np.array(y_pred)).numpy())

        y_true = pd.DataFrame(np.rint(y_true), dtype="int")
        y_pred = pd.DataFrame(np.rint(y_pred), dtype="int")

        subset_acc = accuracy_score(y_true, y_pred)
        hamming_acc = hamming_score(y_true, y_pred)

        # Convert y_true and y_pred from multi-label to multi-class
        y_true["target"] = y_true["Insomnia"].astype(str) + y_true["Schizophrenia"].astype(str) + y_true["Vascular Dementia"].astype(str) + y_true["ADHD"].astype(str) + y_true["Bipolar"].astype(str)
        y_true = y_true["target"].astype(int)
        y_pred["target"] = y_pred[0].astype(str) + y_pred[1].astype(str) + y_pred[2].astype(str) + y_pred[3].astype(str) + y_pred[4].astype(str)
        y_pred = y_pred["target"].astype(int)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        return subset_acc, balanced_acc, hamming_acc, binary_acc

    @staticmethod
    def summarize_scores(accuracy, balanced):

        if not balanced:
            accuracy.drop(["Balanced Accuracy"], axis=1, inplace=True)

        return accuracy.groupby(["Model"]).mean()

    @staticmethod
    @ignore_warnings(category=ConvergenceWarning)
    def evaluate_models(balanced=True, n_splits=10, test_size=0.2):

        models = SklearnMultiLabelExperiment.get_models()

        # load data
        if balanced:
            x = pd.read_csv(os.path.join("data", "x_balanced.csv"))
            y = pd.read_csv(os.path.join("data", "y_balanced.csv"))
        else:
            x = pd.read_csv(os.path.join("data", "x_imbalanced.csv"))
            y = pd.read_csv(os.path.join("data", "y_imbalanced.csv"))

        # Drop our concatenated target variable, so that y only consists of the 5 individual labels
        y = y.drop(["target"], axis=1)

        # create a cross validator
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)

        accuracy_df = pd.DataFrame({"Model": pd.Series(dtype="str"),
                                    "Subset Accuracy": pd.Series(dtype="float"),
                                    "Balanced Accuracy": pd.Series(dtype="float"),
                                    "Hamming Score": pd.Series(dtype="float"),
                                    "Binary Accuracy": pd.Series(dtype="float")})

        for (train_index, test_index) in cv.split(x, y):

            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            for mod_name, mod in models:
                mod.fit(x_train, y_train)
                y_pred = mod.predict(x_test)
                subset_acc, balanced_acc, hamming_acc, binary_acc = SklearnMultiLabelExperiment.score_model(y_test, y_pred)

                accuracy_df = accuracy_df.append({"Model": mod_name,
                                                  "Subset Accuracy": subset_acc,
                                                  "Balanced Accuracy": balanced_acc,
                                                  "Hamming Score": hamming_acc,
                                                  "Binary Accuracy": binary_acc}, ignore_index=True)

        return SklearnMultiLabelExperiment.summarize_scores(accuracy_df, balanced)
