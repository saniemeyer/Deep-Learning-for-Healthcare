import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os
from helper_functions import hamming_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from keras.metrics import binary_accuracy

# set seed
SEED = 100
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


class KerasExperiment:

    @staticmethod
    def get_dnn_multilabel(input_dim, dropout=0.0, exclude_layers=[]):

        mod = tf.keras.models.Sequential()

        mod.add(tf.keras.layers.Dense(15, input_dim=input_dim, activation="relu"))
        mod.add(tf.keras.layers.Dropout(rate=dropout, seed=SEED))

        if 1 not in exclude_layers:
            mod.add(tf.keras.layers.Dense(20, activation="relu"))
            mod.add(tf.keras.layers.Dropout(rate=dropout, seed=SEED))
        if 2 not in exclude_layers:
            mod.add(tf.keras.layers.Dense(20, activation="relu"))
            mod.add(tf.keras.layers.Dropout(rate=dropout, seed=SEED))
        if 3 not in exclude_layers:
            mod.add(tf.keras.layers.Dense(40, activation="relu"))
            mod.add(tf.keras.layers.Dropout(rate=dropout, seed=SEED))

        mod.add(tf.keras.layers.Dense(5, activation="sigmoid"))

        mod.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                    metrics=["accuracy", "binary_accuracy"])

        return mod

    @staticmethod
    def get_dnn_singlelabel(input_dim, dropout=0.0, exclude_layers=[]):

        mod = tf.keras.models.Sequential()
        mod.add(tf.keras.layers.Dense(15, input_dim=input_dim, activation="relu"))
        mod.add(tf.keras.layers.Dropout(rate=dropout, seed=SEED))

        if 1 not in exclude_layers:
            mod.add(tf.keras.layers.Dense(20, activation="relu"))
            mod.add(tf.keras.layers.Dropout(rate=dropout, seed=SEED))
        if 2 not in exclude_layers:
            mod.add(tf.keras.layers.Dense(40, activation="relu"))
            mod.add(tf.keras.layers.Dropout(rate=dropout, seed=SEED))
        if 3 not in exclude_layers:
            mod.add(tf.keras.layers.Dense(50, activation="relu"))
            mod.add(tf.keras.layers.Dropout(rate=dropout, seed=SEED))

        mod.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        mod.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                    metrics=["accuracy"])

        return mod

    @staticmethod
    def score_model_multilabel(y_true, y_pred):

        y_true = pd.DataFrame(np.rint(y_true), dtype="int")
        y_pred = pd.DataFrame(np.rint(y_pred), dtype="int")

        subset_acc = accuracy_score(y_true, y_pred)
        binary_acc = np.mean(binary_accuracy(y_true, y_pred).numpy())
        hamming_acc = hamming_score(y_true, y_pred)

        # Convert y_true and y_pred from multi-label to multi-class
        # Note: We know that this is an invalid technique, but we are trying to replicate the methodology of the paper
        y_true["target"] = y_true["Insomnia"].astype(str) + y_true["Schizophrenia"].astype(str) + y_true["Vascular Dementia"].astype(str) + y_true["ADHD"].astype(str) + y_true["Bipolar"].astype(str)
        y_true = y_true["target"].astype(int)
        y_pred["target"] = y_pred[0].astype(str) + y_pred[1].astype(str) + y_pred[2].astype(str) + y_pred[3].astype(str) + y_pred[4].astype(str)
        y_pred = y_pred["target"].astype(int)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        return subset_acc, balanced_acc, hamming_acc, binary_acc

    @staticmethod
    def score_model_singlelabel(y_true, y_pred):
        y_true = np.rint(y_true)
        y_pred = np.rint(y_pred)
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def summarize_scores_multilabel(accuracy, history, balanced):

        if not balanced:
            accuracy.drop(["Balanced Accuracy"], axis=1, inplace=True)

        return accuracy.groupby(["Model"]).mean(), history.groupby("Epoch").mean().reset_index()

    @staticmethod
    def summarize_scores_singlelabel(accuracy):
        return accuracy.groupby(["Target"]).mean()

    @staticmethod
    def evaluate_multilabel_model(balanced=True, n_splits=10, test_size=0.2, dropout=0.0,
                                  exclude_layers=[]):
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
        history = pd.DataFrame()

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]

        for (train_index, test_index) in cv.split(x, y):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            mod = KerasExperiment.get_dnn_multilabel(x.shape[1], dropout=dropout, exclude_layers=exclude_layers)
            history_last_run = mod.fit(x_train, y_train,
                                       epochs=40,
                                       validation_data=(x_test, y_test),
                                       callbacks=callbacks,
                                       verbose=0)

            y_pred = mod.predict(x_test, verbose=0)
            subset_acc, balanced_acc, hamming_acc, binary_acc = KerasExperiment.score_model_multilabel(y_test, y_pred)
            accuracy_df = accuracy_df.append({"Model": "Keras Multi-Label",
                                              "Subset Accuracy": subset_acc,
                                              "Balanced Accuracy": balanced_acc,
                                              "Hamming Score": hamming_acc,
                                              "Binary Accuracy": binary_acc}, ignore_index=True)
            history_last_run = pd.DataFrame(history_last_run.history)
            history_last_run["Epoch"] = history_last_run.index.values
            history = pd.concat([history, history_last_run], axis=0, ignore_index=True)

        return KerasExperiment.summarize_scores_multilabel(accuracy_df, history, balanced)

    @staticmethod
    def evaluate_singlelabel_model(balanced=True, n_splits=10, test_size=0.2, dropout=0.0, exclude_layers=[]):

        target_cols = ["Insomnia", "Schizophrenia", "Vascular Dementia", "ADHD", "Bipolar"]

        # load data
        if balanced:
            x = pd.read_csv(os.path.join("data", "x_balanced.csv"))
            y = pd.read_csv(os.path.join("data", "y_balanced.csv"))
        else:
            x = pd.read_csv(os.path.join("data", "x_imbalanced.csv"))
            y = pd.read_csv(os.path.join("data", "y_imbalanced.csv"))

        y = y.drop(["target"], axis=1)

        # create a cross validator
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)

        accuracy_df = pd.DataFrame({"Target": pd.Series(dtype="str"),
                                    "Accuracy": pd.Series(dtype="float")})

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]

        for (train_index, test_index) in cv.split(x, y):

            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            for target in target_cols:
                y_train_single = y_train[target]
                y_test_single = y_test[target]

                mod = KerasExperiment.get_dnn_singlelabel(x.shape[1], dropout=dropout, exclude_layers=exclude_layers)
                mod.fit(x_train, y_train_single,
                        validation_data=(x_test, y_test_single),
                        callbacks=callbacks,
                        verbose=0)

                y_pred = mod.predict(x_test, verbose=0)
                acc_score = KerasExperiment.score_model_singlelabel(y_test_single, y_pred)

                accuracy_df = accuracy_df.append({"Target": target,
                                                  "Accuracy": acc_score}, ignore_index=True)

        return KerasExperiment.summarize_scores_singlelabel(accuracy_df)

    # Ablations: Grid Search excluding layers and varying dropout: Balanced Multi-Label DNN
    @staticmethod
    def run_ablations(experiment_name="", balanced=True, n_splits=30):

        if balanced:
            accuracy_type = "Balanced Accuracy"
        else:
            accuracy_type = "Binary Accuracy"

        param_grid = {"dropout": [0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
                      "exclude_layers": [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]}

        best_params = {"best_accuracy": 0}

        run_history = pd.DataFrame({accuracy_type: pd.Series(dtype='float'),
                                    "Dropout": pd.Series(dtype="str"),
                                    "Exclude Layers": pd.Series(dtype="str")})

        for dropout in param_grid["dropout"]:
            for exclude_layers in param_grid["exclude_layers"]:

                results, history = KerasExperiment.evaluate_multilabel_model(balanced=balanced,
                                                                             n_splits=n_splits,
                                                                             test_size=0.3,
                                                                             dropout=dropout,
                                                                             exclude_layers=exclude_layers)

                accuracy = results.iloc[0][accuracy_type]
                strlayers = ', '.join(str(s) for s in exclude_layers)

                if accuracy > best_params["best_accuracy"]:
                    best_params["best_accuracy"] = accuracy
                    best_params["dropout"] = dropout
                    best_params["exclude_layers"] = strlayers

                run_history = run_history.append({accuracy_type: accuracy,
                                                  "Dropout": dropout,
                                                  "Exclude Layers": strlayers}, ignore_index=True)

        print("-" * 80)
        print(f"Experiment: {experiment_name}")
        print(f"Best Model:")
        print(f"  {accuracy_type}: {best_params['best_accuracy']}")
        print(f"  Dropout: {best_params['dropout']}")
        print(f"  Exclude Layers: {best_params['exclude_layers']}")

        print("-" * 80)
        print(f"Grid-Search Run History:")
        display(run_history)