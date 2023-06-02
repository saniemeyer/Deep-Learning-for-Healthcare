import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def print_and_save_results(experiment, results, start_time, overwrite=False):
    print(f"Experiment: {experiment}")
    execution_stats = f"Execution Time (Mins): {round(((time.time() - start_time) / 60.00), 1)}"
    print(execution_stats)
    results = results.reset_index()
    print(results.to_string(index=False))
    print(f"\n")
    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", "results.txt")
    if overwrite:
        results_file = open(filepath, "w")  # overwrite
    else:
        results_file = open(filepath, "a")  # append

    results_file.write(f"Experiment: {experiment}\n")
    results_file.write(f"{execution_stats}\n")
    results_file.write(results.to_string(index=False))
    results_file.write(f"\n\n\n")
    results_file.close()


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []

    for i in range(y_true.shape[0]):

        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])

        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))

        acc_list.append(tmp_a)

    return np.mean(acc_list)


def plot_loss_curves(history):

    loss = history.loss
    val_loss = history.val_loss

    binary_acc = history.binary_accuracy
    val_binary_acc = history.val_binary_accuracy

    epochs = history.Epoch

    # Plot loss
    plt.figure(figsize=(8,3))
    plt.plot(epochs, loss, label='training')
    plt.plot(epochs, val_loss, label='validation')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot binary accuracy
    plt.figure(figsize=(8,3))
    plt.plot(epochs, binary_acc, label='training')
    plt.plot(epochs, val_binary_acc, label='validation')
    plt.title('Binary Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

