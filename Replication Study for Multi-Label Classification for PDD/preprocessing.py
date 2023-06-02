import pandas as pd
import numpy as np
import random
import os

from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

# set seed
SEED = 100
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


class Preprocessing:
    DATA_URL = "https://raw.githubusercontent.com/saniemeyer/dl4h-psychotic-disorders-replication/main/data/dlh-data.csv"

    @staticmethod
    def create_dataset(data_url=DATA_URL, exclude_feature=""):

        df = pd.read_csv(data_url)

        # rename columns to correct spelling mistakes and make them more interpretable
        df.columns = ["Sex", "Age", "Family History", "Religion", "Occupation", "Genetic",
                      "Marital Status", "Loss of Parent", "Divorce", "Head Injury",
                      "Spiritual Consultation", "Insomnia", "Schizophrenia",
                      "Vascular Dementia", "ADHD", "Bipolar", "Age Group"]

        if exclude_feature in ["Age", "Age Group"]:
            df["Age Group"] = 0
            df["Age"] = 0
        else:
            df[exclude_feature] = 0

        # One Hot Encoding
        enc = OneHotEncoder(drop="first", sparse=False, dtype=np.int8)

        # One Hot Encode target variables and concatenate to create a "target" variable
        target_cols = ["Insomnia", "Schizophrenia", "Vascular Dementia", "ADHD", "Bipolar"]
        encoded_target = pd.DataFrame(enc.fit_transform(df[target_cols]))
        encoded_target.columns = target_cols
        encoded_target["target"] = encoded_target["Insomnia"].astype(str) + encoded_target["Schizophrenia"].astype(str) + encoded_target["Vascular Dementia"].astype(str) + encoded_target["ADHD"].astype(str) + encoded_target["Bipolar"].astype(str)
        encoded_target["target"] = encoded_target["target"].astype(int)
        df.drop(target_cols, inplace=True, axis=1)
        df = pd.concat([df, encoded_target], axis=1)

        # One Hot Encode binary-valued variables
        binary_cols = ["Sex", "Family History", "Genetic", "Marital Status", "Loss of Parent", "Divorce", "Head Injury",
                       "Spiritual Consultation"]
        encoded_binary = pd.DataFrame(enc.fit_transform(df[binary_cols]))
        encoded_binary.columns = binary_cols

        # One Hot Encode multi-valued categorical variables
        categorical_cols = ["Religion", "Occupation"]
        encoded_categorical = pd.DataFrame(enc.fit_transform(df[categorical_cols]))
        encoded_categorical.columns = enc.get_feature_names_out()

        # numeric cols
        numeric_cols = pd.DataFrame()
        if exclude_feature not in ["Age", "Age Group"]:
            numeric_cols = df[["Age", "Age Group"]]

        # reconstitute our data set with the encoded columns
        df = pd.concat([numeric_cols, encoded_categorical, encoded_binary, encoded_target], axis=1, ignore_index=True)
        df.columns = numeric_cols.columns.tolist() + encoded_categorical.columns.tolist() + encoded_binary.columns.tolist() + encoded_target.columns.tolist()

        # Remove target classes with less than 6 occurrences
        df = df.groupby(["target"]).filter(lambda x: len(x) > 5).reset_index().drop("index", axis=1)

        # SMOTE the data
        smote = SMOTE(random_state=SEED)

        # We get a synthetic data set of 1212 rows: 101 samples for each of the remaining 12 classes
        df_balanced, _ = smote.fit_resample(df, df.target)

        # Save Imbalanced Multi-Label dataset
        os.makedirs("data", exist_ok=True)
        x = df.drop(target_cols + ["target"], axis=1)
        y = df[target_cols + ["target"]]
        data_path = os.path.join("data", "x_imbalanced.csv")
        x.to_csv(path_or_buf=data_path, header=True, index=False)
        data_path = os.path.join("data", "y_imbalanced.csv")
        y.to_csv(path_or_buf=data_path, header=True, index=False)

        # Save Balanced  Multi-Label dataset
        x = df_balanced.drop(target_cols + ["target"], axis=1)
        y = df_balanced[target_cols + ["target"]]
        data_path = os.path.join("data", "x_balanced.csv")
        x.to_csv(path_or_buf=data_path, header=True, index=False)
        data_path = os.path.join("data", "y_balanced.csv")
        y.to_csv(path_or_buf=data_path, header=True, index=False)