import os
import json
import pandas as pd
from src.data_processing import load_data, preprocess_data, encode_labels, split_data, save_dataframes

data_path = "../data/LSWMD.pkl"
class2idx_path = "../config/config_class2idx.json"

# Load class2idx mapping
with open(class2idx_path, "r") as f:
    class2idx = json.load(f)

# Load and preprocess data
df = load_data(data_path)
df_with_label = preprocess_data(df)
df_with_label = encode_labels(df_with_label, class2idx)

# Split and save data
df_train, df_val, df_test = split_data(df_with_label)
save_dataframes(df_train, df_val, df_test, "../data/dataset_train.pickle", "../data/dataset_val.pickle", "../data/dataset_test.pickle")
