import numpy as np
import pandas as pd

def load_data(file_path):
    df = pd.read_pickle(file_path)
    return df

def preprocess_data(df):
    f_squeeze = lambda x: str(np.squeeze(x))
    df["failureType"] = df["failureType"].map(f_squeeze)
    df["trianTestLabel"] = df["trianTestLabel"].map(f_squeeze)
    df_with_label = df[df['failureType'] != '[]']
    df_with_label = df_with_label[df_with_label['failureType'] != 'none']
    df_with_label['failureType'] = df_with_label['failureType'].replace({'Edge-Loc': 'Loc'})
    return df_with_label

def encode_labels(df_with_label, class2idx):
    df_with_label["encoded_labels"] = df_with_label["failureType"].replace(class2idx)
    return df_with_label

def split_data(df_with_label, train_frac=0.8, val_frac=0.7):
    train_data_length = int(train_frac * len(df_with_label))
    df_with_label = df_with_label.sample(frac=1, random_state=2) 
    df_train, df_test = df_with_label[:train_data_length], df_with_label[train_data_length:]
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    train_data_length = int(val_frac * len(df_train))
    df_train, df_val = df_train[:train_data_length], df_train[train_data_length:]
    return df_train, df_val, df_test

def save_dataframes(df_train, df_val, df_test, train_path, val_path, test_path):
    df_train.to_pickle(train_path)
    df_val.to_pickle(val_path)
    df_test.to_pickle(test_path)
