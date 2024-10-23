import os
import sys
import torch
import yaml

from pathlib import Path
from dotenv import load_dotenv

from data.processors import CSVProcessor
from torch.utils.data import TensorDataset, DataLoader
from data.datasets import AdultDataset
from models.xgboost import XGBoostModel

# Define the path to the CSV data file
config_path = Path.cwd() / 'config'
data_path = Path.cwd() / 'src' / 'datasets' / 'bronze' / 'adult.csv'


# Create dataset and process if needed
adult_dataset = AdultDataset(
    file_path=data_path,
    file_type='csv',
    target_column='income',
    na_values='?',
    encode_cat=True,
    scale_num=True
    )

# train_loader = DataLoader(adult_dataset.get_train_dataset(), batch_size=32, shuffle=True)
# test_loader = DataLoader(adult_dataset.get_test_dataset(), batch_size=32, shuffle=False)

X_train, y_train = adult_dataset.get_train_dataset(split=True)
X_test, y_test = adult_dataset.get_test_dataset(split=True)



model = XGBoostModel()
model.train(X_train, y_train, X_test, y_test)
model.evaluate(X_test, y_test)


  