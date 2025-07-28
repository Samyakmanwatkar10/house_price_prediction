# Importing the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# importing the dataset
house_price_dataset=pd.read_csv('housing.csv')
# print(house_price_dataset)

