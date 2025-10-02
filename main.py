import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

fp_train = os.path.join('data', 'train.csv')

df = pd.read_csv(fp_train)
print(df.head())