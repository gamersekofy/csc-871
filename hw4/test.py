import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv('mlp_regression_data.csv')
X = df['x'].values.astype('float32')
y = df['y'].values.astype('float32')

plt.scatter(X, y, s=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Data')
plt.show()
