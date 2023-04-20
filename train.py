import sys
import os
import matplotlib.pyplot as plt
from torch import nn

sys.path.append(os.path.join(".","SELCON"))

# print(sys.path)
from SELCON.datasets import load_def_data
from SELCON.logistic import Classification

criterion = nn.BCELoss
reg = Classification(num_cls=1, criterion=criterion)
# Load data and prepare linear regression model
# (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_def_data("cadata")
(x_train,y_train),(x_val,y_val),(x_test,y_test)=load_def_data('Custom_Data',class_sep=1)
print(y_train)
# import numpy as np
# X=np.load('./Dataset/features.npy')
# y=np.load('./Dataset/labels.npy')
# reg = Regression()

# Trains SELCON model for a subset fraction of 0.03 on the training subset (no fairness)
reg.train_model(x_train, y_train, x_val, y_val, fraction = 0.2, fair=True)

# Return optimal subset indices
subset_idxs = reg.return_subset()

# Returns the optimal subset of the training data for further use
X_sub = x_train[subset_idxs]
y_sub = y_train[subset_idxs]

xx = x_train.cpu().numpy()
yy = y_train.cpu().numpy()

xx_v = X_sub.cpu().numpy()
yy_v = y_sub.cpu().numpy()

plt.scatter(xx[:, 0], xx[:, 1], marker="o", c=yy, s=25, edgecolor="k")
plt.scatter(xx_v[:, 0], xx_v[:, 1], marker="^", c=yy_v, s=70, edgecolor="k")
plt.show()
