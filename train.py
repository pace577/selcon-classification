import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(".","SELCON"))
sys.path.append(os.path.join(".","SELCON\\utils"))
sys.path.append(os.path.join(".","SELCON\\utils\\custom_dataset"))
sys.path.append(os.path.join(".","SELCON\\model"))
sys.path.append(os.path.join(".","SELCON\\gen_result"))

# print(sys.path)
from SELCON.datasets import load_def_data
from SELCON.logistic import Regression
from SELCON.utils import cross_entropy_loss


reg = Regression()
# Load data and prepare linear regression model
# (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_def_data("cadata")
(x_train,y_train),(x_val,y_val),(x_test,y_test)=load_def_data('Custom_Data',class_sep=1)
print(y_train)
# import numpy as np
# X=np.load('./Dataset/features.npy')
# y=np.load('./Dataset/labels.npy')
# reg = Regression()

# Trains SELCON model for a subset fraction of 0.03 on the training subset (no fairness)
np.random.seed(42)
reg.train_model_fair(x_train, y_train, x_val, y_val, fraction = 0.1)

# Return optimal subset indices
subset_idxs = reg.return_subset()

# Returns the optimal subset of the training data for further use
X_sub = x_train[subset_idxs]
y_sub = y_train[subset_idxs]

xx = x_train.cpu().numpy()
yy = y_train.cpu().numpy()

xx_s = X_sub.cpu().numpy()
yy_s = y_sub.cpu().numpy()

xx_v = x_val.cpu().numpy()
yy_v = y_val.cpu().numpy()

# Trains logistic regression model on the training set and the subset
# And evaluates both models on the validation set. Returns the validation loss on trainset and subset
trnset_loss, subset_loss = cross_entropy_loss.cross_entropy_loss(x_train, y_train, X_sub, y_sub, x_val, y_val, lr=0.01, epochs= 1000)
print(f"Validation loss on model trained on training set: {trnset_loss.item():.4f}")
print(f"Validation loss on model trained on subset: {subset_loss.item():.4f}")

plt.scatter(xx[:, 0], xx[:, 1], marker="o", c=yy, s=25, edgecolor="k")
plt.scatter(xx_s[:, 0], xx_s[:, 1], marker="^", c=yy_s, s=70, edgecolor="k")
plt.scatter(xx_v[:, 0], xx_v[:, 1], marker="*", c=yy_v, s=70, edgecolor="k")
plt.show()