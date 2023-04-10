import sys
import os
sys.path.append(os.path.join(".","SELCON"))
sys.path.append(os.path.join(".","SELCON\\utils"))
sys.path.append(os.path.join(".","SELCON\\model"))
sys.path.append(os.path.join(".","SELCON\\gen_result"))
# print(sys.path)
from SELCON.datasets import load_def_data
from SELCON.linear import Regression

# Load data and prepare linear regression model
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_def_data("cadata")
reg = Regression()

# Trains SELCON model for a subset fraction of 0.03 on the training subset (no fairness)
reg.train_model_fair(x_train, y_train, x_val, y_val, fraction = 0.03)

# Return optimal subset indices
subset_idxs = reg.return_subset()

# Returns the optimal subset of the training data for further use
X_sub = x_train[subset_idxs]
y_sub = y_train[subset_idxs]