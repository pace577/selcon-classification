import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def generate_data(n_features=2,n_samples=100,n_classes=2,class_sep=0.1):

    X1, Y1 = make_classification(n_features=n_features, n_classes=n_classes,n_samples=n_samples,n_redundant=0,class_sep = class_sep)

    X_train, X_v, y_train, y_v = train_test_split(X1,Y1, test_size=0.20, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_v,y_v, test_size=0.20, random_state=42)

    return (X_train,y_train),(X_val,y_val),(X_test,y_test)
