import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_features",type=int, default=2, help="number of features in the training data")
parser.add_argument("--n_samples", type=int,default=100, help="number of samples to be generated")
parser.add_argument("--n_classes", type=int,default=2, help="number of unique classes")
parser.add_argument("--save", type=str, default="csv", choices=["npy","csv"],help="save format of the dataset")
parser.add_argument("--class_sep",type=float,default=1.0,help="the class separation (float value default 1.0), More value, more seperation")


args = parser.parse_args()
features = args.n_features
samples = args.n_samples
classes = args.n_classes
formats = args.save

X1, Y1 = make_classification(n_features=features, n_classes=classes,n_samples=samples,n_redundant=0)

if features==2:
    plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")
    plt.savefig('./data_plot.png')
if formats=='npy':
    np.save('./Dataset/features.npy',X1, allow_pickle=True, fix_imports=True)
    np.save('./Dataset/labels.npy',Y1, allow_pickle=True, fix_imports=True)
elif formats=='csv':
    np.savetxt('./Dataset/features.csv',X1, delimiter=",")
    np.savetxt('./Dataset/labels.csv', Y1, delimiter=",")