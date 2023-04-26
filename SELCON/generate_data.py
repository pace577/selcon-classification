import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns



def generate_data(n_features=13,n_samples=10000,n_classes=2,class_sep=3,dataset='generation'):
    scale = StandardScaler()

    if dataset =='generation':

        X1, Y1 = make_classification(n_features=n_features, n_classes=n_classes,n_samples=n_samples,n_redundant=5,class_sep = class_sep)

        X_train, X_v, y_train, y_v = train_test_split(X1,Y1, test_size=0.30, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_v,y_v, test_size=0.20, random_state=42)
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(X_train)
        df = pd.DataFrame()
        df["y"] = y_train
        df["comp-1"] = z[:,0]
        df["comp-2"] = z[:,1]

        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", 2),
                        data=df).set(title="Generated data T-SNE projection class sep="+str(class_sep)) 
        plt.show()

    # reading csv files
    # data =  pd.read_csv(r'C:\Users\hp\Desktop\selcon-classification\Dataset\Madelon\madelon_train.data',sep=' ')
    # print(data)
    elif dataset =='madelon':
        data_X_train= pd.read_csv(r'C:\Users\hp\Desktop\selcon-classification\Dataset\Madelon\madelon_train.data', delimiter=' ', header=None)
        data_y_train = pd.read_csv(r'C:\Users\hp\Desktop\selcon-classification\Dataset\Madelon\madelon_train.labels', delimiter=' ', header=None, names=['target'])
        data_X_train.drop([500], axis=1, inplace=True)
        data_y_train['target'] = data_y_train['target'].replace(to_replace=-1, value=0)
        data_y_train = data_y_train.values
        data_X_train = (data_X_train-data_X_train.min())/(data_X_train.max()-data_X_train.min())
        data_X_train =data_X_train.values
        # data_X_train = scale.fit_transform(data_X_train)
        
        X_train, X_val, y_train, y_val = train_test_split(data_X_train,data_y_train, test_size=0.20, random_state=42)
        X_test= pd.read_csv(r'C:\Users\hp\Desktop\selcon-classification\Dataset\Madelon\madelon_valid.data', delimiter=' ', header=None)
        X_test.drop([500], axis=1, inplace=True)
        X_test = (X_test-X_test.min())/(X_test.max()-X_test.min())
        X_test = X_test.values
        # X_test = scale.transform(X_test)
        # X_test = X_test.values
        y_test = pd.read_csv(r'C:\Users\hp\Desktop\selcon-classification\Dataset\Madelon\madelon_valid.labels', delimiter=' ', header=None, names=['target'])
        y_test['target'] = y_test['target'].replace(to_replace=-1, value=0)
        y_test = y_test.values
        y_test = y_test.reshape(-1)
        y_val = y_val.reshape(-1)
        y_train = y_train.reshape(-1)
    elif dataset =='gisette':
        data_X_train= pd.read_csv(r'C:\Users\hp\Desktop\selcon-classification\Dataset\gisette\gisette_train.data', delimiter=' ', header=None)
        data_y_train = pd.read_csv(r'C:\Users\hp\Desktop\selcon-classification\Dataset\gisette\gisette_train.labels', delimiter=' ', header=None, names=['target'])
        data_X_train.drop([5000], axis=1, inplace=True)
        data_y_train['target'] = data_y_train['target'].replace(to_replace=-1, value=0)
        data_y_train = data_y_train.values
        # print(data_X_train.isnull().values.any())
        data_X_train =data_X_train.values
        data_X_train = scale.fit_transform(data_X_train)
        # data_X_train = (data_X_train-data_X_train.min())/(data_X_train.max()-data_X_train.min())
        # print(data_X_train)
        X_train, X_val, y_train, y_val = train_test_split(data_X_train,data_y_train, test_size=0.20, random_state=42)
        X_test= pd.read_csv(r'C:\Users\hp\Desktop\selcon-classification\Dataset\gisette\gisette_valid.data', delimiter=' ', header=None)
        X_test.drop([5000], axis=1, inplace=True)
        # X_test = (X_test-X_test.min())/(X_test.max()-X_test.min())
        # print(X_test)
        X_test = X_test.values
        X_test = scale.transform(X_test)
        y_test = pd.read_csv(r'C:\Users\hp\Desktop\selcon-classification\Dataset\gisette\gisette_valid.labels', delimiter=' ', header=None, names=['target'])
        y_test['target'] = y_test['target'].replace(to_replace=-1, value=0)
        y_test = y_test.values
        y_test = y_test.reshape(-1).astype(int)
        y_val = y_val.reshape(-1).astype(int)
        y_train = y_train.reshape(-1).astype(int)
        # print(y_train)
        # print(y_test)
        # print(y_val.shape)
        print(np.isnan(X_val).any())
        print(X_val.shape)
        # print(np.unique(y_val))
        # print(np.unique(y_test))
    return (X_train,y_train),(X_val, y_val),(X_test,y_test)
