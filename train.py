import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
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
# print(y_train)
# import numpy as np
# X=np.load('./Dataset/features.npy')
# y=np.load('./Dataset/labels.npy')
# reg = Regression()
val_train_loss=[]
val_train_sub_loss=[]
val_train_rand_loss=[]
# Trains SELCON model for a subset fraction of 0.03 on the training subset (no fairness)
for i in [0.01,0.03,0.05,0.07,0.1]:
    reg = Regression()
    reg.train_model_fair(x_train, y_train, x_val, y_val, fraction = i)
    # if i==0.03:
    #     break
    # Return optimal subset indices
    subset_idxs = reg.return_subset()


    # Returns the optimal subset of the training data for further use
    X_sub = x_train[subset_idxs]
    y_sub = y_train[subset_idxs]

    idx = np.arange(len(x_train))
    X_rand=[]
    Y_rand=[]
    for k in range(20):
        l = np.random.choice(idx, size=len(X_sub ), replace=False)
        print(x_train[l].size())
        X_rand.append(x_train[l])
        Y_rand.append(y_train[l])

    X_rand = torch.stack(X_rand)
    Y_rand = torch.stack(Y_rand)
    xx = x_train.cpu().numpy()
    yy = y_train.cpu().numpy()

    xx_v = X_sub.cpu().numpy()
    yy_v = y_sub.cpu().numpy()


    trnset_loss, subset_loss, rand_loss = cross_entropy_loss.cross_entropy_loss(x_train, y_train, X_sub, y_sub,X_rand,Y_rand,x_val, y_val,x_test,y_test, lr=0.01, epochs= 1000)
    val_train_loss.append(trnset_loss)
    val_train_sub_loss.append(subset_loss)
    val_train_rand_loss.append(rand_loss)

    # trnset_loss, subset_loss = cross_entropy_loss.cross_entropy_loss(x_train, y_train, X_rand, y_, x_val, y_val, lr=0.01, epochs= 1000)
    # print(f"Validation loss on model trained on training set: {trnset_loss.item():.4f}")
    # print(f"Validation loss on model trained on subset: {subset_loss.item():.4f}")
t=0
print(f"Validation loss on model trained on training set: {val_train_loss[t].item():.4f}")
for i in [0.01,0.03,0.05,0.07,0.1]:
    print(f"Validation loss on model trained on subset: {val_train_sub_loss[t].item():.4f} subset size: {i}")
    print(f"Validation loss on model trained on random set: {val_train_rand_loss[t].item():.4f} subset size: {i}")
    t+=1
# plt.scatter(xx[:, 0], xx[:, 1], marker="o", c=yy, s=25, edgecolor="k")
# plt.scatter(xx_v[:, 0], xx_v[:, 1], marker="^", c=yy_v, s=70, edgecolor="k")
val_train_loss = torch.stack(val_train_loss).cpu().detach().numpy()
val_train_sub_loss = torch.stack(val_train_sub_loss).cpu().detach().numpy()
val_train_rand_loss = torch.stack(val_train_rand_loss).cpu().detach().numpy()

plt.plot(np.array([1,3,5,7,10]),val_train_loss,c='g',label='training set')
plt.plot(np.array([1,3,5,7,10]),val_train_sub_loss,c='r',label='subset')
plt.plot(np.array([1,3,5,7,10]),val_train_rand_loss,c='b',label='random subset')
plt.legend()
plt.show()

