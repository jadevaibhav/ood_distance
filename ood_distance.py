import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
import torch

embed_classwise = {}
fea_path = "/home/vaibhav/Desktop/stud/datasets/esmart/bigdetect_features/GT/pooled/"

for im in os.listdir(fea_path):
    x = {}
    with open(os.path.join(fea_path,im), 'rb') as f:
        x = pickle.load(f)
        
    idx = 0
    for i in x['preds']:
        if embed_classwise.get(i.item()) == None:
            embed_classwise[i.item()] = [x['features'][idx].detach().cpu().numpy()]
        else:
            embed_classwise[i.item()].append(x['features'][idx].detach().cpu().numpy())
        idx += 1

def sample_estimator(embed_classwise):
    classwise_mean = np.zeros((len(embed_classwise.keys()),embed_classwise[0][0].shape[0]),dtype='float32')
    precision = []
    group_lasso = EmpiricalCovariance(assume_centered=False)
    for c in range(len(embed_classwise.keys())):
        f_x = np.array(embed_classwise[c])
        classwise_mean[c] = f_x.mean(axis=0)
        f_x = f_x - classwise_mean[c][np.newaxis,:]
        
        ### finding inverse of covariance
        group_lasso.fit(f_x)
        temp_precision = group_lasso.precision_
        precision.append(temp_precision)
    precision = np.array(precision)
   

    return classwise_mean,precision