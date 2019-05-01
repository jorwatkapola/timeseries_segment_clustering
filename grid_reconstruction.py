import importlib
import os
import fnmatch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import csv
from sklearn import tree
import sys
sys.stdout.flush()
import math
import matplotlib.pyplot as plt
from matplotlib.table import Table
import segment_cluster as sc
import importlib
importlib.reload(sc)
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import LeaveOneOut

np.random.seed(0)
rho_file=np.loadtxt("synthetic_rhos.csv", delimiter=',')
rho_train, rho_valid, rho_train_ids, rho_valid_ids= train_test_split(rho_file, list(range(len(rho_file))) ,test_size=0.25)

seg_slide=1
seg_lens=[4,6,8,12,20,30,50,100]
k_clusters=[10,25,50,75,100,150]

print("Error of reconstruction of synthetic rho light curves", flush=True)
for k_cluster in k_clusters:
    for seg_len in seg_lens:
        all_train_segments=[]
        for rho in rho_train:
            train_segments=sc.segmentation(rho, seg_len, seg_slide, time_stamps=False)
            all_train_segments.append(train_segments)
        all_train_segments=np.vstack(all_train_segments)
        
        cluster=KMeans(n_clusters=k_cluster, random_state=0)
        cluster.fit(all_train_segments)
        
        reco_error=[]
        for n_rho, rho in enumerate(rho_valid):
            valid_segments= sc.segmentation(rho, seg_len, seg_len , time_stamps=False)
            reco = sc.reconstruct(valid_segments, rho, cluster, rel_offset=False, seg_slide=seg_len)
            error=np.sqrt(np.mean((rho[seg_len:-seg_len]-reco[seg_len:-seg_len])**2))
            reco_error.append((n_rho, error))
        print("k: {}, len: {}, mean: {}, min: {}, max: {}".format(k_cluster,seg_len, np.mean(np.array(reco_error)[:,1]),np.min(np.array(reco_error)[:,1]),np.max(np.array(reco_error)[:,1])), flush=True)
        best_reco_ind=np.argmin(np.array(reco_error)[:,1])
        worst_reco_ind=np.argmax(np.array(reco_error)[:,1])
        
        plt.gcf().set_size_inches(30,10)
        plt.subplot(2,1,1)
        valid_segments= sc.segmentation(rho_valid[best_reco_ind], seg_len, seg_len , time_stamps=False)
        reco = sc.reconstruct(valid_segments, rho_valid[best_reco_ind], cluster, rel_offset=False, seg_slide=seg_len)
        plt.plot(rho_valid[best_reco_ind])
        plt.plot(reco)
        plt.subplot(2,1,2)
        valid_segments= sc.segmentation(rho_valid[worst_reco_ind], seg_len, seg_len , time_stamps=False)
        reco = sc.reconstruct(valid_segments, rho_valid[worst_reco_ind], cluster, rel_offset=False, seg_slide=seg_len)
        plt.plot(rho_valid[worst_reco_ind])
        plt.plot(reco)
        plt.savefig("grid_figs/k{}_len{}_mean{}.png".format(k_cluster, seg_len, int(np.mean(np.array(reco_error)[:,1]))))