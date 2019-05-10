import os
import fnmatch
import csv
import numpy as np
import sys
sys.stdout.flush()
import math

from collections import Counter

from sklearn import tree


import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import segment_cluster as sc
import importlib
importlib.reload(sc)


np.random.seed(0)

#"normal" lightcurves
rho_file=np.loadtxt("synthetic_rhos.csv", delimiter=',')

rho_train, rho_valid, rho_train_ids, rho_valid_ids= train_test_split(rho_file, list(range(len(rho_file))) ,test_size=0.25)

#"outlier" lightcurves
inverted_rho=np.copy(rho_valid)
for n, in_rho in enumerate(inverted_rho):
    rho_mean=np.mean(in_rho)
    inverted_rho[n]-=np.mean(in_rho)
    inverted_rho[n]*=-1
    inverted_rho[n]+=rho_mean

reco_error=[]
#reco_classes=[]
k_clusters=[10, 50, 100]
seg_lens=[4,8,12,30,50,100]

for k_id, k_cluster in enumerate(k_clusters):
    for len_id, seg_len in enumerate(seg_lens):
        ##train the model
        #loop throught the light curves of a given class and segments them
        all_train_segments=[]
        for rho in rho_train:
            train_segments=sc.segmentation(rho, seg_len, seg_len, time_stamps=False)
            all_train_segments.append(train_segments)
        all_train_segments=np.vstack(all_train_segments)
        #cluster the segments
        cluster=KMeans(n_clusters=k_cluster, random_state=0)
        cluster.fit(all_train_segments)

        ### reconstruction of the training class
        for n_rho, rho in enumerate(rho_valid):
            valid_segments= sc.segmentation(rho, seg_len, seg_len , time_stamps=False)
            reco = sc.reconstruct(valid_segments, rho, cluster, rel_offset=False, seg_slide=seg_len)
            error=np.sqrt(np.mean((rho[seg_len:-seg_len]-reco[seg_len:-seg_len])**2))
            reco_error.append((k_id,len_id,0, n_rho, error))
            print((k_id,len_id,0, n_rho, error), flush=True)


        #reconstruction loop through light curves for every class other than rho              
        for n_sine, sine in enumerate(inverted_rho):
            valid_segments= sc.segmentation(sine, seg_len, seg_len , time_stamps=False)
            reco = sc.reconstruct(valid_segments, sine, cluster, rel_offset=False, seg_slide=seg_len)
            error=np.sqrt(np.mean((sine[seg_len:-seg_len]-reco[seg_len:-seg_len])**2))
            reco_error.append((k_id,len_id,1,n_sine, error))
            print((k_id,len_id,1,n_sine, error), flush=True)
reco_error_ar=np.array(reco_error)
np.savetxt("valid_results_20190509_2.csv", reco_error_ar, delimiter=",") 