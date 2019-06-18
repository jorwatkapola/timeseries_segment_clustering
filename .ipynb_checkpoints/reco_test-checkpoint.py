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

from scipy.stats import zscore



np.random.seed(0)

#"normal" lightcurves
rho_file=np.loadtxt("synthetic_rhos_v2.csv", delimiter=',')
#"outlier" lightcurves
sine_file=np.loadtxt("synthetic_sines_v3.csv", delimiter=',')

rho_train, rho_valid, rho_train_ids, rho_valid_ids= train_test_split(rho_file, list(range(len(rho_file))) ,test_size=0.25, random_state=0)

reco_error=[]
#reco_classes=[]
k_clusters=[150]
seg_lens=[10, 50, 100,150,200]

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
            reco= sc.reconstruct(valid_segments, rho, cluster, rel_offset=False, seg_slide=seg_len)
            
            reco[0:-seg_len]=reco[0:-seg_len]
            rho_expected=np.copy(rho[0:-seg_len])
            rho_error=np.power(np.e,np.log(rho_expected)*0.5+1.0397207708265923)
            error = np.mean(((reco[0:-seg_len]-rho_expected)/rho_error)**2.0)
            reco_error.append((k_id,len_id,0, rho_valid_ids[n_rho], error))
            print((k_id,len_id,0, rho_valid_ids[n_rho], error), flush=True)


        #reconstruction loop through light curves for every class other than rho              
        for n_sine, sine in enumerate(sine_file):
            valid_segments= sc.segmentation(sine, seg_len, seg_len , time_stamps=False)
            reco = sc.reconstruct(valid_segments, sine, cluster, rel_offset=False, seg_slide=seg_len)
            
            #reco[0:-seg_len] = np.mean(sine[0:-seg_len])+ (reco[0:-seg_len]- np.mean(reco[0:-seg_len]))*(np.std(sine[0:-seg_len])/np.std(reco[0:-seg_len]))
            reco[0:-seg_len]=(reco[0:-seg_len])
            sine_expected=np.copy((sine[0:-seg_len]))
            sine_error=np.power(np.e,np.log(rho_expected)*0.5+1.0397207708265923)
            error = np.mean(((reco[0:-seg_len]-sine_expected)/sine_error)**2.0)
            reco_error.append((k_id,len_id,1,n_sine, error))
            print((k_id,len_id,1,n_sine, error), flush=True)
reco_error_ar=np.array(reco_error)
np.savetxt("chi2_20190607_5.csv", reco_error_ar, delimiter=",") 