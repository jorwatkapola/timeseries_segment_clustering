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

clean_belloni = open('1915Belloniclass_updated.dat')
lines = clean_belloni.readlines()
states = lines[0].split()
belloni_clean = {}
for h,l in zip(states, lines[1:]):
    belloni_clean[h] = l.split()
    #state: obsID1, obsID2...
ob_state = {}
for state, obs in belloni_clean.items():
    if state == "chi1" or state == "chi2" or state == "chi3" or state == "chi4": state = "chi"
    for ob in obs:
        ob_state[ob] = state

available = []
pool=[]

#/home/jkok1g14/Documents/GRS1915+105/data
#/export/data/jakubok/GRS1915+105/Std1_PCU2
for root, dirnames, filenames in os.walk("/export/data/jakubok/GRS1915+105/Std1_PCU2"):
    for filename in fnmatch.filter(filenames, "*_std1_lc.txt"):
        available.append(filename)
for ob, state in ob_state.items():
    if ob+"_std1_lc.txt" in available:
        pool.append(ob)  

#create a list of arrays with time and counts for the set of Belloni classified observations
lc_dirs=[]
lcs=[]
ids=[]
for root, dirnames, filenames in os.walk("/export/data/jakubok/GRS1915+105/Std1_PCU2"):    
    for filename in fnmatch.filter(filenames, "*_std1_lc.txt"):
        if filename.split("_")[0] in pool:
            lc_dirs.append(os.path.join(root, filename))

            
#make 2D arrays for light curves, with columns of counts and time values
for lc in lc_dirs:
    ids.append(lc.split("/")[-1].split("_")[0])
    f=np.loadtxt(lc)
    f=np.transpose(f)#,axis=1)
    f=f[0:2]
    ###1s average and time check to eliminate points outside of GTIs
    f8t = np.mean(f[0][:(len(f[0])//8)*8].reshape(-1, 8), axis=1)
    f8c = np.mean(f[1][:(len(f[1])//8)*8].reshape(-1, 8), axis=1)
    f8c=f8c-np.mean(f8c)#normalisation/mean centering/whatever you desire most
    rm_points = []
    skip=False
    for i in range(len(f8t)-1):
        if skip==True:
            skip=False
            continue
        delta = f8t[i+1]-f8t[i]
        if delta > 1.0:
            rm_points.append(i+1)
            skip=True
            
####### normalise the count rates! think about the effect of 0-1 normalisation on the distance calculation
####### due to the energy integration in Std1 diefferences between different epochs shouldn't matter; there would be very few photons found at the extremes of the range            
    times=np.delete(f8t,rm_points)
    counts=np.delete(f8c,rm_points)
    lcs.append(np.stack((times,counts)))
    
lc_classes=[]
for i in ids:
    lc_classes.append(ob_state[i])
lc_classes

drop_classes=[]
for clas, no in Counter(lc_classes).items():
    if no<7:
        drop_classes.append(clas)

lcs_abu = []
classes_abu = []
ids_abu = []
for n, lc in enumerate(lc_classes):
    if lc not in drop_classes:
        classes_abu.append(lc)
        lcs_abu.append(lcs[n])
        ids_abu.append(ids[n])  
x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(lcs_abu, classes_abu, ids_abu, test_size=0.5, random_state=0, stratify=classes_abu)
x_valid, x_test, y_valid, y_test, id_valid, id_test = train_test_split(x_test, y_test, id_test, test_size=0.5, random_state=0, stratify=y_test)

import segment_cluster as sc
import importlib
importlib.reload(sc)
pro_clusters=[100, 150, 200]
seg_lens=[30, 50, 70]
classes=set(y_train)
results=np.zeros((len(pro_clusters), len(seg_lens), len(classes), len(classes), 2))
for n_pro, proportion in enumerate(pro_clusters):
    for n_len, length in enumerate(seg_lens):
        for n_model, model_class in enumerate(classes):

            ##train the model
            time_stamps=False
            offset=True
            training_ys=np.where(np.array(y_train)=='{}'.format(model_class))[0]
            all_train_segments=[]
            for ts_i in training_ys:
                ts=x_train[ts_i]
                train_segments=sc.segmentation(ts, length, 2, time_stamps=time_stamps)
                c_train_segments=sc.center_offset(train_segments, ts, offset=offset, time_stamps=time_stamps)
                all_train_segments.append(c_train_segments)
            all_train_segments=np.vstack(all_train_segments)
            if proportion > len(all_train_segments): proportion = len(all_train_segments)
            #cluster=KMeans(n_clusters=int(proportion*len(all_train_segments)), random_state=0)
            cluster=KMeans(n_clusters=proportion, random_state=0)
            cluster.fit(all_train_segments)
            
            ##test against the validation set
            for n_test, test_class in enumerate(classes):
                testing_ys=np.where(np.array(y_valid)=='{}'.format(test_class))[0]
                time_stamps=True
                offset=False
                reco_error=[]
                seg_len=length
                seg_slide=int(length*0.5)
                for ts_id in testing_ys:
                    test_ts=x_valid[ts_id]
                    test_segments= sc.segmentation(test_ts, seg_len, seg_slide, time_stamps=time_stamps)
                    c_test_segments=sc.center_offset(test_segments, test_ts, offset=offset, time_stamps=time_stamps)
                    reco = sc.reconstruct(c_test_segments, test_ts, cluster, rel_offset=offset)
                    error=np.sqrt(np.mean((test_ts[1][seg_len:-seg_len]-reco[1][seg_len:-seg_len])**2))
                    reco_error.append((ts_id, error))
                results[n_pro, n_len, n_model, n_test, 0]=np.mean(np.array(reco_error)[:,1])
                results[n_pro, n_len, n_model, n_test, 1]=np.std(np.array(reco_error)[:,1])
                print(n_pro, n_len, n_model, n_test, results[n_pro, n_len, n_model, n_test, 0], results[n_pro, n_len, n_model, n_test, 1], flush=True)

np.savetxt("model_errors.csv", results, delimiter=",")