import time

t0 = time.process_time()

import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import importlib
from sklearn.cluster import KMeans
from scipy.stats import zscore
import segment_cluster as sc
importlib.reload(sc)
sys.stdout.flush()
np.random.seed(0)

#"ordinary" lightcurves
ordinary_file=np.loadtxt("synthetic_rhos_v2.csv", delimiter=',')
#"outlier" lightcurves
#sine_file=np.loadtxt("synthetic_sines_v3.csv", delimiter=',')
outlier_file=np.loadtxt("synthetic_boxes_thick.csv", delimiter=',')


ordinary_train, ordinary_valid, ordinary_train_ids, ordinary_valid_ids= train_test_split(ordinary_file, list(range(len(ordinary_file))) ,test_size=0.25, random_state=0)

k_clusters=[10, 50, 100, 200]
seg_lens=[10, 50, 100,150,200]

reco_error=[]
for k_id, k_cluster in enumerate(k_clusters):
    for len_id, seg_len in enumerate(seg_lens):
        ##train the model
        #loop throught the light curves of a given class and segments them
        all_train_segments=[]
        for rho in ordinary_train:
            train_segments=sc.segmentation(rho, seg_len, 1, time_stamps=False)
            all_train_segments.append(train_segments)
        all_train_segments=np.vstack(all_train_segments)
        #cluster the segments
        cluster=KMeans(n_clusters=k_cluster, random_state=0)
        cluster.fit(all_train_segments)     
        

        ### reconstruction of the training class
        for n_rho, rho in enumerate(ordinary_valid):
            valid_segments= sc.segmentation(rho, seg_len, seg_len , time_stamps=False)
            reco= sc.reconstruct(valid_segments, rho, cluster, rel_offset=False, seg_slide=seg_len)
            
            reco[0:-seg_len]=zscore(reco[0:-seg_len])
            rho_expected=zscore(np.copy(rho[0:-seg_len]))
           # rho_error=np.power(np.e,np.log(rho_expected)*0.5+1.0397207708265923)
            error = np.mean(((reco[0:-seg_len]-rho_expected))**2.0)
            reco_error.append((k_id,len_id,0, ordinary_valid_ids[n_rho], error))
           # print((k_id,len_id,0, ordinary_valid_ids[n_rho], error), flush=True)


        #reconstruction loop through light curves for every class other than rho              
        for n_sine, sine in enumerate(outlier_file):
            valid_segments= sc.segmentation(sine, seg_len, seg_len , time_stamps=False)
            reco = sc.reconstruct(valid_segments, sine, cluster, rel_offset=False, seg_slide=seg_len)
            
            reco[0:-seg_len]=zscore(reco[0:-seg_len])
            sine_expected=zscore(np.copy((sine[0:-seg_len])))
            #sine_error=np.power(np.e,np.log(rho_expected)*0.5+1.0397207708265923)
            error = np.mean(((reco[0:-seg_len]-sine_expected))**2.0)
            reco_error.append((k_id,len_id,1,n_sine, error))
           # print((k_id,len_id,1,n_sine, error), flush=True)
        print(time.process_time()-t0)
reco_error_ar=np.array(reco_error)
np.savetxt("old_20190617.csv", reco_error_ar, delimiter=",") 
print(time.process_time()-t0)