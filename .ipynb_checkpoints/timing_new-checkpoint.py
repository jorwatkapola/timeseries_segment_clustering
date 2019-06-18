import time

t0 = time.process_time()

import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import importlib

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

import segment_cluster as sc
importlib.reload(sc)


validation_data = np.vstack((ordinary_valid, outlier_file)) # stack validation data of ordinary and outlier time series
validation_labels = np.hstack((np.zeros(len(ordinary_valid)), np.ones(len(outlier_file)))).T # generate labels for valdiation data


validation_result = []
for k_id, k_cluster in enumerate(k_clusters):
    for len_id, seg_len in enumerate(seg_lens): #for every combination of hyperparameters
        TSSCOD = sc.TSSCOD(k_clusters = k_cluster, seg_len = seg_len) #initialise outlier detection class

        TSSCOD.train(ordinary_train, random_state = 0) # train on the subset with no outliers; segment each series with a slide of 1 and cluster the segments
        validation_iteration = TSSCOD.validate(validation_data,
                                               validation_labels) #validation reconstructs the provided series and saves error values together with indices
        hyperparameter_ids = np.vstack((np.ones(len(validation_iteration))*k_id, # add indices of hyperparameters to the results array; this part makes it easier to feed the result into the analysis pipeline
                                        np.ones(len(validation_iteration))*len_id)).T
        validation_iteration = np.hstack((hyperparameter_ids, validation_iteration))
        validation_result.append(validation_iteration) # append results for this set of hyperparameters
        print(time.process_time()-t0)
validation_result=np.vstack(validation_result)
np.savetxt("new_20190617.csv", validation_result, delimiter=",") 
print(time.process_time()-t0)