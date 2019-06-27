import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import zscore
import sys
import os
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
import datetime

#time series and light curve can be used interchangebly below
def segmentation(time_series, seg_len, stride, time_stamps=True):
    """
    Create a list of 1D (when time_stamps=False) or 2D (when time_stamps=True) arrays, which are overlappig segments of ts. Incomplete fragments are rejected.
    
    time_series = time series to be segmented
    seg_len = length of a segment, 
    stride = step size; difference in the starting position of the consecutive segments
    """
    
    if time_stamps==True:
        segments=[] #probably no way to make an array apriori
        for start in range(0, len(time_series[0])-seg_len, stride):
            end=start+seg_len
            if time_series[0][end]-time_series[0][start] != seg_len: #don't allow temporally discontinous segments
                continue
            segments.append(time_series[:,start:end])
        return np.array(segments) # check why time stamps are kept 
    else:
        no_segments = int((len(time_series)-seg_len)/stride +1)
        segments = np.zeros((no_segments, seg_len))
        for seg_count, start in enumerate(range(0, len(time_series)-seg_len+1, stride)):
            end=start+seg_len
            segments[seg_count] = time_series[start:end]
        return segments

def reconstruct(test_segments, test_ts, kmeans_model, rel_offset=True, stride=1):
    """
    Reconstruct a time series with segments derived from centroids of k-means clusters. Clusters must be fitted in n-dimensional space to time series segments of length n.
    
    test_segments = segments of the time series to be reconstructed. Use stride equal to the segment length to make these.
    test_ts = the original time series that is to be reconstructed
    kmeans_model = sklearn.cluster.KMeans object that has been fit to the training segments
    rel_offset = offset the reconstructed time series to start at time zero
    stride = needed when time stamps are not provided (i.e. test_segments are 1 dimensional). Defaults to the segment length.
    """
    error=0
    centroids=kmeans_model.cluster_centers_
    if np.shape(test_segments)[1] == 2:
        reco= np.zeros(np.shape(test_ts))
        if rel_offset == True:
            ts_time=np.copy(test_ts[0])-test_ts[0][0]
        else:
            ts_time=np.copy(test_ts[0])
        reco[0]=np.copy(ts_time)
        for n_seg, segment in enumerate(test_segments):
            start=np.where(ts_time==segment[0][0])[0][0]
            end=int(start+len(segment[0]))
            reco_seg=reco[1][start:end]
            pred_centroid_index=kmeans_model.predict(np.array(segment[1]).reshape(1, -1))[0]
            pred_centroid=centroids[pred_centroid_index][0:len(reco_seg)]
            ###
            #scaling of the predicted centroid in the y direction to the standard deviation of the original segment
            
            scaled_centroid=pred_centroid
            std_ori=np.std(np.array(test_segments[n_seg,1]))
            mean_ori=np.mean(np.array(test_segments[n_seg,1]))
            std_pred=np.std(pred_centroid)
            mean_pred=np.mean(pred_centroid)
            scaled_centroid=mean_ori+(pred_centroid-mean_pred)*(std_ori/std_pred)
            
            
            ###
            reco[1,start:end]+=scaled_centroid#*window_sin            
        return "test this"#reco
    else:
        reco= np.zeros(len(test_ts))
        for n_seg, segment in enumerate(test_segments):
            pred_centroid_index=kmeans_model.predict(np.array(segment).reshape(1, -1))[0]
            pred_centroid=centroids[pred_centroid_index]
            scaled_centroid=np.copy(pred_centroid)
            start=n_seg*stride
            end=start+len(segment)
            reco[start:end]+=scaled_centroid
        return reco
    
def analyse(validation_results, k_clusters, seg_lens, save_histograms=False, save_grid=False):
    """
    
    """
    
    #create a directory for the plots
    if save_histograms == True or save_grid == True:
        results_dir=os.getcwd()+"/"+validation_results.split(".")[0]
        os.system("mkdir {}".format(results_dir))

    if type(validation_results) == str:
        results = np.loadtxt(validation_results, dtype=float, delimiter=",")
    else:
        results = validation_results
    
    counts=np.zeros((3, len(k_clusters)+1,len(seg_lens)+1))
    counts[0, 1:,0]=np.array(k_clusters).T
    counts[0, 0,1:]=np.array(seg_lens)
    counts[1, 1:,0]=np.array(k_clusters).T
    counts[1, 0,1:]=np.array(seg_lens)
    counts[2, 1:,0]=np.array(k_clusters).T
    counts[2, 0,1:]=np.array(seg_lens)
    for k_id, k_cluster in enumerate(k_clusters):
        for len_id, seg_len in enumerate(seg_lens):
            ordinary=results[(results[:,0]==k_cluster) & (results[:,1]==seg_len) & (results[:,2]==0)]
            max_ordinary=np.max(ordinary[:,-1])
            outliers=results[(results[:,0]==k_cluster) & (results[:,1]==seg_len) & (results[:,2]==1)]
            min_outlier=np.min(outliers[:,-1])
            counter_fn=0
            for t in outliers[:,-1]:
                if t<max_ordinary:
                    counter_fn+=1
            
            counter_fp=0
            for t in ordinary[:,-1]:
                if t>min_outlier:
                    counter_fp+=1
            counts[0, k_id+1,len_id+1]=counter_fn
            counts[1, k_id+1,len_id+1]=counter_fp
            
            tp=(len(outliers)-counter_fn)
            tn=(len(ordinary)-counter_fp)
            precision = tp/(tp+counter_fp)
            recall = tp/len(outliers)
            accuracy = (tp+tn)/(len(outliers)+len(ordinary))
            if precision * recall == 0:
                F1 = 0
            else:
                F1 = 2 * (precision * recall) / (precision + recall)
            counts[2, k_id+1,len_id+1]=F1
            
            if save_histograms == True:
                f = plt.figure()
                ax = f.add_subplot(111)
                plt.hist(ordinary[:,-1])
                plt.hist(outliers[:,-1],alpha=0.5)
                plt.text(0.85,0.75,"k= {}\nlen= {}\noverlap= {}\nfn= {}\nfp= {}\nprecision = {}\nrecall = {}\naccuracy = {}\nF1 = {}".format(k_cluster, seg_len, round(min_outlier-max_ordinary,1), counter_fn, counter_fp, round(precision,3), round(recall,3), round(accuracy,3), round(F1,3)), ha='center', va='center', transform=ax.transAxes)
                plt.savefig("{}.png".format(results_dir+"/"+"k{}_len{}".format(k_cluster, seg_len)))
                plt.close()
    
    F1s = counts[2, :,:]
    F1s[1:,1:] = F1s[1:,1:]*1000
    F1s = F1s.astype(int)
    
    if save_grid == True:
        
        np.savetxt("{}/grid_search.csv".format(results_dir), F1s, delimiter=",", fmt='%i') 
        
    
    return counts

def validate_algorithm(ordinary_file, outlier_file, k_clusters, seg_lens, save_results=True):
    """
    Function performs segmentation and clustering on 75% of the ordinary_file time series. 
    Then it reconstructs the remaining 25% of ordinary_file and all of the outlier_file time series.
    Process is repeated for every combination of k_clusters and seg_lens parameters.
    The output is an array of reconstruction error values; [k_clusters_index, seg_lens_index, classification, time_series_id, error]
    
    """

    process_t0 = time.process_time()
    real_t0 = time.time()
    
    ordinary_train, ordinary_valid, ordinary_train_ids, ordinary_valid_ids= train_test_split(ordinary_file, np.array(range(len(ordinary_file))) ,test_size=0.25, random_state=0)
    outlier_ids = np.array(range(len(outlier_file)))+len(ordinary_file)
    
    validation_data = np.vstack((ordinary_valid, outlier_file)) # stack validation data of ordinary and outlier time series
    validation_labels = np.hstack((np.zeros(len(ordinary_valid)), np.ones(len(outlier_file)))).T # generate labels for valdiation data
    

    validation_result = np.zeros((int(len(validation_data)*len(k_clusters)*len(seg_lens)),5))
    loop_counter = 0
    for k_id, k_cluster in enumerate(k_clusters):
        for len_id, seg_len in enumerate(seg_lens): #for every combination of hyperparameters
            tsscod = TSSCOD(k_clusters = k_cluster, seg_len = seg_len) #initialise outlier detection class

            tsscod.train(ordinary_train, random_state = 0) # train on the subset with no outliers; segment each series with a slide of 1 and cluster the segments
            
            start_index = int(loop_counter*len(validation_data))
            validation_result[start_index:start_index+len(validation_data), 4] = tsscod.validate(validation_data) #validation reconstructs the provided series and saves error
            validation_result[start_index:start_index+len(validation_data), 3] = np.hstack((ordinary_valid_ids, outlier_ids))
            validation_result[start_index:start_index+len(validation_data), 2] = validation_labels
            validation_result[start_index:start_index+len(validation_data), 1] = seg_len # add indices of hyperparameters to the results array; this part makes it easier to feed the result into the analysis pipeline
            validation_result[start_index:start_index+len(validation_data), 0] = k_cluster

            loop_counter += 1
            print("Hyperparameter sets completed: {}/{}, ".format(loop_counter, int(len(k_clusters)*len(seg_lens))) + "elapsed CPU time: {}s".format(time.process_time()-process_t0))
    
    if save_results == True:
        file_name = "validation_results_{}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        np.savetxt(file_name, validation_result, delimiter=",") 
        return file_name, validation_result
    
    print()
    print("Finished, elapsed time: {}s".format(time.time() - real_t0)+", total CPU time: {}s".format(time.process_time()-process_t0))
    
    return validation_result

def plot_time_series(time_series_file):
    fig, axes = plt.subplots(nrows=5, ncols=1)
    for index, time_series in enumerate(time_series_file[0:5]):
        axes[index].plot(time_series)
        axes[index].get_xaxis().set_visible(False)
    axes[-1].get_xaxis().set_visible(True)
    fig.tight_layout()
    fig.show()

class TSSCOD():
    """Time series segmentation, clustering, outlier detection
    """
    def __init__(self, k_clusters, seg_len):
        self.k_clusters = k_clusters
        self.seg_len = seg_len
        
    def train(self, training_data, stride=1, random_state=None):
        """Segment the set of ordinary time series and cluster the segments.
        """
        if len(np.shape(training_data))==2:#check if time data is provided
            time_stamps=False
        elif len(np.shape(training_data))==3:
            time_stamps=True
        else:
            raise ValueError("Time series must be 1D or 2D arrays.")
        
        all_train_segments=[]#loop throught the light curves of a given class and segments them
        for time_series in training_data:
            train_segments=segmentation(time_series, 
                                        self.seg_len, 
                                        stride,
                                        time_stamps=time_stamps)
            all_train_segments.append(train_segments)
        all_train_segments=np.vstack(all_train_segments)
        self.cluster=KMeans(n_clusters=self.k_clusters,
                            random_state=random_state)#cluster the segments
        self.cluster.fit(all_train_segments)
        return self
    
    def reconstruct(self, time_series, output="both"):
        """Reconstruct a time series
        """
        if len(np.shape(time_series))==1:#check if time data is provided
            time_stamps=False
        elif len(np.shape(time_series))==2:
            time_stamps=True
        else:
            raise ValueError("Time series must be 1D or 2D arrays.")
        
        outputs = ["reconstruction", "error", "both"]
        if output not in outputs:
            raise ValueError("Unavailable output argument. Available outputs: {}.".format(outputs))
            
        seg_len = self.seg_len
        stride = self.seg_len
            

        segments = segmentation(time_series, 
                                seg_len, 
                                stride, 
                                time_stamps = False)

        reco = reconstruct(segments, 
                           time_series, 
                           self.cluster, 
                           rel_offset=False, 
                           stride=stride)
            

        error = np.mean(((reco[0:-seg_len]-time_series[0:-seg_len]))**2.0)
        
        if output == "both":
            return reco, error
        elif output == "reconstruction":
            return reco
        elif output == "error":
            return error
    
    def validate(self, validation_data):
        """Reconstruct a bunch of time series and save errors
        """
        reco_errors = np.zeros(len(validation_data))
        
        for series_index, time_series in enumerate(validation_data):
            reco_errors[series_index] = self.reconstruct(time_series, output = "error")
            
        return reco_errors