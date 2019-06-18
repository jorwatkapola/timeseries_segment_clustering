import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import zscore
import sys

#time series and light curve can be used interchangebly below
def segmentation(ts, seg_len, seg_slide, time_stamps=True):
    """ creates a list of 1D (when time_stamps=False) or 2D (when time_stamps=True) arrays, which are overlappig fragments of ts. Incomplete fragments are rejected.
    ts=time series to be segmented
    seg_len=size of the moving window, 
    seg_slide=difference in the starting position of the consecutive windows"""
    segments=[]
    if time_stamps==True:
        for start in range(0, len(ts[0])-seg_len, seg_slide):
            end=start+seg_len
            if ts[0][end]-ts[0][start] != seg_len: ####don't allow segments with missing data
                continue
            segments.append(np.copy(ts[:,start:end]))
        return np.array(segments)
    else:
        for start in range(0, len(ts)-seg_len, seg_slide):
            end=start+seg_len
            segments.append(np.copy(ts[start:end]))
        return np.array(segments)

def center_window(segments, ts, time_stamps=True, offset=True):
    """multiplies the segments by a waveform to emphesise the features in the centre and zero the ends so that the segments can be joined smoothly together. Use cluster.fit(np.array(c_train_segments)[:,1]) on the time stamped output
    segments = segmented time series, the output of segmentation function
    ts = the original time series
    time_stamps = set to False if the input segments are 1 dimensional
    offset = offset the time stamps as if the time series started at time zero (not sure if this is needed any more...)"""
    c_segments=[]
    if time_stamps==True:
        window_rads = np.linspace(0, np.pi, len(segments[0][0]))
        window_sin = np.sin(window_rads)**2
        #window_sin = np.sin(window_rads)**(1/2)
        #window_sin = window_rads*0+1
        if offset==True:
            for segment in segments:
                segment[1]*=window_sin
                segment[0]-=ts[0][0]
                c_segments.append(segment)
            return c_segments
        else:
            for segment in segments:
                segment[1]*=window_sin
                c_segments.append(segment)
            return c_segments
    else:
        window_rads = np.linspace(0, np.pi, len(segments[0]))
        window_sin = np.sin(window_rads)**2
        for segment in segments:
            segment*=window_sin
            c_segments.append(segment)
        return c_segments

def reconstruct(test_segments, test_ts, kmeans_model, rel_offset=True, seg_slide=25):
    """function uses the kmeans clusters trained on the centered segments to rebuild a time series
    test_segments = the output of center_offset function applied time series to be reconstructed
    test_ts = the original time series that is to be reconstructed
    kmeans_model = sklearn.cluster.KMeans object that has been fit to the training segments
    rel_offset = offset the reconstructed time series to start at time zero
    seg_slide = needed when time stamps are not provided (i.e. test_segments are 1 dimensional)"""
    error=0
    centroids=kmeans_model.cluster_centers_
    if np.shape(test_segments)[1] == 2:
        # window_rads = np.linspace(0, np.pi, len(test_segments[0][0]))
        # window_sin = np.sin(window_rads)**2
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
        # window_rads = np.linspace(0, np.pi, len(test_segments[0]))
        # window_sin = np.sin(window_rads)**2
        reco= np.zeros(len(test_ts))
        for n_seg, segment in enumerate(test_segments):
#             std_ori=np.std(np.array(segment))
#             mean_ori=np.mean(np.array(segment))
#             scaled_centroids=np.copy(centroids)
#             scaled_centroids=mean_ori+(pred_centroid-mean_pred)*(std_ori/std_pred)
            
            
            pred_centroid_index=kmeans_model.predict(np.array(segment).reshape(1, -1))[0]
            pred_centroid=centroids[pred_centroid_index]
            scaled_centroid=np.copy(pred_centroid)
            
            
#             std_ori=np.std(np.array(test_segments[n_seg]))
#             mean_ori=np.mean(np.array(test_segments[n_seg]))
#             std_pred=np.std(pred_centroid)
#             mean_pred=np.mean(pred_centroid)
#             scaled_centroid=mean_ori+(pred_centroid-mean_pred)*(std_ori/std_pred)
            
            
            start=n_seg*seg_slide
            end=start+len(segment)
            reco[start:end]+=scaled_centroid#*window_sin
            
            
            
            
            
        return reco


class TSSCOD():
    """Time series segmentation, clustering, outlier detection
    """
    def __init__(self, k_clusters, seg_len):
        self.k_clusters = k_clusters
        self.seg_len = seg_len
        
    def train(self, training_data, seg_slide=1, random_state=None):
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
                                        seg_slide,
                                        time_stamps=time_stamps)
            all_train_segments.append(train_segments)
        all_train_segments=np.vstack(all_train_segments)
        print("all_train_segments", sys.getsizeof(all_train_segments))
        self.cluster=KMeans(n_clusters=self.k_clusters,
                            random_state=random_state)#cluster the segments
        self.cluster.fit(all_train_segments)
        print("cluster", sys.getsizeof(self.cluster))
        return self
    
    def reconstruct(self, time_series):
        """Reconstruct a time series
        """
        if len(np.shape(time_series))==1:#check if time data is provided
            time_stamps=False
        elif len(np.shape(time_series))==2:
            time_stamps=True
        else:
            raise ValueError("Time series must be 1D or 2D arrays.")
            
        seg_len = self.seg_len
        seg_slide = self.seg_len
            

        segments = segmentation(time_series, 
                                seg_len, 
                                seg_slide, 
                                time_stamps = False)

        reco = reconstruct(segments, 
                           time_series, 
                           self.cluster, 
                           rel_offset=False, 
                           seg_slide=seg_slide)
            
        reco[0:-seg_len]=zscore(reco[0:-seg_len])
        expectation=zscore(np.copy(time_series[0:-seg_len]))
        error = np.mean(((reco[0:-seg_len]-expectation))**2.0)

        return reco, error
    
    def validate(self, validation_data, labels):
        """Reconstruct a bunch of time series and save errors
        """
        unique_labels = np.unique(labels)
        reco_error = np.zeros((len(validation_data), 3))
        validation_counter=0
        for label_index, label in enumerate(unique_labels):
            single_class_indices = np.where(labels == label)[0]
            for single_class_index in single_class_indices:
                time_series = validation_data[single_class_index]
                reco, error = self.reconstruct(time_series)
                reco_error[validation_counter, 0] = label_index
                reco_error[validation_counter, 1] = single_class_index
                reco_error[validation_counter, 2] = error
                validation_counter+=1
        
        return reco_error