import numpy as np
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
        return segments
    else:
        for start in range(0, len(ts[0])-seg_len, seg_slide):
            end=start+seg_len
            if ts[0][end]-ts[0][start] != seg_len: ####don't allow segments with missing data
                continue
            segments.append(np.copy(ts[1][start:end]))
        return segments

def center_offset(segments, ts, time_stamps=True, offset=True):
    """multiplies the segments by a waveform to emphesise the features in the centre and zero the ends so that the segments can be joined smoothly together. Use cluster.fit(np.array(c_train_segments)[:,1]) on the time stamped output
    segments = segmented time series, the output of segmentation function
    ts = the original time series
    time_stamps = set to False if the input segments are 1 dimensional
    offset = offset the time stamps as if the time series started at time zero (not sure if this is needed any more...)"""
    c_segments=[]
    if time_stamps==True:
        window_rads = np.linspace(0, np.pi, len(segments[0][0]))
        window_sin = np.sin(window_rads)**2
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
            pred_centroid=kmeans_model.predict(np.array(segment[1]).reshape(1, -1))[0]
            reco[1,start:end]+=centroids[pred_centroid][0:len(reco_seg)]            
        return reco
    else:
        reco= np.zeros(np.shape(test_ts)[1])
        for n_seg, segment in enumerate(test_segments):
            pred_centroid=kmeans_model.predict(np.array(segment).reshape(1, -1))[0]
            start=n_seg*seg_slide
            end=start+len(segment)
            reco[start:end]+=centroids[pred_centroid]
        return reco

def scaling(data, method, no_sigma=5, center="minimum"):
    """ Normalise or standardise the y-values of time series.
    method =    "normal" for normalisation y_i_norm = (y_i - y_center)/(y_max - y_min), where y_center is either y_mean or y_min as dictated                    by center argument
                "standard" for standardisation y_i_stand = (y_i - y_mean)/y_std
    no_sigma = the value of sigma to be assumed as the maximum value of y (to truncate the outliers).
    center =    "minimum" for min-max normalisation
                "mean" for mean normalisation
    """
    data_dims = np.shape(data[0])[0]
    all_counts=[]
    if data_dims == 2:
        for lc in data:
            all_counts.append(lc[1])
    else:
        all_counts=data
    all_counts_ar=np.concatenate(all_counts, axis=0)
    armean=np.mean(all_counts_ar)
    arstd=np.std(all_counts_ar)
    armedian=np.median(all_counts_ar)
    armin=np.min(all_counts_ar)
    armax=armean+no_sigma*arstd
    
    lcs_std=[]
    if method == "normal":
        if center == "minimum":
            center=armin
        elif center == "mean":
            center=armean
        else:
            print("{} is not a valid center".format(center))
            return
        if data_dims == 2:
            for ts in data:
                lc=np.copy(ts)
                lc[1]=(lc[1]-center)/(armax-armin)
                over_max=np.where(lc[1]>1.)[0]
                lc[1][over_max]=1.
                lcs_std.append(lc)
        else:
            for ts in data:
                lc=np.copy(ts)
                lc=(lc-center)/(armax-armin)
                over_max=np.where(lc>1.)[0]
                lc[over_max]=1.
                lcs_std.append(lc)
        return lcs_std
    
    elif method == "standard":
        if data_dims == 2:
            for ts in data:
                lc=np.copy(ts)
                lc[1]=(lc[1]-armean)/arstd
                lcs_std.append(lc)
        else:
            for ts in data:
                lc=np.copy(ts)
                lc=(lc-armean)/arstd
                lcs_std.append(lc)
        return lcs_std
    
    else:
        print("{} is not a valid method".format(method))
        return