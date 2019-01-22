import numpy as np
#segmentation of the ts
def segmentation(ts, seg_len, seg_slide, time_stamps=True):
    """ts=time series, seg_len=size of the moving window, seg_slide=difference in the starting position of the consecutive windows"""
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

#multiplication of the segments by a waveform to emphesise the features in the centre and zero the ends so that the segments can be joined smoothely together
def center_offset(segments, ts, time_stamps=True, offset=True):
    """Use cluster.fit(np.array(c_train_segments)[:,1]) with the time stamped output"""
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

#use the kmeans clusters trained on the centered segments to rebuild a time series 
def reconstruct(test_segments, test_ts, kmeans_model, rel_offset=True, seg_slide=25):
    """seg_slide needed when time stamps are not provided (i.e. 1D fragments),"""
    centroids=kmeans_model.cluster_centers_
    if np.shape(test_segments)[1] == 2:
        reco= np.zeros(np.shape(test_ts))
        reco[0]=np.copy(test_ts[0])
        if rel_offset == True:
            ts_time=np.copy(test_ts[0])-test_ts[0][0]
        else:
            ts_time=np.copy(test_ts[0])
        for n_seg, segment in enumerate(test_segments):
            start=np.where(ts_time==segment[0][0])[0][0]
            print(start)
            end=int(start+len(segment[0]))
            reco_seg=reco[1][start:end]
            #print(start, end)
            pred_centroid=kmeans_model.predict(np.array(segment[1]).reshape(1, -1))[0]
            reco[1,start:end]+=centroids[pred_centroid][0:len(reco_seg)]            
            #print(centroids[pred_centroid][0:len(reco_seg)])
            
            
            
            # # seg_start_t=segment[0,0]
            # # exp_t=n_seg*seg_slide+offset_val
            # # correction=seg_start_t-exp_t
            # print(correction)
            # pred_centroid=kmeans_model.predict(np.array(segment[1]).reshape(1, -1))[0]
            # start=int(segment[0,0]-offset_val-correction)
            # end=int(segment[0,-1]+1-offset_val-correction)
            # #print(segment[0,0], start,end, correction)
            # #print(prev_seg_start, seg_start, start, end, correction)
            # # if reco[0,start]-start != correction:
            # #     correction=reco[0,start]-start
            # #     start=int(segment[0,0]-offset_val-correction)
            # #     end=int(segment[0,-1]+1-offset_val-correction)
            # reco_seg=reco[1][start:end]
            # reco[1,start:end]+=centroids[pred_centroid][0:len(reco_seg)]
        return reco
    else:
        reco= np.zeros(np.shape(test_ts)[1])
        for n_seg, segment in enumerate(test_segments):
            pred_centroid=kmeans_model.predict(np.array(segment).reshape(1, -1))[0]
            start=n_seg*seg_slide
            end=start+len(segment)
            reco[start:end]+=centroids[pred_centroid]
        return reco