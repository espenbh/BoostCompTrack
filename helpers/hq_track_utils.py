import numpy as np
import os
import cv2
import imutils
import sys
base = os.getcwd()
sys.path.append(base + '\\helpers')
import keybox_utils as ku

def remove_incomplete_salmon(trackers, config, verbose = True, ncomp = 9, size_thresh = 600):
    '''
    Function to remove instances salmon instances if the bounding box diagonal is below size_thresh,
    or any body parts are hidden.
    
    Args:
        trackers: A numpy array where each row is on the format [frame_num, ID, x, y, w, h, conf, -1, -1, -1, comp type]
        config: Tracking config file
        verbose: Whether to print processed frame number
        ncomp: Number of components in the tracker data
        size_thresh: Minimum diagonal salmon length
    Returns:
        trackers: Trackers that have passed the pruning process
        id_cnt_dict: A dictionary with the number of frames each salmon is visible
    '''
    if verbose:
        print('Removing incomplete salmon')
    id_cnt_dict = {id:0 for id in list(set(list(trackers[:,1])))}
    accepted_trk = []
    for frame_num in range(config['start_frame'], config['end_frame']+1):
        if verbose:
            print(frame_num)
        frame_trackers = trackers[trackers[:,0]==str(frame_num)]
        ids = list(set(list(frame_trackers[:,1])))
        for id in ids:
            id_trks = frame_trackers[frame_trackers[:,1]==id]
            if id_trks.shape[0] == ncomp:
                accepted_trk.append(id_trks)
                salmon_wh = id_trks[id_trks[:,10]=='salmon'][0][4:6].astype(float)
                salmon_diag = np.sqrt(salmon_wh[0]**2+salmon_wh[1]**2)
                if salmon_diag > size_thresh:
                    id_cnt_dict[id] = id_cnt_dict[id]+1
            
    trackers = np.vstack(accepted_trk)
    return trackers, id_cnt_dict

def remove_small_salmon(trackers, config, id_cnt_dict, num_thresh = 10, verbose = True):
    '''
    Function to remove all salmon tracks containing under num_thresh salmon
    
    Args:
        trackers: A numpy array where each row is on the format [frame_num, ID, x, y, w, h, conf, -1, -1, -1, comp type]
        config: Tracking config file
        id_cnt_dict: Dictionary with the counts of all IDs
        num_thresh: Minimum number of frames in a track
    Returns:
        trackers: Trackers that have passed the pruning process
    '''
    if verbose:
        print('Removing small salmon')
    valid_ids = [int(k) for k,v in id_cnt_dict.items() if v >= num_thresh]
    valid_ids.sort()

    acc_trk = []
    for frame_num in range(config['start_frame'], config['end_frame']+1):
        frame_trackers = trackers[trackers[:,0]==str(frame_num)]
        for id in valid_ids:
            id_trks = frame_trackers[frame_trackers[:,1]==str(id)]
            acc_trk.append(id_trks)
    trackers = np.vstack(acc_trk)
    return trackers

def remove_non_consecutive_tracks(trackers, num_thresh = 10, verbose = True):
    '''
    Function to remove all non-consecutive salmon tracks. If the longest, consecutive track is below 
    num_thresh in length, this track is also removed.
    
    Args:
        trackers: A numpy array where each row is on the format [frame_num, ID, x, y, w, h, conf, -1, -1, -1, comp type]
        num_thresh: Minimum number of frames in a track
    Returns:
        trackers: Trackers that have passed the pruning process
        valid_ids: IDs that has passed the pruning process
    '''
        
    if verbose:
        print('Removing all frame detections that are not part of the longest, consecutive track')
    valid_ids = []
    salmon_ids = list(set(list(trackers[:,1])))
    salmon_ids.sort()
    acc_trackers = []
    for salmon_id in salmon_ids:
        if verbose:
            print(salmon_id)
        frames = [int(f) for f in list(set(list(trackers[trackers[:,1]==salmon_id][:,0])))]
        frames.sort()
        frames = np.array(frames)
        diffs = np.diff(frames)  # Compute differences between consecutive elements
        split_indices = np.where(diffs > 1)[0] + 1  # Find indices where the sequence breaks
        sublists = np.split(frames, split_indices)  # Split the array into sublists
        longest = max(sublists, key=len)  # Find the longest sublist
        if len(longest) < num_thresh:
            continue
        else:
            valid_ids.append(salmon_id)
        id_trks = trackers[trackers[:,1]==salmon_id]
        for frame_num in longest:
            frame_trackers = id_trks[id_trks[:,0]==str(frame_num)]
            acc_trackers.extend(list(frame_trackers))
    trackers = np.array(acc_trackers)
    return trackers, valid_ids
