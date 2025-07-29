import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
base = os.getcwd()
sys.path.append(base)
sys.path.append(base + '\\helpers')

from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from helpers import *
import keybox_utils as ku
import pandas as pd

def create_data_for_GT_labeling(trackers, config, imgs_folder, only_salmon=True):
    '''
    Set up data that can be labeled with tail beat extremes.
    Args:
        trackers: Trackers on the standard MOT format
        config: Tracking config file
        imgs_folder: Folder where everything should be saved
        only_salmon: Only annotate salmon (not body parts)
    Operations:
        Saves all frames to imgs_folder, annotated with bounding boxes around all objects
        Crops out salmon, and saves them to ID-spesific subfolders
    '''
    if only_salmon:
        plot_trackers = trackers[trackers[:, 10] == 'salmon']
    else:
        plot_trackers = trackers

    config['bp_bbox_thickness'] = 6
    config['salmon_bbox_thickness'] = 10

    frames = list(set(list(plot_trackers[:, 0].astype(int))))
    frames.sort()
    video = cv2.VideoCapture(config['video_path'])
    video.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    frame_num = frames[0]
    end_frame = frames[-1]

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(imgs_folder, 'annotated_output.mp4')
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
    fps = video.get(cv2.CAP_PROP_FPS)
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while frame_num <= end_frame:
        succ, frame = video.read()
        if not succ:
            break

        frame_trackers = plot_trackers[plot_trackers[:, 0] == str(frame_num)]
        if frame_trackers.shape[0] > 0:
            for tracker in frame_trackers:
                obj_id = tracker[1]
                xywh = tracker[2:6].astype(float)
                x1, y1, x2, y2 = map(int, ku.xywh2xyxy(xywh))
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0]*crop.shape[1]*crop.shape[2] == 0:
                    continue

                id_folder = os.path.join(imgs_folder, str(obj_id))
                os.makedirs(id_folder, exist_ok=True)

                crop_filename = os.path.join(id_folder, f"{frame_num}.png")
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                plt.imsave(crop_filename, crop_rgb)

            aframe = annotate_frame_mot(frame_trackers, frame, frame_num, config)
        else:
            aframe = frame

        aframe = cv2.resize(aframe, (frame_width, frame_height))
        aframe_rgb = cv2.cvtColor(aframe, cv2.COLOR_BGR2RGB)
        plt.imsave(os.path.join(imgs_folder, f"{frame_num}.png"), aframe_rgb)

        # Write frame to video
        out_video.write(cv2.cvtColor(aframe_rgb, cv2.COLOR_RGB2BGR))

        frame_num += 1

    video.release()
    out_video.release()


def load_gt(path):
    '''
    Load salmon tail beat extreme ground truth data
    Args:
        path: Path to ground truth data
    Returns:
        Loaded ground truth data
    '''
    df = pd.read_excel(path)
    gt = np.array(df)
    gt_dict = {}
    for g in gt:
        gt_dict[g[0][2:]] = []
        for i in g[1:]:
            if type(i) != float:
                gt_dict[g[0][2:]].append([i[0], int(i[1:])])
    return gt_dict


def plot_ax(ax, data_dict, data_smoothed, data_maxima, data_minima, color = 'green'):
    """
    Plots smoothed data, its maxima and minima, and original data points on a given matplotlib axis.
    Args:
        ax: matplotlib axis
        data_smoothed: Time series smoothed by Savitsky-Golay filter
        data_maxima: Maxima indices
        data_minima: Minima indices
        color: Plot color
    Returns:
        Annotated axis
    """
    
    x_all = np.array(range(min(list(data_dict.keys())), max(list(data_dict.keys()))+1))
    ax.plot(x_all, data_smoothed, label='', color=color, linewidth=3, zorder=1)
    ax.scatter(x_all[data_maxima], data_smoothed[data_maxima], color='red', label='maxima', s = 100, zorder = 3)
    ax.scatter(x_all[data_minima], data_smoothed[data_minima], color='blue', label='minima', s = 100, zorder = 3)
    ax.scatter([float(i) for i in list(data_dict.keys())], [float(i) for i in list(data_dict.values())], color=color, zorder = 2)
    ax.tick_params(axis='y', labelcolor=color)
    ax.tick_params(axis='both', labelsize=16) 
    return ax

def get_single_track_metrics(hyp, gt, threshes = [1,2,3,4], id=''):
    '''
    Get false positives, false negatives and true positives counts 
    from a single time series and a single representation. 
    Each thresh is a seperate value in the outputed lists for each event.
    Args:
        hyp (list): frame number of hypothesis extremes
        gt (list): frame number of ground truth extremes
        threshes: Maximum frame number difference before considered a match
        id: Salmon ID
    Returns:
        FPs: list of number of false positives for each thresh
        FNs: list of number of false positives for each thresh
        TPs: list of number of false positives for each thresh
        ret_hyp_ind: Hypothesis indices matched given a frame threshold of 2.
    '''
    ret_hyp_ind = None
    FNs, FPs, TPs = [], [], []
    for thresh in threshes:
        cm = np.zeros([len(gt), len(hyp)])
        for i in range(len(gt)):
            for j in range(len(hyp)):
                c = np.abs(gt[i]-hyp[j])
                cm[i][j] = c

        gt_ind, hyp_ind = [], []
        while True:
            min_index = np.unravel_index(np.argmin(cm), cm.shape)
            if cm[min_index] > thresh:
                break
            if cm[min_index] <= thresh and min_index[0] not in gt_ind and min_index[1] not in hyp_ind:
                gt_ind.append(min_index[0])
                hyp_ind.append(min_index[1])
                cm[min_index] = 100
            else:
                cm[min_index] = 100
                continue
        if thresh == 2:
            ret_hyp_ind = hyp_ind

        FNs.append(abs(len(gt_ind) - len(gt)))
        FPs.append(abs(len(hyp_ind) - len(hyp)))
        TPs.append(len(hyp_ind))
    return FPs, FNs, TPs, ret_hyp_ind


def angle_between_points(a, b, c):
    '''
    Angle between points a, b, and c (b being the root)
    '''
    # a, b, c are np.array([x, y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Clamp value to avoid numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def binary_search_prominence(smoothed, num_peaks, max_iter=50, tol=1, verbose = 0, id=''):
    '''
    Attempts to match the predicted number of peaks to the ground truth number of peaks.
    Args:
        Smoothed: Smoothed time series data
        num_peaks: Number of grund truth peaks
        max_iter: Maximum number of iterations of the binary search
        tol: Allowable difference between the ground truth extrema count and the hypothesis extrema count
        verbose: Whether to print results
        id: Salmon ID
    Returns:
        best_prom: Best prominence parameter
        maxima: Found maxima
        minima: Found minima
    '''
    low = 1e-6
    high = np.max(smoothed) - np.min(smoothed)
    best_prom = None

    for i in range(max_iter):
        prom = (low + high) / 2
        maxima, _ = find_peaks(smoothed, prominence=prom)
        minima, _ = find_peaks(-smoothed, prominence=prom)
        total_extrema = len(maxima) + len(minima)

        if verbose >= 2: print(f"Iteration {i}: Prominence = {prom}, Extrema = {total_extrema}")

        if abs(total_extrema - num_peaks) <= tol:
            best_prom = prom
            break

        if total_extrema > num_peaks:
            low = prom  # Need more prominence to reduce peaks
        else:
            high = prom  # Need less prominence to increase peaks

    if i == max_iter-1 and abs(total_extrema-num_peaks)>=1 and verbose >= 1:
        print('Number of peaks deviate from GT by ' + str(abs(total_extrema-num_peaks)) + ' extrema for id ' + str(id))

    return best_prom, maxima, minima

def find_tailbeat(data_dict, height = None, threshold = None, distance = None, prominence = None, width = None, rel_height = None, plateau_size = None, wlen = None, num_peaks=None, id = ''):
    '''
    Find tailbeat extremes from tail beat time series.
    Args:
        data_dict: Dictionary with time series data
        num_peaks: Number of grund truth peaks
        id: Salmon id
        The rest of the arguments are directly passed to the find_peaks function, given a num_peaks of None
    Returns
        maxima: Found maxima
        minima: Found minima
        smoothed: Smoothed time series
    '''

    data_interp = np.array([data_dict[frame] for frame in data_dict.keys()]).copy()
    nans = np.isnan(data_interp)
    if np.any(~nans):
        data_interp[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), data_interp[~nans])

    # Apply Savitzky-Golay filter (window length must be odd and <= len(dists_interp))
    window_length = min(11, len(data_interp) if len(data_interp)%2==1 else len(data_interp)-1)
    smoothed = savgol_filter(data_interp, window_length=window_length, polyorder=2)

    if num_peaks is None:
        # # Find local peaks in the smoothed signal
        maxima, _ = find_peaks(smoothed, height, threshold, distance, prominence, width, rel_height, plateau_size, wlen)

        # Find local minima by inverting the smoothed signal
        minima, _ = find_peaks(-smoothed, height, threshold, distance, prominence, width, rel_height, plateau_size, wlen)
        return maxima, minima, smoothed
    else:
       best_prom, maxima, minima =  binary_search_prominence(smoothed, num_peaks, max_iter=50, tol=0, id=id)
       

    return maxima, minima, smoothed

def line_intersection(line1, line2):
    '''
    Find the intersection point of two lines.
    Args:
        line: Two endpoints of a line segment ([x, y], [x, y])
    returns:
        The intersection point; (x,y)
    '''
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y