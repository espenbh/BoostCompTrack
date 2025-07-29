import shutil
import yaml
import sys
import os
sys.path.append(os.path.split(os.getcwd())[0] + '\\salmon_component_tracking\\helpers\\')
sys.path.append(os.path.split(os.getcwd())[0] + '\\salmon_component_tracking\\evaluation\\py-motmetrics\\')
import motmetrics as mm
from file_utils import load_labelme_data_for_eval, load_txt_data_for_eval, load_txt_data_for_eval_SintefCam
from file_utils import write_MOT_results
import numpy as np
from collections.abc import Iterable
from plot_utils import plot_tracking_results
import keybox_utils as ku
from draw_utils import *


def set_up_motmetrics_folder(hyp_data_path, file_name = 'motmetrics_evaluation'):
    '''
    Set up a folder for motmetrics result in the analysis folder

    Args:
        hyp_data_path (str): The path to the hypothesis data (tracker data)
    Operations:
        Deletes the evaluation folder if it exists
    Returns:
        config dict): The tracker configuration file
        motmetrics_path (str): The path to the evaluation folder
    
    '''
    motmetrics_path = '\\'.join(hyp_data_path.split('\\')[:-1]) + '\\' + file_name 
    if file_name in os.listdir('\\'.join(hyp_data_path.split('\\')[:-1])):
        shutil.rmtree(motmetrics_path)
    os.mkdir(motmetrics_path)

    with open('\\'.join(hyp_data_path.split('\\')[:-1]) + '\\config.yml', 'r') as f:
        config = yaml.safe_load(f)
    return config, motmetrics_path

def write_objs_to_motmetrics_folder(obj_data_path, motmetrics_path, config, frame_offset = 14100, splitter = '_f', scramble_ids = False, components = [], only_load = False):
    '''
    Write a text file with object (Ground Truth) data to the evaluation folder.

    Args: 
        obj_data_root (str): The file root of the object data (labelme)
        motmetrics_path (str): The path to the evaluation folder
        config (dict): the tracker configuration file
        frame_offset: Offset between video frame number and label frame number
        splitter (str): String between video name and frame number in the file names of the images and labels
        scramble_ids (bool): If True, overwrite labled IDs such that all salmon have differen IDs
    Operations:
        Write obj.txt file to motmetrics_path. This file ocntains all objects (ground truth objects)
    Returns:
        objs (numpy array): All objects on the MOT format
    '''

    # Load object data
    if not obj_data_path.endswith('txt'):
        objs = load_labelme_data_for_eval(obj_data_path, config, frame_offset = frame_offset, splitter = splitter, scramble_ids=scramble_ids)
    else:
        objs = load_txt_data_for_eval(obj_data_path)

        # IDs should be without decimals
        objs[:,1] = objs[:,1].astype(float).astype(int).astype(str)
        kb = objs[:,2:6].astype(float)
        kb = np.vstack([kb[:,0]-kb[:,2]/2, kb[:,1]-kb[:,3]/2, kb[:,2], kb[:,3]]).T
        objs[:,2:6] = kb.astype(str)


    # Frame number should be integers, and start at 1
    objs[:,0] = objs[:,0].astype(int) + 1

    if len(components) > 0:
        updated_objs = []
        for c in components:
            updated_objs.append(objs[objs[:,10] == c])
        objs = np.concatenate(updated_objs)

    # Save object data
    if not only_load:
        write_MOT_results(objs[:,:10], motmetrics_path, 'obj.txt')

    return objs

def write_hyps_to_motmetric_folder(hyp_data_path, motmetrics_path, config, frame_offset = 14100, components = [], only_load = False):
    '''
    Write a text file with hypotheses (tracker) data to the evaluation folder.

    Args: 
        hyp_data_path (str): The file root of the hypothesis data (tracker data)
        motmetrics_path (str): The path to the evaluation folder
        config (dict): the tracker configuration file
        frame_offset: Offset between video frame number and label frame number
    Operations:
        Write hyp.txt file to motmetrics_path. This file ocntains all hypotheses (tracker objects)
    Returns:
        hyps (numpy array): All hypotheses on the MOT format
    '''
    # Load hypotheses
    hyps = load_txt_data_for_eval(hyp_data_path)

    # Subtract frame number with offset, such that the frame numbers for the objects and hypotheses are aligned
    hyps[:,0] = hyps[:,0].astype(int)-frame_offset

    # Discard hypotheses data not in saved_frames
    hyps = hyps[np.isin(hyps[:,0].astype(int), config['saved_frames'])]

    # Frame number should be integers, and start at 1
    hyps[:,0] = hyps[:,0].astype(int) + 1

    # IDs should be without decimals
    hyps[:,1] = hyps[:,1].astype(float).astype(int).astype(str)

    if len(components) > 0:
        updated_hyps = []
        for c in components:
            updated_hyps.append(hyps[hyps[:,10] == c])
        hyps = np.concatenate(updated_hyps)

    # Warning if no hypothesis objects are found
    if len(hyps) == 0:
        print('Warning: No hypothesis objects in the correct frames found')

    # Save tracker data
    if not only_load:
        write_MOT_results(hyps[:,:10], motmetrics_path, 'hyp.txt')

    return hyps

def compute_motchallenge(dir_name, th_list, iou_class_penalties = None):
    '''
    Calculate motmetrics accumulators based on generated MOT textfiles

    Args:
        dir_name (str): The location of the evaluation data
        th_list (list(float)): A list of all evaluation threshold. 
        The evaluation threshold sets the minimum amount of overlap between a ground truth object and a tracker hypothesis
        iou_class_penalities (dict(np.array)): The dictionary contain one matrix (value) for each frame number(key). 
        Each matrix has shape num_objects x num_hypotheses. All entries where the object and hypothesis class is the same is 1.
        All entries where the object and hypothesis class is different is 0.
    Returns:
        res_list (list[MOTAccumulator]): A list of motmetrics accumulators, 1 for each threshold.

    '''
    # Load objects and hypotheses
    # `gt.txt` and `test.txt` should be prepared in MOT15 format
    df_gt = mm.io.loadtxt(os.path.join(dir_name, "obj.txt"))
    df_test = mm.io.loadtxt(os.path.join(dir_name, "hyp.txt"))

    # Calculate accumulators
    res_list = mm.utils.compare_to_groundtruth_reweighting(df_gt, df_test, "iou", distth=th_list, iou_class_penalties = iou_class_penalties)

    return res_list

def calculate_iou_class_penalties(gt, trackers, config, only_consider_equal_classes):
    '''
    Calculate iou class penalties. 
    The iou_class_penalty is a matrix that, when multiplied to the iou matrix between objects and hypotheses, ensures no association between different classes.

    Args:
        gt (numpy array): A numpy array on the MOT format over all objects
        trackers (numpy array): A numpy array on the MOT format over all hypotheses
        config (dict): The tracking configuration file
        only_consider_equal_classes (bool): Decides whether objects of unequal classes should be assigned during evaluation
    Returns:
        iou_class_penalties (dict{int: numpy array}). A dictionary, where the keys are frame numbers, and the values are corresponding iou_class_penalty matrices.
        The iou_class_penalty matrices have shape num_objects x num_hypotheses. All entries where the object and hypothesis class is the same is 1.
        All entries where the object and hypothesis class is different is 0.   
    '''

    iou_class_penalties = {}
    for frame_num in config['saved_frames']:
        if only_consider_equal_classes:
            frame_trackers = trackers[trackers[:,0] == str(frame_num+1)]
            frame_gt = gt[gt[:,0] == str(frame_num+1)]
            iou_class_penalties[str(frame_num+1)] = np.zeros((frame_gt.shape[0], frame_trackers.shape[0]))
            for gti, i in zip(frame_gt, range(frame_gt.shape[0])):
                for trki, j in zip(frame_trackers, range(frame_trackers.shape[0])):
                    if gti[10] == trki[10]:
                        iou_class_penalties[str(frame_num+1)][i,j] = 1
        else:
            iou_class_penalties[str(frame_num+1)] = None
    return iou_class_penalties


def evaluate_tracker_motmetrics(obj_data_path, hyp_data_path, 
                                only_consider_equal_classes = True,
                                frame_offset = 14100, 
                                splitter = '_f', 
                                scramble_ids = False, 
                                metrics = ["hota_alpha",'recall', 'precision', 'num_false_positives', 'num_misses', 'num_switches', 'num_transfer','num_matches','num_ascend','num_migrate'],
                                components = [],
                                results_file_name = 'results'
                                ):
    '''
    Create plot and save text file of the tracking results.

    Args:
        obj_data_root (str): The file root of the object data
        hyp_data_path (str): The path to the tracking data
        only_consider_equal_classes (bool): Decides whether objects of unequal classes should be assigned during evaluation
        frame_offset: Offset between video frame number and label frame number
        splitter (str): String between video name and frame number in the file names of the images and labels
        scramble_ids (bool): If True, overwrite labled IDs such that all salmon have differen IDs
        metrics (list[str]): List over metrics to calculate
    Operations:
        Save text file of the results to disk
        Save plot of the results to disk
    Returns:
        acc (list[MOTAccumulator]): List of accumulators
        summary (pandas array): A pandas array of all MOT metrics at all IOU thresholds
        th_list (list[float]): A list of values between 0 and 1, specifying the minimum IoU limit for the accumulators at the same index
        motmetrics_path (str): Path to the motmetrics results
        trackers (numpy array): A numpy array containing all trackers (hypotheses) in the MOT format
        gt (numpy array): A numpy array containing all ground truth objects in the MOT format

    '''

    # Calculate motmetrics accumulators
    config, motmetrics_path = set_up_motmetrics_folder(hyp_data_path, 'motmetrics_evaluation_' + results_file_name)
    gt = write_objs_to_motmetrics_folder(obj_data_path, motmetrics_path, config, frame_offset = frame_offset, splitter = splitter, scramble_ids=scramble_ids, components=components)
    trackers = write_hyps_to_motmetric_folder(hyp_data_path, motmetrics_path, config, frame_offset=frame_offset, components=components)
    
    iou_class_penalties = calculate_iou_class_penalties(gt, trackers, config, only_consider_equal_classes)
    th_list = np.arange(0.05, 0.99, 0.05)
    acc = compute_motchallenge(motmetrics_path, th_list, iou_class_penalties)

    # Calculate motmetrics results
    mh = mm.metrics.create()
    summary = mh.compute_many(
        acc,
        metrics=metrics,
    )

    # Calculate results
    num_obj = gt.shape[0]
    num_hyp = trackers.shape[0]

    results = {}
    results['num_obj'] = num_obj
    results['num_hyp'] = num_hyp
    results['th_list'] = th_list

    metrics_to_plot = []

    for name, val in zip(summary.columns, np.array(summary).T):
        results[name] = val
        if name in ['hota_alpha', 'recall', 'precision','idf1']:
            metrics_to_plot.append(name)

    if 'num_transfer' in metrics:
        transfer_ratio = (num_hyp-np.array(summary['num_transfer']))/num_hyp
        results['transfer_ratio'] = transfer_ratio
        metrics_to_plot.append('transfer_ratio')

    if 'num_switches' in metrics:
        switch_ratio = (num_obj-np.array(summary['num_switches']))/num_obj
        results['switch_ratio'] = switch_ratio
        metrics_to_plot.append('switch_ratio')

    # Save result text file
    with open(motmetrics_path + '\\' + results_file_name + '.txt', 'w') as f:
        for k, v in results.items():

            f.write(k + ',')
            if isinstance(v, Iterable):
                f.write(','.join(list(v.astype(str))))
            else:
                f.write(str(v))
            f.write('\n')
    
    # Plot and save tracking results
    plot_tracking_results(results, motmetrics_path, title=config['associator'], plot_metrics = metrics_to_plot, results_file_name = results_file_name)

    return acc, summary, th_list, motmetrics_path, trackers, gt



def evaluate_tracker_SintefCam(obj_data_path, hyp_data_path,
                                components = [],
                                ):
    '''
    Evaluate trackers on the TurnSalmon dataset.
    Args:
        obj_data_path: Path to the ground truth data
        hyp_data_path: Path to the tracker data
        components: Components to evaluate. 
    Operations:
        Write the results to the tracker folder.
    '''

    trackers = load_txt_data_for_eval(hyp_data_path)
    gt = load_txt_data_for_eval_SintefCam(obj_data_path)
    gt_trackers_analysed = np.hstack([gt.copy(), np.zeros([gt.shape[0], 3])])

    metrics = {'T': [], 'TO':[], 'BO': [], 'S': [], 'O':[]}

    for g, i in zip(gt, range(gt.shape[0])):
        if g[12] not in components: continue
        m = check_match(g, trackers, 0.2, g[12])
        metrics[g[11]].append(m[2])
        gt_trackers_analysed[i,13:16] = m
    for k in metrics.keys():
        print(k + ':' + ' match ' + str(sum(np.array(metrics[k])=='match')) + ' switch ' + str(sum(np.array(metrics[k])=='switch')) + ' transfer ' + str(sum(np.array(metrics[k])=='transfer')))
    print('\n')
    write_MOT_results(gt_trackers_analysed, '\\'.join(hyp_data_path.split('\\')[:-1]), filename='MOT_results_evaluated.txt')

def check_match(g, comp_trackers, iou_thresh, comp):
        '''
        The grund truth (g) contains two items, g_start and g_end.
        This functions finds the tracker matches to g_start and g_end.
        Subsequently, they assign match, transfer or switch to the ground truth pair.
            - Match if the tracker associated with g_start and g_end is the same
            - Transfer if the tracker associated to g_start exists at the frame of g_end, 
            but is not associated to g_end
            - Switch if neither of the two above occured.
        NOTE: This method works for distinguishing correct tracks (match) from errors (switch and transfers).
        However, the switch and transfer labels are not certain.
        Args:
            g: ground truth label. On the form [ID, frame_num1, x1, y1, x1, y1, frame_num2, x2, y2, x2, y2, TurnLabel, component]
            comp_trackers: Hypothesis trackers on the standard MOT format
            iou_thresh: Minimum IoI overlap between ground truth and trackers
            comp: Component to evaluate
        
        Returns:
            id_match_start: The tracker ID matched to g_start
            id_match_end: The tracker ID matched to g_end
            label: match, transfer or switch
                    
        '''
        g_xyxy1 = ku.xywh2xyxy(g[2:6].astype(float))
        g_xyxy_dict1 = {'x1': g_xyxy1[0], 'y1': g_xyxy1[1], 'x2': g_xyxy1[2], 'y2': g_xyxy1[3]}
        g_xyxy2 = ku.xywh2xyxy(g[7:11].astype(float))
        g_xyxy_dict2 = {'x1': g_xyxy2[0], 'y1': g_xyxy2[1], 'x2': g_xyxy2[2], 'y2': g_xyxy2[3]}

        tracker_cands = comp_trackers[comp_trackers[:,0]==g[1]]
        best_iou = iou_thresh
        id_match_start = -1
        for c in tracker_cands:
            c_xyxy = ku.xywh2xyxy(c[2:6].astype(float))
            c_xyxy_dict = {'x1': c_xyxy[0], 'y1': c_xyxy[1], 'x2': c_xyxy[2], 'y2': c_xyxy[3]}
            iou = ku.get_iou(g_xyxy_dict1, c_xyxy_dict)
            if iou >= best_iou:
                best_iou=iou
                id_match_start = c[1]

        tracker_cands = comp_trackers[comp_trackers[:,0]==g[6]]
        tracker_cands = tracker_cands[tracker_cands[:,10]==comp]
        best_iou = iou_thresh
        id_match_end = -1
        for c in tracker_cands:
            c_xyxy = ku.xywh2xyxy(c[2:6].astype(float))
            c_xyxy_dict = {'x1': c_xyxy[0], 'y1': c_xyxy[1], 'x2': c_xyxy[2], 'y2': c_xyxy[3]}
            iou = ku.get_iou(g_xyxy_dict2, c_xyxy_dict)
            if iou >= best_iou:
                best_iou=iou
                id_match_end = c[1]

        if id_match_end==id_match_start and id_match_end != -1:
            return id_match_start, id_match_end, 'match'
        elif comp_trackers[comp_trackers[:,1]==id_match_start].shape[0] > 0 and comp_trackers[comp_trackers[:,1]==id_match_start][:,0].astype(int).max() >= int(g[6]):
            return id_match_start, id_match_end, 'transfer'
        else:
            return id_match_start, id_match_end, 'switch'