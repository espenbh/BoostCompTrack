import os
import yaml
import shutil
import json
import os
import sys
sys.path.append(os.path.split(os.getcwd())[0] + '\\salmon_component_tracking\\helpers')
import keybox_utils as ku
import numpy as np
from comp_utils import get_comp_id_from_salmon_ID_and_comp_type
import cv2
import ultralytics
import imutils
sys.path.append(os.path.split(os.getcwd())[0] + '\\salmon_component_tracking\\associator\\CompTrack\\')
sys.path.append(os.path.split(os.getcwd())[0] + '\\associator\\CompTrack\\')
from CompTrack import YOLO2CompTrack



def create_folders_for_salmon_tracking(config_file_path):
    '''
    Create the folder structure for a new tracking run.
    
    Args:
        config_file_path (str): The path to the input config file
    Operations:
        Creates an analysis folder at the specified location
        Copies the configuration file into the new analysis folder
    Return:
        The loaded config file
        The path to the created analysis folder
    '''

    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    salmon_tracking_root = '\\'.join(config_file_path.split('\\')[:-3])

    if not os.path.exists(salmon_tracking_root + r'\\associator'):
        os.mkdir(salmon_tracking_root + r'\\associator')
    if not os.path.exists(salmon_tracking_root + r'\\associator\\' + config['associator']):
        os.mkdir(salmon_tracking_root + r'\\associator\\' + config['associator'])

    associator_track_root = salmon_tracking_root + r'\\associator\\' + config['associator']
    if not os.path.exists(associator_track_root + r'\\output'):
        os.makedirs(associator_track_root + r'\\output')
    if not os.path.exists(associator_track_root + r'\\output\\' + config['project']):
        os.makedirs(associator_track_root + r'\\output\\' + config['project'])

    if len(os.listdir(associator_track_root + r'\\output\\' + config['project'])) == 0:
        os.makedirs(associator_track_root + r'\\output\\' + config['project'] + '\\analysis1')
        analysis_path = associator_track_root + r'\\output\\' + config['project'] + '\\analysis1'
    else:
        max_analysis = max(int(f.split('ysis')[1]) for f in os.listdir(associator_track_root + r'\\output\\' + config['project']) if f.startswith('analysis'))
        os.makedirs(associator_track_root + r'\\output\\' + config['project'] + '\\analysis' + str(max_analysis+1))
        analysis_path = associator_track_root + r'\\output\\' + config['project'] + '\\analysis' + str(max_analysis+1)

    shutil.copy(config_file_path, analysis_path + '\\config.yml')

    return config, analysis_path

def create_folders_for_detector_optimization(config_file_path):
    '''
    Create the folder structure for a new tracking run.
    
    Args:
        config_file_path (str): The path to the input config file
    Operations:
        Creates an analysis folder at the specified location
        Copies the configuration file into the new analysis folder
    Return:
        The loaded config file
        The path to the created analysis folder
    '''

    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    salmon_tracking_root = '\\'.join(config_file_path.split('\\')[:-3])

    if not os.path.exists(salmon_tracking_root + r'\\detector_optimization'):
        os.mkdir(salmon_tracking_root + r'\\detector_optimization')
    if not os.path.exists(salmon_tracking_root + '\\detector_optimization\\' + config['detector_name']):
        os.mkdir(salmon_tracking_root + '\\detector_optimization\\' + config['detector_name'])

    detector_root = salmon_tracking_root + '\\detector_optimization\\' + config['detector_name']

    if len(os.listdir(detector_root)) == 0:
        os.makedirs(detector_root + '\\analysis1')
        analysis_path = detector_root + '\\analysis1'
    else:
        max_analysis = max(int(f.split('ysis')[1]) for f in os.listdir(detector_root) if f.startswith('analysis'))
        os.makedirs(detector_root + '\\analysis' + str(max_analysis+1))
        analysis_path = detector_root + '\\analysis' + str(max_analysis+1)

    shutil.copy(config_file_path, analysis_path + '\\config.yml')

    return config, analysis_path

def write_MOT_results(results, analysis_path, filename = 'MOT_results.txt'):
    ''' 
    Write a text file to disk that follows the MOT format.
    The MOT format is: [frame, ID, min x value, min y value, w, h, -1, -1, -1, class]
    Example: [10, 29, 2000, 3000, 120, 45, conf, -1, -1, -1, head]

    Args:
        results (list[list[str]]). A list of MOT-formatted objects. All values should be strings
        analysis_path (str): The path to the location where the MOT results should be saved.
    Operations:
        Write a .txt mot file to disk

    '''
    with open(analysis_path + '\\' + filename, 'w') as f:
        for res in results:
            f.write(','.join(res))
            f.write('\n')


def load_labelme_data_for_eval(labelme_root, config, frame_offset = 14100, splitter = '_f', scramble_ids = False):
    '''
    Load data on the labelme format into an array in the MOT format.

    Args:
        labelme_root (str): The path to the root of the labelme data
        config (dict): Tracker config file
        frame_offset: Offset between video frame number and label frame number
        splitter (str): String between video name and frame number in the file names of the images and labels
        scramble_ids (bool): If True, overwrite labled IDs such that all salmon have differen IDs
    Returns:
        A numpy array on the format [frame, ID, center x value, center y value, w, h, -1, -1, -1, -1, class]
    '''
    return_array = []
    group_id_and_frame_to_id = []

    # Iterate over all labels
    for label_name in os.listdir(labelme_root + '\\labels\\'):

        # Get frame number
        frame = int(label_name.split(splitter)[1].split('.json')[0]) - frame_offset

        # Load label
        with open(labelme_root + '\\labels\\' + label_name, 'r') as f:
            label = json.load(f)

            # Iterate over all label objects
            for item in label['shapes']:

                # Randomize ID if scramble_ID
                if scramble_ids:
                    k = str(item['group_id']) + '_' + str(frame)
                    if k not in group_id_and_frame_to_id:
                        group_id_and_frame_to_id.append(k)
                    id = group_id_and_frame_to_id.index(k)+1
                else:
                    id = int(item['group_id'])

                if item['shape_type'] == 'rectangle':
                #if np.array(item['points']).flatten().shape[0] == 4:
                    # Get object coordinates on right format
                    xyxy = list(np.array(item['points']).flatten())
                    xywh = ku.xyxy2xywh(xyxy)
                elif item['shape_type'] == 'point' and 'eye' not in config['components']:
                    xywh = list(np.array(item['points']).flatten()) + [-1,-1]
                elif item['shape_type'] == 'line' and 'eye' in config['components']:
                    xyxy = list(np.array(item['points']).flatten())
                    xywh1 = xyxy[:2] + [-1,-1]
                    xywh2 = xyxy[2:] + [-1,-1]
                
                    if item['label'] == 'line_ujaw':
                        return_array.append([frame] + [get_comp_id_from_salmon_ID_and_comp_type(id, config['components'].index('eye'), num_comp = len(config['components']))] + xywh1 + ['1', '-1', '-1', '-1'] + ['eye'])
                        return_array.append([frame] + [get_comp_id_from_salmon_ID_and_comp_type(id, config['components'].index('ujaw'), num_comp = len(config['components']))] + xywh2 + ['1', '-1', '-1', '-1'] + ['ujaw'])
                    if item['label'] == 'line_ljaw':
                        return_array.append([frame] + [get_comp_id_from_salmon_ID_and_comp_type(id, config['components'].index('ljaw'), num_comp = len(config['components']))] + xywh2 + ['1', '-1', '-1', '-1'] + ['ljaw'])
                    continue

                # Append object to return array
                if item['label'] in config['components']:
                    return_array.append([frame] + [get_comp_id_from_salmon_ID_and_comp_type(id, config['components'].index(item['label']), num_comp = len(config['components']))] + xywh + ['1', '-1', '-1', '-1'] + [item['label']])
    return np.array(return_array)


def load_txt_data_for_eval(pred_data_path):
    '''
    Load results stored in .txt file

    Args:
        pred_data_path (str): The path to the text data
    Returns:
        A numpy array on the format [frame, ID, center x value, center y value, w, h, conf, -1, -1, -1, class]
    '''
    return_array = []
    with open(pred_data_path, mode ='r') as file:
        for line in file:
            line = line.rstrip().split(',')
            line[2:6] = ku.bblbbtwh2xywh(line[2:6])
            return_array.append(line)
    return np.array(return_array)

def load_txt_data_for_eval_SintefCam(pred_data_path):
    return_array = []
    with open(pred_data_path, mode ='r') as file:
        for line in file:
            line = line.rstrip().split(',')
            line[2:6] = ku.bblbbtwh2xywh(line[2:6])
            line[7:11] = ku.bblbbtwh2xywh(line[7:11])
            return_array.append(line)
    return np.array(return_array)


def create_salmon_tracking_config_file( path: str, 
                                        video_path: str,
                                        detection_model_path: str, 
                                        associator: str,
                                        detector: str,
                                        use_embedding = False,
                                        project = r'salmon_tracking',

                                        start_frame = 0,
                                        end_frame = 99,
                                        video_shape = [4242,4242],
                                        rotate = 135,            
                                                        
                                        salmon_conf = 0.2,
                                        salmon_iou = 0.7,
                                        bp_conf = 0.5,
                                        bp_loc = 5,

                                        max_hidden_length = 0,
                                        min_hits = 0,
                                        trk_det_min_iou_threshold = 0.1,
                                        debug = False,
                                        CompTrack_max_cost = 10,

                                        saved_frames = [0, 10, 20, 30, 40, 50],

                                        skeleton = [['head', 'dorsal_fin'], ['head', 'body'], ['head', 'pec_fin'], ['body', 'dorsal_fin'], ['dorsal_fin', 'adi_fin'], ['adi_fin', 'anal_fin'], ['adi_fin', 'tail_fin'], ['tail_fin', 'body'], ['tail_fin', 'anal_fin'], ['anal_fin', 'pelv_fin'], ['pelv_fin', 'body'], ['pelv_fin', 'pec_fin']],
                                        component_colors = {'salmon': (200,100,50),'body': (0,255,0),'head': (255,0,0),'tail_fin': (255,0,255),'dorsal_fin': (0,0,255),'anal_fin': (0,255,255),'adi_fin': (255,255,0),'pelv_fin': (120,120,120),'pec_fin': (50, 100, 200)},
                                        salmon_bbox_thickness = 6,
                                        bp_bbox_thickness = 4,

                                        bodyparts =            ['head', 'dorsal_fin', 'adi_fin', 'tail_fin', 'anal_fin','pelv_fin','pec_fin', 'body'],
                                        components = ['salmon', 'head', 'dorsal_fin', 'adi_fin', 'tail_fin', 'anal_fin','pelv_fin','pec_fin', 'body'],

                                        eps = 0.01,

                                        detector_name = 'generic_detector',
                                        additional_kps = [],

                                        use_ecc = True,
                                        activate_turn = True,
                                        activate_salmon_error_bp_disagreement_check = True,
                                        activate_salmon_error_no_bp_overlap_check = True,
                                        activate_bp_error_bp_disagreement_check = True,
                                        ):
    """
    This config file specifies parameters for salmon tracking.
    """

    config = {
        r'path': path,                  # Path to salmon tracking root folder
        r'video_path': video_path,      # Path to the video that will be subjected to the salmon tracking algorithm
        r'detection_model_path': detection_model_path,    # Path to the weights of the YOLO detection model
        r'associator': associator,      # Associator name. Any of "CompTrack", "BoostTrack" and "BoostCompTrack"
        r'detector': detector,          # Detector name. Any of "bounding_box_detection" and "keybox_detection"
        r'use_embedding': use_embedding,# Whether to use the mot20 person re-id appearence embedding

        r'project': project,            # Project name

        r'start_frame': start_frame,    # Video start frame
        r'end_frame': end_frame,        # Video end frame
        r'video_shape': video_shape,    # Shape of the input video [h, w]
        r'rotate': rotate,              # Rotation of the input video frames

        r'salmon_conf': salmon_conf,    # YOLO confidence threshold for object detection
        r'salmon_iou': salmon_iou,      # YOLO IoU threshold for object detection
        r'bp_conf': bp_conf,            # YOLO confidence threshold for bodypart detection
        r'bp_loc': bp_loc,              # Location threshold for bodypart detection

        r'max_hidden_length': max_hidden_length,    # Number of frames a salmon can be undetected before it is removed
        r'min_hits': min_hits,                      # Number of consecutive detection observations before a tracker is considered spawned
        r'trk_det_min_iou_threshold': trk_det_min_iou_threshold,    # If the tracker and detection overlap is below this value, the match is discarded. Implemented for BoostTrack and BoostCompTrack 
        r'debug': debug,
        r'CompTrack_max_cost': CompTrack_max_cost,  # Maximum cost allowed for matching detection and tracker

        r'bodyparts': bodyparts,    # List of salmon body parts
        r'additional_kps': additional_kps,    # List of additional kps

        r'component_colors': component_colors,    # Dictionary matching salmon components and plot colors

        r'skeleton': skeleton,  # Define salmon skeleton for drawing

        r'saved_frames': saved_frames,  # Results and image annotations will only be saved for these frames. Frame number relative to start_frame.

        r'eps': eps,    # A small value

        # Drawing parameters
        r'salmon_bbox_thickness':  salmon_bbox_thickness,
        r'bp_bbox_thickness': bp_bbox_thickness,

       # List of salmon components
        r'components': components,

        # Detector name. Used in detector hyperparameter optimization
        r'detector_name': detector_name,
        
        r'use_ecc': use_ecc, # Activate camera compensation

        # Activate BoostCompTrack modules
        r'activate_turn': activate_turn,
        r'activate_salmon_error_bp_disagreement_check': activate_salmon_error_bp_disagreement_check,
        r'activate_salmon_error_no_bp_overlap_check': activate_salmon_error_no_bp_overlap_check,
        r'activate_bp_error_bp_disagreement_check': activate_bp_error_bp_disagreement_check,
    }
    
    with open(path + r'\\' + 'config.yml', 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

def remove_and_create_validation_folder_structure(pred_data_root):
    '''
    Create a subfolder in the tracking folder where all validation data are saved.
    
    Args:
        pred_data_root (str): The path to the tracking analysis folder.
    Operations:
        Create a fodler structure that can be used for saving validation data
    '''
    if 'validation' in os.listdir(pred_data_root):
        shutil.rmtree(pred_data_root +'\\validation\\')
    os.mkdir(pred_data_root +'\\validation\\')
    os.mkdir(pred_data_root +'\\validation\\switch')
    os.mkdir(pred_data_root +'\\validation\\loss')
    os.mkdir(pred_data_root +'\\validation\\all')


def create_detector_MOT_results(config_file_path):
    '''
    Write associator-free results to the analysis folder. Each detection is given a new ID.

    Args:
        config_file_path (str): Path to the config file
    Operations:
        Generate a folder as specified by the config file, where all tracking information is stored
        Generates a .txt file on the MOT format with tracking information.
        Moves the config file to the created analysis folder
    '''
    # Initialize ID counter
    id_cnt = 1

    # Create analysis folder
    config, analysis_path = create_folders_for_detector_optimization(config_file_path)

    # Set up video stream
    cap = cv2.VideoCapture(config['video_path'])

    # Set up detection model
    if config['detector'] == 'keybox_detection':
        detection_model = ultralytics.YOLO(config['detection_model_path'], task = 'pose')
    else:
        detection_model = ultralytics.YOLO(config['detection_model_path'])

    # Initialize results list
    results = []

    for saved_frame in config['saved_frames']:
        # Load frame
        frame_num = saved_frame #+ config['start_frame']
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()
        
        if not success:
            print('Cannot read frame')
            break
        
        # Rotate frame
        if config['rotate'] != 0:
            frame = imutils.rotate_bound(frame, config['rotate']).astype(np.uint8)

        # Retrieve detections
        detections = YOLO2CompTrack(detection_model(frame, iou=config['salmon_iou'], conf = config['salmon_conf'], agnostic_nms = False), config) # Shape Nsal X Nbp X 5
        plot_targets = []

        # Update results list
        for salmon, id in zip(detections, range(detections.shape[0])):
            for comp, comp_type in zip(salmon, range(salmon.shape[0])):
                bblbbtwh = ku.xyxy2bblbbtwh(comp[0:4])
                conf = comp[4]
                comp_id = get_comp_id_from_salmon_ID_and_comp_type(id_cnt + id, comp_type)
                if conf > config['eps']:
                    results.append([str(int(frame_num)), str(int(comp_id))] + [str(l) for l in bblbbtwh] + [str(conf)] + ['-1', '-1', '-1'] + [config['components'][int(comp_type)]])
                    plot_targets.append([comp[0], comp[1], comp[2], comp[3], comp_id, conf, comp_type])
        id_cnt = id_cnt + detections.shape[0]

    # Release video operation
    cap.release()

    # Save results to disk
    write_MOT_results(results, analysis_path)

    return analysis_path

def read_config(config_path):
    '''
    Read configuration file

    Args:
        config_path (str): Path to the configuration file
    Returns:
        config (dict): Tracker configuration file
    '''
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def read_results(results_path):
    '''
    Read tracker results file

    Args:
        results_path (str): Path to the results file
    Returns:
        results (dict): Tracker results
    '''
    
    results = {}
    try:
        with open(results_path, 'r') as file:
            for line in file:
                k = line.rstrip().split(',')[0]
                v = line.rstrip().split(',')[1:]
                results[k] = [float(i) for i in v]
    except:
        print('Could not read ' + results_path)
    return results

def get_frame(frame_num, config):
    '''
    Get image from video

    Args:
        frame_num (int): Frame number to be extracted from the video
        config (dict): Tracker configuration file
    Returns:
        frame (numpy array): Image intensity array
    '''
    
    cap = cv2.VideoCapture(config['video_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    success, frame = cap.read()
    if success:
        frame = imutils.rotate_bound(frame, 135)
        return frame
    else:
        print('No frame found')
        return None