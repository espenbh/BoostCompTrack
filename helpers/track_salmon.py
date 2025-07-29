# Imports
import ultralytics
import cv2
import torchvision.transforms as transforms
import imutils
import numpy as np

# Code paths
import os
import sys
sys.path.append(os.path.split(os.getcwd())[0] + '\\salmon_component_tracking\\helpers')
sys.path.append(os.path.split(os.getcwd())[0] + '\\salmon_component_tracking\\associator\\CompTrack\\')
sys.path.append(os.path.split(os.getcwd())[0] + '\\salmon_component_tracking\\associator\\BoostTrack\\')
sys.path.append(os.path.split(os.getcwd())[0] + '\\salmon_component_tracking\\associator\\BoostTrack\\external')

import keybox_utils as ku
from file_utils import create_folders_for_salmon_tracking, write_MOT_results
from CompTrack import CompTrack, YOLO2CompTrack
from tracker.boost_track import BoostTrack
from draw_utils import annotate_frame
from default_settings import GeneralSettings, BoostTrackSettings, BoostTrackPlusPlusSettings
from tracker.boost_track import SalmonTracker, KalmanBoxTracker
from comp_utils import get_comp_id_from_salmon_ID_and_comp_type


def track_salmon(config_file_path):
    '''
    Track salmon.

    Args:
        config_file_path (str): Path to the config file
    Operations:
        Generate a folder as specified by the config file, where all tracking information is stored
        Generates an annotated video displaying the tracking behaviour. Saves this video to disk.
        Generates images displaying the tracking behaviour. Saves this video to disk.
        Generates a .txt file on the MOT format with tracking information.
    '''
    # Create analysis folder
    config, analysis_path = create_folders_for_salmon_tracking(config_file_path)

    # Specify names of input and output video
    input_video_name = config['video_path'].split('\\')[-1].split('.')[0]
    output_video_name = analysis_path + '\\' + input_video_name + '.mp4'

    # Set up video stream
    cap = cv2.VideoCapture(config['video_path'])

    # Specify start frame
    frame_num = config['start_frame']
    cap.set(cv2.CAP_PROP_POS_FRAMES, config['start_frame'])

    # Set up detection model
    if config['detector'] == 'keybox_detection':
        detection_model = ultralytics.YOLO(config['detection_model_path'], task = 'pose')
    else:
        detection_model = ultralytics.YOLO(config['detection_model_path'])

    tracker = None

    # Set up tracker
    if config['associator'] == 'CompTrack':
        consider_salmon_composition = True
        tracker = CompTrack(config)
        assert config['use_embedding'] == False, 'Embedding is not implemented for CompTrack'
        assert config['detector'] == 'keybox_detection', 'CompTrack can only be used with keybox detectors'
    elif config['associator'] == 'BoostTrack' or config['associator'] == 'BoostCompTrack':
        if config['associator'] == 'BoostTrack':
            consider_salmon_composition = False
        elif config['associator'] == 'BoostCompTrack':
            consider_salmon_composition = True
            assert config['detector'] == 'keybox_detection', 'BoostCompTrack can only be used with keybox detectors'
        generalsettings = GeneralSettings()
        boosttracksettings = BoostTrackSettings()
        boosttrackplusplussettings = BoostTrackPlusPlusSettings()

        generalsettings.values['consider_salmon_composition'] = consider_salmon_composition
        generalsettings.values['eps'] = config['eps']
        generalsettings.values['ncomp'] = len(config['components'])
        generalsettings.values['dataset'] = 'mot20'
        generalsettings.values['use_embedding'] = config['use_embedding']
        generalsettings.values['use_ecc'] = config['use_ecc']
        generalsettings.values['test_dataset'] = False
        generalsettings.values['det_thresh'] = config['salmon_conf']
        generalsettings.values['max_age'] = config['max_hidden_length']
        generalsettings.values['iou_threshold'] = config['trk_det_min_iou_threshold']
        generalsettings.values['debug'] = config['debug']
        generalsettings.values['min_hits'] = config['min_hits']

        generalsettings.values['activate_turn'] = config['activate_turn']
        generalsettings.values['activate_salmon_error_bp_disagreement_check'] = config['activate_salmon_error_bp_disagreement_check']
        generalsettings.values['activate_salmon_error_no_bp_overlap_check'] = config['activate_salmon_error_no_bp_overlap_check']
        generalsettings.values['activate_bp_error_bp_disagreement_check'] = config['activate_bp_error_bp_disagreement_check']

        boosttracksettings.values['s_sim_corr'] = True
        boosttracksettings.values['use_dlo_boost'] = True
        boosttracksettings.values['use_duo_boost'] = True


        boosttrackplusplussettings.values['use_rich_s'] = True
        boosttrackplusplussettings.values['use_sb'] = True
        boosttrackplusplussettings.values['use_vt'] = True

        tracker = BoostTrack(video_name=input_video_name,
                             generalsettings = generalsettings,
                             boosttracksettings = boosttracksettings,
                             boosttrackplusplussettings = boosttrackplusplussettings)

    # Specify writer of output video
    writer = cv2.VideoWriter(output_video_name,  
                        cv2.VideoWriter_fourcc(*'XVID'), 
                        10, (config['video_shape'][0],config['video_shape'][1])) 
    
    # Initialize results list
    results = []
    debugs = []

    # Transform for BoostTrack
    transform = transforms.ToTensor()
    
    # Iterate over frames
    while cap.isOpened():
        print(frame_num)
        tag = f"{input_video_name}:{frame_num}"
        
        success, frame = cap.read()
        
        if not success:
            print('Cannot read frame')
            break
        
        # Rotate frame
        if config['rotate'] != 0:
            frame = imutils.rotate_bound(frame, config['rotate']).astype(np.uint8)
    
        # Retrieve detections
        detections = YOLO2CompTrack(detection_model(frame, iou=config['salmon_iou'], conf = config['salmon_conf'], agnostic_nms = False), config) # Shape Nsal X Nbp X 5
        if detections.shape[0] == 0:
            detections = np.zeros([1,len(config['components']), 5])

        if tracker != None:
            # Update tracker with detections
            if 'Boost' in config['associator']:
                targets, debug = tracker.update(detections, transform(frame).unsqueeze(0), frame, tag, frame_num)
                debugs.extend(debug)
            else:
                targets = tracker.update(detections, transform(frame).unsqueeze(0), frame, tag, frame_num)
                targets = [t + [-1] for t in targets]

            # Update results list
            for t in targets:
                bblbbtwh = ku.xyxy2bblbbtwh(t[0:4])
                results.append([str(frame_num), str(t[4])] + [str(l) for l in bblbbtwh] + [str(t[5])] + ['-1', '-1', '-1'] + [config['components'][int(t[6])] + ',' + str(t[7])])
  
            # Annotate frame
            annotated_frame = annotate_frame(targets, frame.copy(), frame_num, config, consider_salmon_composition, font_scale = 4)
        else:
            annotated_frame = frame.copy()


        # Store some frames to disk
        if frame_num-config['start_frame'] in config['saved_frames']:
            cv2.imwrite(analysis_path + '\\' + 'frame' + str(frame_num) + '.jpg', annotated_frame)

        # Write annotated frame to video
        aframe = cv2.resize(annotated_frame, [config['video_shape'][0], config['video_shape'][1]])
        print(aframe.shape)
        writer.write(aframe)

        # Check termination criterion
        frame_num = frame_num + 1
        if frame_num > config['end_frame']:
            break
        
    # Release video operations
    cap.release()
    writer.release()
    SalmonTracker.count = 1
    KalmanBoxTracker.count = 0
    print(KalmanBoxTracker)
    
    # Save results to disk
    write_MOT_results(results, analysis_path)
    write_MOT_results(debugs, analysis_path, filename='debug.txt')
    return analysis_path
        

def not_track_salmon(config_file_path):
    '''
    Provide the same output as when tracking salmon, but generate a new ID for each salmon each frame.

    Args:
        config_file_path (str): Path to the config file
    Operations:
        Generate a folder as specified by the config file, where all tracking information is stored
        Generates an annotated video displaying the tracking behaviour. Saves this video to disk.
        Generates images displaying the tracking behaviour. Saves this video to disk.
        Generates a .txt file on the MOT format with tracking information.
    '''
    # Create analysis folder
    config, analysis_path = create_folders_for_salmon_tracking(config_file_path)

    # Specify names of input and output video
    input_video_name = config['video_path'].split('\\')[-1].split('.')[0]
    output_video_name = analysis_path + '\\' + input_video_name + '.avi'

    # Set up video stream
    cap = cv2.VideoCapture(config['video_path'])

    # Specify start frame
    frame_num = config['start_frame']
    cap.set(cv2.CAP_PROP_POS_FRAMES, config['start_frame'])

    # Set up detection model
    if config['detector'] == 'keybox_detection':
        consider_salmon_composition = True
        detection_model = ultralytics.YOLO(config['detection_model_path'], task = 'pose')
    else:
        consider_salmon_composition = False
        detection_model = ultralytics.YOLO(config['detection_model_path'])

    # Specify writer of output video
    writer = cv2.VideoWriter(output_video_name,  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        10, (config['video_shape'][0],config['video_shape'][1])) 
    
    # Initialize results list
    results = []

    sal_id = 0

    # Iterate over frames
    while cap.isOpened():
        tag = f"{input_video_name}:{frame_num}"
        
        success, frame = cap.read()
        
        if not success:
            print('Cannot read frame')
            break
        
        # Rotate frame
        if config['rotate'] != 0:
            frame = imutils.rotate_bound(frame, config['rotate']).astype(np.uint8)
    
        # Retrieve detections
        detections = YOLO2CompTrack(detection_model(frame, iou=config['salmon_iou'], conf = config['salmon_conf'], agnostic_nms = False), config) # Shape Nsal X Nbp X 5

        targets = []
        # Update results list
        for sal in detections:
            for comp, i in zip(sal, range(sal.shape[0])):
                bblbbtwh = ku.xyxy2bblbbtwh(comp[0:4])
                comp_type = config['components'][int(i)]
                comp_id = get_comp_id_from_salmon_ID_and_comp_type(sal_id, i)
                conf = str(comp[4])
                if (float(conf) >= float(config['bp_conf']) and i != 0) or (float(conf) >= float(config['salmon_conf']) and i == 0):
                    results.append([str(frame_num), str(comp_id)] + [str(l) for l in bblbbtwh] + [conf] + ['-1', '-1', '-1'] + [comp_type])
                    targets.append(list(comp[0:4].astype(int)) + [str(comp_id)] + [str(conf)] + [i])
            sal_id = sal_id + 1

        # Annotate frame
        annotated_frame = annotate_frame(targets, frame.copy(), frame_num, config, consider_salmon_composition, font_scale=1, font_thickness=1, skeleton_line_thickness=1)

        # Store some frames to disk
        if frame_num-config['start_frame'] in config['saved_frames']:
            cv2.imwrite(analysis_path + '\\' + 'frame' + str(frame_num) + '.jpg', annotated_frame)

        # Write annotated frame to video
        writer.write(annotated_frame)

        # Check termination criterion
        frame_num = frame_num + 1
        #if frame_num > config['start_frame'] + 3:
        #    break
        if frame_num > config['end_frame']:
            break

    # Release video operations
    cap.release()
    writer.release()
    SalmonTracker.count = 1
    KalmanBoxTracker.count = 0
    
    # Save results to disk
    write_MOT_results(results, analysis_path)
    return analysis_path
        
