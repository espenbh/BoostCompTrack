import numpy as np

import os
import sys
sys.path.append(os.path.split(os.getcwd())[0] + '\\helpers')

import keybox_utils as ku
from comp_utils import get_comp_id_from_salmon_ID_and_comp_type


def assign_salmon_matches(cost_matrix, max_cost):
    '''
    A function that matches trackers and salmon detections. It uses a greedy strategy, where the least cost assignments are accepted first.

    Args:
        cost_matrix (np.array): A matrix of shape salmon_detections x salmon_trackers that states the cost of matching these two items.
        max_cost (float): Maximum acceptable cost of a match.
    
    Returns:
        1. A list of detection indices
        2. A list of tracker indices

        det_idces[i] is a match with trk_idces[i]
    '''

    det_idces = []
    trk_idces = []
    max_matches = min(cost_matrix.shape[0], cost_matrix.shape[1])
    if max_matches > 0:
        possible_matches = np.transpose(np.unravel_index(cost_matrix.flatten().argsort(), cost_matrix.shape))
        for m in possible_matches:
            if cost_matrix[m[0], m[1]] > max_cost:
                break
            if cost_matrix[m[0], m[1]] == np.nan:
                continue
            if m[0] not in det_idces and m[1] not in trk_idces:
                det_idces.append(m[0])
                trk_idces.append(m[1])
    return det_idces, trk_idces

class Bodypart():
    """
    A class for handling salmon body parts.

    Attributes:
        type (str): One of ['head', 'dorsal_fin', 'adi_fin', 'tail_fin', 'anal_fin', 'pelv_fin', 'pec_fin', 'body'].
        location_history (dict{int:list[float]}): A dict where the keys are the frame index, and the value is the location and confidence of the body part at that frame (xyxyc).
        last_observation (int): The frame number of the last time the bodypart was observed.
    """

    def __init__(self, type):
        """Initialize a bodypart object"""
        self.type = type
        self.location_history = {}
        self.last_observation = -1
           
    def update(self, xyxyc, frame_num):
        '''Update a bodypart object when it is detected in a new frame'''
        self.location_history[frame_num] = xyxyc
        self.last_observation = frame_num

class Salmon():
    """
    A class for handling salmon.

    Attributes:
        config (dict): The configuration information specifying the tracking hyperparameters.
        location_history (dict{int:list[float]}): A dict where the keys are the frame index, and the value is the location of the salmon at that frame (xyxyc).
        bodyparts: (dict{str:Bodypart}): A dictionary where the key is a body part string, and the value is a body part object.
        last_observation (int): The frame number of the last time the salmon was observed.
        id (int): The ID of the salmon
    """
    def __init__(self, id, config):
        """Initialize a salmon object"""
        self.config = config
        self.location_history = {}
        self.bodyparts = {k:Bodypart(k) for k in self.config['bodyparts']}
        self.last_observation = -1
        self.id = id

    def update(self, xyxycs, frame_num):
        '''Update a salmon object when it is detected in a new frame'''
        self.location_history[frame_num] = xyxycs[0,:]
        self.last_observation = frame_num
        for xyxyc, bp in zip(xyxycs[1:,:], self.config['bodyparts']):
            # Only update body part if it is visible
            if xyxyc[4] > self.config['eps']:
                self.bodyparts[bp].update(list(xyxyc), frame_num)

class CompTrack():
    '''
    A class for tracking salmon in a video.

    Attributes:
        config (dict): Information from the config file.
        salmons (list[Salmon]): A list of all salmon in the camera field of view.
        old_salmons (list[Salmon]): A list of all salmon not in the camera field of view
        num_salmon (int): The number of previously observed salmon.
    '''

    def __init__(self, config):
        """Initialize a VideoMonitor object"""
        self.config = config
        
        self.salmons = []
        self.old_salmons = []

        # Start at 1 to ensure ID > 0
        self.num_salmon = 1

    def calculate_assignment_cost(self, trk, xyxycs):
        ''' 
        A function to calculate the location similarity between two salmon.

        args:
            salmon_trk (Salmon): A Salmon object
            detection (list[list[float]]): xywhc for all body parts. Shape 8x5.
        '''
        
        # find ious for all bodyparts
        ious = []
        for bp in list(trk.bodyparts.keys()):
            if trk.bodyparts[bp].last_observation == -1:
                continue
            trk_bp_xyxy = trk.bodyparts[bp].location_history[trk.bodyparts[bp].last_observation][:4]
            detection_bp_xyxy = xyxycs[self.config['bodyparts'].index(bp)][:4]

            if xyxycs[self.config['bodyparts'].index(bp)][4] > self.config['eps']:
                salmon_trk_bbox = {'x1': trk_bp_xyxy[0], 'x2': trk_bp_xyxy[2], 'y1': trk_bp_xyxy[1], 'y2': trk_bp_xyxy[3]}
                salmon_detection_bbox = {'x1': detection_bp_xyxy[0], 'x2': detection_bp_xyxy[2], 'y1': detection_bp_xyxy[1], 'y2': detection_bp_xyxy[3]}
                iou = ku.get_iou(salmon_trk_bbox, salmon_detection_bbox)
                ious.append(iou)
                
        
        # Average iou cost
        iou_cost = 1000
        if len(ious) > 0:
            iou_rev = sum(ious)/len(ious)
            if iou_rev > self.config['eps']:
                iou_cost = 1/(iou_rev)

        # Penalize incomplete salmon
        iou_cost = iou_cost - (len(self.config['bodyparts'])-len(ious))

        return iou_cost
        
        
    def update(self, salmon_dets, frame_tensor, frame, tag, frame_num):
        '''
        Function to update salmons in a new frame

        Args:
            salmon_dets (np.array): An array of shape Nsal x Nbp x 5. Each Nsal specify a salmon. Each Nbp specify a observation in a salmon.
            Each observation is specified by [x, y, x, y, conf]. If a body part is occluded, all values in the observation is set to 0 ([0, 0, 0, 0, 0]).
            frame_num (int): The frame count
            frame_tensor, frame, tag: Dummy variables that is used for other trackers
        Return:
            A list of all components observed in the current frame. Each objects is described by a 7-tuple ( [x, y, x, y, id, conf, class] )
        Operations:
            Updates the salmon in the self.salmons list
        '''
        processed_det_idces = []

        # Cost matrix of shape (dets x trks)
        cost_matrix = np.ones([salmon_dets.shape[0], len(self.salmons)])*1000

        # Fill cost matrix
        # Iterate over all salmon trackers
        for trk, trk_idx in zip(self.salmons, range(len(self.salmons))):
            xyxyc_trk = trk.location_history[trk.last_observation]
            # Iterate over all salmon detections
            for xyxycs, det_idx in zip(salmon_dets, range(salmon_dets.shape[0])):
                # Check if a salmon detection is inside a salmon tracker
                if ku.point_inside_box(ku.xyxy2xywh(xyxycs[0,:4])[0:2], ku.xyxy2xywh(xyxyc_trk[:4])):
                    # Update cost matrix
                    cost_matrix[det_idx, trk_idx] = self.calculate_assignment_cost(trk, xyxycs[1:,:])
            

        # Find salmon matches, and update salmon with matches
        det_assignment_idces, trk_assignment_idces = assign_salmon_matches(cost_matrix, self.config['CompTrack_max_cost'])
        for trk_assignment_idx, det_assignment_idx in zip(trk_assignment_idces, det_assignment_idces):
            if cost_matrix[det_assignment_idx, trk_assignment_idx] < self.config['CompTrack_max_cost']:
                self.salmons[trk_assignment_idx].update(salmon_dets[det_assignment_idx,:,:], frame_num)
                processed_det_idces.append(det_assignment_idx)

        # Create new salmon for the unmatched predictions
        unprocessed_det_idces = [i for i in range(salmon_dets.shape[0]) if i not in processed_det_idces]
        for det_idx in unprocessed_det_idces:
            s = Salmon(self.num_salmon, self.config)
            self.num_salmon = self.num_salmon + 1
            s.update(salmon_dets[det_idx,:,:], frame_num)
            self.salmons.append(s)

        # Remove salmon if they have been hidden for the specified time
        for trk, trk_idx in zip(reversed(self.salmons), reversed(range(len(self.salmons)))):
            if (frame_num - trk.last_observation) > self.config['max_hidden_length']:
                self.old_salmons.append(self.salmons[trk_idx])
                del self.salmons[trk_idx]

        # Return trackers
        res = []
        for trk in self.salmons:
            if trk.last_observation == frame_num:
                res.append(list(trk.location_history[frame_num])[0:4] + [get_comp_id_from_salmon_ID_and_comp_type(trk.id, 0)] + [list(trk.location_history[frame_num])[4]] + [0]) # [x, y, x, y, id, conf, class]
                for bp_str, bp_int in zip(self.config['bodyparts'], range(len(self.config['bodyparts']))):
                    if trk.bodyparts[bp_str].last_observation == frame_num:
                        res.append(list(trk.bodyparts[bp_str].location_history[frame_num])[0:4] + [get_comp_id_from_salmon_ID_and_comp_type(trk.id, bp_int+1)] + [list(trk.bodyparts[bp_str].location_history[frame_num])[4]] + [bp_int+1])
        return res

def YOLO2CompTrack(YOLO_detections, config):
    '''
    Args:
        YOLO_detections: Output of a YOLO model

    Return:
        - dets (np.ndarray): An array of shape Nsal x Nbp x 5. Each salmon (Nsal) has a 5-tuple (x, y, x, y, conf) for each body part (Nbp).
        If the YOLO detection is from a keybox_detection model, the return tensor will be structured according to salmon individuals.
        Else, the detections will be structured randomly.
        - config (dict): Config information for salmon tracking
    '''
    
    if config['detector'] == 'keybox_detection':
        xyo = YOLO_detections[0].keypoints.data.detach().cpu().numpy()
        xywh = YOLO_detections[0].boxes.xywh.detach().cpu().numpy()
        conf = YOLO_detections[0].boxes.conf.detach().cpu().numpy()
        dets = []
        for xywh_ind, xyo_ind, c in zip(xywh, xyo, conf):
            det = []
            det.append(ku.xywh2xyxy(list(xywh_ind)) + [c])
            for bp in config['bodyparts']:
                bp_xyoxyo = list(np.array(xyo_ind[2*config['bodyparts'].index(bp):2*config['bodyparts'].index(bp)+2]).flatten())
                if ku.valid_xyoxyo(bp_xyoxyo, config):
                    box_conf = c*((bp_xyoxyo[2]+bp_xyoxyo[-1])/2)
                    det.append(ku.xyoxyo2xyxy(bp_xyoxyo) + [box_conf])
                else:
                    det.append([0, 0, 0, 0, 0])
            for kp in config['additional_kps']:
                kp_xyo = list(np.array(xyo_ind[2*len(config['bodyparts']) + config['additional_kps'].index(kp)]))
                if kp_xyo[2] > config['bp_conf']:
                    box_conf = c*kp_xyo[2]
                    det.append([kp_xyo[0]-2, kp_xyo[1]-2, kp_xyo[0]+2, kp_xyo[1]+2, box_conf])
                else:
                    det.append([0, 0, 0, 0, 0])
            dets.append(det)
        return np.array(dets)
    
    elif config['detector'] == 'bounding_box_detection':
        xywh = YOLO_detections[0].boxes.xywh.detach().cpu().numpy()
        conf = YOLO_detections[0].boxes.conf.detach().cpu().numpy()
        components = YOLO_detections[0].boxes.cls.detach().cpu().numpy()
        comp_types = set(list(components))

        if len(list(components)) == 0:
            return np.zeros(shape=(0, len(config['components']), 5))
        num_salmon = max([list(components).count(comp) for comp in comp_types])
        dets = np.zeros((num_salmon, len(config['components']), 5))
        for comp_type in comp_types:
            comp_xywh = xywh[components == comp_type,:]
            comp_xyxy = np.zeros(comp_xywh.shape)
            for i in range(comp_xywh.shape[0]):
                comp_xyxy[i,:] = ku.xywh2xyxy(list(comp_xywh[i,:]))
            comp_conf = conf[components == comp_type]
            dets[:comp_xyxy.shape[0], int(comp_type)] = np.hstack([comp_xyxy, comp_conf.reshape(-1,1)])
        return dets