"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import os
from copy import deepcopy
from typing import Optional, List

import cv2
import numpy as np
import keybox_utils as ku

from default_settings import GeneralSettings, BoostTrackSettings, BoostTrackPlusPlusSettings
from tracker.embedding import EmbeddingComputer
from tracker.assoc import associate, iou_batch, MhDist_similarity, shape_similarity, soft_biou_batch
from tracker.ecc import ECC
from tracker.kalmanfilter import KalmanFilter
import imutils


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,h,r] where x,y is the centre of the box and h is the height and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0

    r = w / float(h + 1e-6)

    return np.array([x, y, h, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,h,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """

    h = x[2]
    r = x[3]
    w = 0 if r <= 0 else r * h

    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, emb: Optional[np.ndarray] = None):
        """
        Initialises a tracker using initial bounding box.
        """

        self.bbox_to_z_func = convert_bbox_to_z
        self.x_to_bbox_func = convert_x_to_bbox

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.kf = KalmanFilter(self.bbox_to_z_func(bbox))
        self.emb = emb
        self.hit_streak = 0
        self.age = 0

    def get_confidence(self, coef: float = 0.9) -> float:
        n = 7

        if self.age < n:
            return coef ** (n - self.age)
        return coef ** (self.time_since_update-1)

    def update(self, bbox: np.ndarray, score: float = 0):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(self.bbox_to_z_func(bbox), score)

    def camera_update(self, transform: np.ndarray):
        x1, y1, x2, y2 = self.get_state()[0]
        x1_, y1_, _ = transform @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = transform @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.kf.x[:4] = [cx, cy, h,  w / h]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_state()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb
    
    def get_comp_kf_x(self, n_dims: float): # Add
        return self.kf.x[:n_dims] # Add
    
    def get_comp_kf_cv(self, n_dims: float): # Add
        return self.kf.covariance[:n_dims, :n_dims] # Add


class SalmonTracker(object):
    '''
    Object to represent a complete salmon.
    
    Attributes:        
        id (int): Identifier of the fish
        hit_streak (int): Number of frames since the last complete salmon occlusion
        time_since_update (int): Number of frames since the last update
        eps (float): A small number
        tracker_dict (dict): A dictionary with KalmanBoxTracker objects for all salmon components
        turn (int): A flag specifying if the SalmonTracker is turning. if turn > 0, the tracker is turning.
    '''

    count = 1

    def __init__(self, xyxycs: np.ndarray, embs: np.ndarray, eps: float = 0.01):
        '''
        Initialize a SalmonTracker object

        Args:
            xyxycs (np.ndarray): A numpy array of size 9x5, where each of the 9 row describes the bounding box (xyxy) and the confidence (c) of a salmon component
            embs: A numpy array of size 9x512, where each of the 9 rows describe the appaerance of the salmon component
            eps: A small value
        '''

        self.id = SalmonTracker.count
        SalmonTracker.count += 1

        self.hit_streak = 0
        self.time_since_update = 0

        self.eps = eps
        self.tracker_dict = {k:None for k in range(xyxycs.shape[0])}
        for xyxyc, emb, comp_int in zip(xyxycs, embs, range(xyxycs.shape[0])):
            if xyxyc[4] > self.eps:
                self.tracker_dict[comp_int] = KalmanBoxTracker(xyxyc, emb=emb)
        self.turn = 0

    def update(self, xyxycs: np.ndarray, embs: np.ndarray, alphas: np.ndarray, turn):
        ''' 
        Update a SalmonTracker objects with new detections.

        Args:
            xyxycs (np.ndarray): A numpy array of size 9x5, where each of the 9 row describes the bounding box (xyxy) and the confidence (c) of a salmon component
            embs (np.ndarray): A numpy array of size 9x512, where each of the 9 rows describe the appaerance of the salmon component
            alphas (np.ndarray): A numpy array of size 9, where each value describe the updating factor alpha of the embedding
        '''
        xyxys = xyxycs[:,:4]
        scores = xyxycs[:,4]
        upd = False
        for xyxy, score, emb, alpha, comp_int in zip(xyxys, scores, embs, alphas, range(xyxycs.shape[0])):
            if score < self.eps:
                # Body part not detected. Do not update body part tracker
                continue
            elif self.tracker_dict[comp_int] == None:
                # No tracker available. Create tracker.
                self.tracker_dict[comp_int] = KalmanBoxTracker(xyxy, emb=emb)
            else:
                # Available detection and available body part tracker.
                self.tracker_dict[comp_int].update(xyxy, score)
                self.tracker_dict[comp_int].update_emb(emb, alpha)
                upd = True
        if upd:
            self.hit_streak = self.hit_streak + 1
            self.time_since_update = 0

    def camera_update(self, transform: np.ndarray):
        ''' 
        Update camera compensation for all salmon components
        '''
        for c in self.tracker_dict.keys():
            if self.tracker_dict[c] != None:
                self.tracker_dict[c].camera_update(transform)


    def predict(self):
        '''
        Kalman predict step for all salmon components

        Returns:
            - A 9x4 numpy array containing the new location of the bounding boxes of all salmon components.
            If a component does not have a Kalman filter, the position is set to [0,0,0,0] 
        '''
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        xyxys = []
        for c in self.tracker_dict.keys():
            if self.tracker_dict[c] != None:
                xyxys.append(self.tracker_dict[c].predict()[0])
            else:
                xyxys.append([0,0,0,0])
        return np.array(xyxys)

    def get_confidence(self, coef: float = 0.9) -> float:
        '''
        Get confidence of all salmon components

        Returns:
            - A numpy array of shape 9 containing the tracking confidence of all salmon components. 
            If a component does not have a Kalman filter, the confidence is set to 0
        '''
        confs = []
        for bp_int in self.tracker_dict.keys():
            if self.tracker_dict[bp_int] != None:
                confs.append(self.tracker_dict[bp_int].get_confidence(coef=coef))
            else:
                confs.append(0)
        return np.array(confs)

    def get_comp_state(self, comp_int):
        '''
        Get state of a single salmon component

        Args:
            - comp_int (int): The index of the salmon component

        Returns:
            - A numpy array with the bounding box location of the salmon component (xyxy) 
        '''

        if self.tracker_dict[comp_int] == None:
            return np.zeros([1,4]) 
        return self.tracker_dict[comp_int].get_state()

    def get_salmon_ID(self):
        '''
        Get the ID of the salmon
        '''
        
        return self.id
    
    def get_comp_ID(self, comp_int, num_comps: int = 9):
        ''' 
        Get the component ID of a slamon component
        
        Args:
            - comp_int (int): The index of the salmon component
            - num_comps (int): The number of salmon components

        Returns:
            - The ID of the salmon component
        
        '''

        return self.id*num_comps + comp_int
    
    def get_comp_kf_x(self, comp_int, n_dims):
        '''
            Return the Kalman states (xywa) of a specified salmon component

            Args:
                - comp_int (int): The index of the salmon component
                - n_dims (int): The number of included values in the Kalman state vector

            Returns:
                - A numpy array of size n_dims containing Kalman state values
        '''

        if self.tracker_dict[comp_int] == None:
            return np.zeros(n_dims)
        return self.tracker_dict[comp_int].get_comp_kf_x(n_dims)

    def get_comp_kf_cv(self, comp_int, n_dims):
        '''
        Get the Kalman covariance matrix of a salmon component

        Args:
            - comp_int (int): The index of the salmon component
            - n_dims (int): The number of included values in the Kalman state vector
        
        Returns:
            - A numpy array of size n_dims x n_dims containing Kalman state values.
            If a component does not have a Kalman filter, the covariance matrix is set to a diagonal, high-valued matrix

        '''
        if self.tracker_dict[comp_int] == None:
            # Large variance if the body part is not yet detected
            return np.diag([1e6 for i in range(n_dims)])
        return self.tracker_dict[comp_int].get_comp_kf_cv(n_dims)
    
    def get_emb(self, comp_int: int, emb_size: int = 512):
        '''
        Get the appearence embedding of a salmon component

        Args:
            - comp_int (int): The index of the salmon component
            - emb_size (int): The size of the embedding
        '''
        if self.tracker_dict[comp_int] is not None:
            return self.tracker_dict[comp_int].get_emb()
        else:
            return np.zeros(emb_size)



class BoostTrack(object):
    def __init__(self, video_name: Optional[str] = None, 
                 generalsettings: GeneralSettings = GeneralSettings(), 
                 boosttracksettings: BoostTrackSettings = BoostTrackSettings(),
                 boosttrackplusplussettings: BoostTrackPlusPlusSettings = BoostTrackPlusPlusSettings()):

        self.frame_count = 0
        self.trackers: List[KalmanBoxTracker] = []

        self.max_age = generalsettings.max_age(video_name)
        self.max_age = generalsettings.values['max_age']
        self.debug = generalsettings.values['debug']
        self.iou_threshold = generalsettings.values['iou_threshold']
        self.det_thresh = generalsettings.values['det_thresh']
        self.min_hits = generalsettings.values['min_hits']
        self.consider_salmon_composition = generalsettings.values['consider_salmon_composition']
        self.eps = generalsettings.values['eps'] 
        self.ncomp = generalsettings.values['ncomp'] 
        self.embedding_size = generalsettings.values['embedding_size'] 
        if not self.consider_salmon_composition: 
            self.trackers: dict[int, KalmanBoxTracker] =  {k: [] for k in range(self.ncomp)}  
        else: 
            self.trackers: List[SalmonTracker] = [] 

        self.lambda_iou = boosttracksettings.values['lambda_iou']
        self.lambda_mhd = boosttracksettings.values['lambda_mhd']
        self.lambda_shape = boosttracksettings.values['lambda_shape']
        self.use_dlo_boost = boosttracksettings.values['use_dlo_boost']
        self.use_duo_boost = boosttracksettings.values['use_duo_boost']
        self.dlo_boost_coef = boosttracksettings.values['dlo_boost_coef']

        self.use_rich_s = boosttrackplusplussettings.values['use_rich_s']
        self.use_sb = boosttrackplusplussettings.values['use_sb']
        self.use_vt = boosttrackplusplussettings.values['use_vt']

        if generalsettings.values['use_embedding']:
            self.embedder: dict[int, EmbeddingComputer] = {k: EmbeddingComputer(GeneralSettings['dataset'], GeneralSettings['test_dataset'], True) for k in range(self.ncomp)}
        else:
            self.embedder = {k: None for k in range(self.ncomp)}

        if generalsettings.values['use_ecc']:
            print('Using camera compensation')
            self.ecc: dict[int, ECC] = {k: ECC(scale=350, video_name=video_name, use_cache=True) for k in range(self.ncomp)}
        else:
            self.ecc = {k: None for k in range(self.ncomp)}

        # Using rotated images
        acceptable_area = np.zeros([2160,3840])
        acceptable_area[5:2160-5, 5:3840-5] = 1
        self.acceptable_grid = imutils.rotate_bound(acceptable_area, 135)

        self.activate_turn = generalsettings.values['activate_turn'] 

        self.activate_salmon_error_bp_disagreement_check = generalsettings.values['activate_salmon_error_bp_disagreement_check'] 
        self.activate_salmon_error_no_bp_overlap_check = generalsettings.values['activate_salmon_error_no_bp_overlap_check'] 
        self.activate_bp_error_bp_disagreement_check = generalsettings.values['activate_bp_error_bp_disagreement_check'] 

    def update(self, dets, img_tensor, img_numpy, tag, frame_num):
        """
        Params:
          dets - a 3D numpy array of detections in the format [[[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]]. Shape Nsal x ncomp x 5
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 9, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        # Check if no detections are provided, and that the detections are located at the cpu
        if dets is None:
            return np.empty((0, self.ncomp, 5))
        if not isinstance(dets, np.ndarray):
            dets = dets.cpu().detach().numpy()

        # Update frame count
        self.frame_count += 1

        # Rescale
        scale = min(img_tensor.shape[2] / img_numpy.shape[0], img_tensor.shape[3] / img_numpy.shape[1])

        # Camera compensation
        for comp_int in range(self.ncomp):
            trackers = self.trackers if self.consider_salmon_composition else self.trackers[comp_int]

            # Camera calibration
            if self.ecc[comp_int] is not None:
                transform = self.ecc[comp_int](img_numpy, self.frame_count, tag)
                for trk in trackers:
                    trk.camera_update(transform)

        # Only consider detections with confidence above det_thresh
        dets[dets[..., :]<self.det_thresh] = 0
        turns = np.zeros(len(self.trackers)).astype(int)

        if self.consider_salmon_composition:
            # get predicted locations from existing trackers.
            trks = np.zeros((len(self.trackers), self.ncomp, 5))
            trk_confs = np.zeros((len(self.trackers), self.ncomp))

            for t in range(len(trks)):
                turns[t] = self.trackers[t].turn
                trks[t,:,:4] = self.trackers[t].predict()
                trk_confs[t] = self.trackers[t].get_confidence()

            trks[:,:,4] = trk_confs

            # Intialize embedding and alpha arrays for all salmon detections
            embs = np.zeros((dets.shape[0], dets.shape[1], self.embedding_size))
            alphas = np.zeros(dets.shape[0:2])

            # Initialize data structures to store candidate salmon component matches
            cost_matrices = np.zeros((self.ncomp, dets.shape[0], len(self.trackers)))
            detidx2trkidx = {detid: (np.ones(self.ncomp)*-1).astype(int) for detid in range(dets.shape[0])}

        # Initialize list to store all tracked salmon components
        ret = []

        # Consider each salmon component sequentially
        for comp_int in range(self.ncomp):
            trackers = self.trackers if self.consider_salmon_composition else self.trackers[comp_int]

            # Extract tracker location and confidence for given component
            if self.consider_salmon_composition:
                comp_trks = trks[:,comp_int,:]
                comp_trk_confs = trks[:,comp_int,4]
            else:
                # get predicted locations from existing trackers.
                comp_trks = np.zeros((len(trackers), 5))
                comp_trk_confs = np.zeros((len(trackers)))

                for t in range(len(comp_trks)):
                    pos = trackers[t].predict()[0]
                    comp_trk_confs[t] = trackers[t].get_confidence()
                    comp_trks[t] = [pos[0], pos[1], pos[2], pos[3], comp_trk_confs[t]]

            # Extract detection location and confidence for given component
            comp_dets = deepcopy(dets[:,comp_int,:])
            comp_dets[:, :4] /= scale

            # Only pass visible salmon components through BoostTrack operations
            nonzero_dets = comp_dets[comp_dets[:,4]>self.eps] #if self.consider_salmon_composition else comp_dets 

            # BoostTrack detection operations
            if self.use_dlo_boost:
                nonzero_dets = self.dlo_confidence_boost(nonzero_dets, comp_int, self.use_rich_s, self.use_sb, self.use_vt)

            if self.use_duo_boost:
                nonzero_dets = self.duo_confidence_boost(nonzero_dets, comp_int)
            
            # Add the modified detections back into the original array, to retain the appropriate structure of the detection array
            if self.consider_salmon_composition:
                comp_dets[comp_dets[:,4]>self.eps] = nonzero_dets
            else:
                comp_dets = nonzero_dets[nonzero_dets[:, 4] > self.eps]

            # Extract confidence for the current components
            comp_det_scores = comp_dets[:, 4]

            # Generate embeddings
            dets_embs = np.ones((comp_dets.shape[0], 1))
            emb_cost = None
            
            if self.embedder[comp_int] and comp_dets.size > 0:
                # Calculate the embeddings of all non-zero detections
                nonzero_idces = np.sum(comp_dets, axis = 1) > self.eps
                dets_embs[nonzero_idces] = self.embedder[comp_int].compute_embedding(img_numpy, comp_dets[nonzero_idces, :4], tag)
                
                # Calculate tracker embeddings
                trk_embs = []
                for t in range(len(trackers)):
                    if not self.consider_salmon_composition:
                        trk_embs.append(trackers[t].get_emb())
                    else:
                        trk_embs.append(trackers[t].get_emb(comp_int))
                trk_embs = np.array(trk_embs)
                if trk_embs.size > 0 and comp_dets.size > 0:

                    # Calculate a matrix with embedding similarities of shape(Ndets x Ntrks)
                    emb_cost = dets_embs.reshape(dets_embs.shape[0], -1) @ trk_embs.reshape((trk_embs.shape[0], -1)).T
            emb_cost = None if self.embedder[comp_int] is None else emb_cost
            mahalanobis_matrix = self.get_mh_dist_matrix(comp_dets, comp_int)

            pass_turns = []
            if self.debug == False and self.consider_salmon_composition and self.activate_turn:
                pass_turns = turns
            matched, unmatched_dets, _, cost_matrix = associate(
                comp_dets,
                comp_trks,
                self.iou_threshold,
                mahalanobis_distance=mahalanobis_matrix,
                track_confidence=comp_trk_confs,
                detection_confidence=comp_det_scores,
                emb_cost=emb_cost,
                lambda_iou=self.lambda_iou,
                lambda_mhd=self.lambda_mhd,
                lambda_shape=self.lambda_shape,
                turns = pass_turns,
            )

            # Calculate the embedding update parameter alpha
            trust = (comp_dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
            af = 0.95
            dets_alpha = af + (1 - af) * (1 - trust)

            # Assign matches if salmon composition is not considered
            if not self.consider_salmon_composition:

                # If a tracker and detection is matched
                for m in matched:
                    trackers[m[1]].update(comp_dets[m[0], :], comp_det_scores[m[0]])
                    trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

                # Create trackers for all unmatched detections
                for i in unmatched_dets:
                    trackers.append(KalmanBoxTracker(comp_dets[i, :], emb=dets_embs[i]))

                # Retrieve a list of tracked detections
                ret_comp, trackers = self.construct_comp_ret_list(trackers, comp_int)

                # Update self.tracker from temporary tracker object
                self.trackers[comp_int] = trackers

                # Add tracked detections to return list
                ret.extend(ret_comp)


            if self.consider_salmon_composition:
                # Update data structures with information from individual component association
                dets[:,comp_int,:] = comp_dets
                embs[:,comp_int,:] = dets_embs
                alphas[:,comp_int] = dets_alpha

                if cost_matrix.shape[0] > 0 and cost_matrix.shape[1] > 0:
                    cost_matrices[comp_int,:,:] = cost_matrix

                for m in matched:
                    detidx2trkidx[m[0]][comp_int] = m[1] #trkidx
        
        debug = []
        # Refine individual component associations into salmon associations
        if self.consider_salmon_composition:
            video_shape = (img_numpy.shape[1], img_numpy.shape[0])
            matches, unmatched_dets, error_trackers, debug, turns, holdout_bps = self.comp_matches_to_salmon_matches(dets, cost_matrices, detidx2trkidx, frame_num, video_shape, turns)
            holdout_bps = np.array(holdout_bps)

            # Update turns flag in the trackers
            for t in range(turns.shape[0]):
                self.trackers[t].turn = turns[t]

            #dets[dets[..., :]<self.det_thresh] = 0
            debug = [[str(int(i[0])), str(self.trackers[i[1]].id+1), str(i[2]), str(i[3])] for i in debug]

            # The SalmonTracker object performs both Kalman updates and embedding updates in its update() function
            for m in matches:
                pass_dets = dets[m[0], :, :]
                for holdout_bp in holdout_bps:
                    if holdout_bp[1] == m[0] and holdout_bp[2] == m[1]:
                        pass_dets[holdout_bp[0], 4] = 0
                self.trackers[m[1]].update(pass_dets, embs[m[0], :], alphas[m[0], :], turns[m[1]])

            # Create new SalmonTracker objects for unmatched detections
            for i in unmatched_dets:
                pass_dets = dets[i, :, :]
                for holdout_bp in holdout_bps:
                    if holdout_bp[1] == m[0] and holdout_bp[2] == m[1]:
                        pass_dets[holdout_bp[0], 4] = 0
                self.trackers.append(SalmonTracker(pass_dets, embs[i, :]))

            # Fill return list of tracked detections
            for comp_int in range(self.ncomp):
                ret_bp, trackers = self.construct_comp_ret_list(self.trackers, comp_int, error_trackers = error_trackers)
                if comp_int == 0:
                    self.trackers = trackers
                ret.extend(ret_bp)

        # Return ret list
        if len(ret) > 0:
            return np.concatenate(ret), debug
        return np.empty((0, 5)), debug

    def construct_comp_ret_list(self, tracker_list, comp_int, error_trackers = []):
        '''
        Construct a list of tracked detections, and remove old trackers from the tracker list.

        Args:
            - tracker_list (list[TrackerObject]): A list of tracker objects (SalmonTracker or KalmanBoxTracker)
            - comp_int (int): The index of the salmon component
            - error_trackers: List of indices of erroneous trackers.
        
        Returns:
            - ret (list[np.ndarray]): A list of numpy arrays, each on the form [x, y, x, y, ID, conf, class]
            - tracker_list (list[TrackerObject]): A list of tracked objects with old trackers removed
        '''
        ret = []
        i = len(tracker_list)
        for trk in reversed(tracker_list):
            if i not in error_trackers or comp_int >= 1 or (not self.consider_salmon_composition):
                comp_trk = trk.tracker_dict[comp_int] if self.consider_salmon_composition else trk
                if comp_trk == None:
                    continue
                d = comp_trk.get_state()[0]
                # Filter out which tracks to return
                if (comp_trk.time_since_update < 1) and (comp_trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                    id = trk.get_comp_ID(comp_int, self.ncomp) if self.consider_salmon_composition else trk.id
                    # id+ncomp due to evaluation requirements
                    ret.append(np.concatenate((d, [id+self.ncomp], [comp_trk.get_confidence()], [comp_int], [-1])).reshape(1, -1))

            # Update tracker index counter
            i -= 1

            # remove dead tracks
            if comp_int == 0 or (not self.consider_salmon_composition):
                if trk.time_since_update > self.max_age:
                    tracker_list.pop(i)
                elif not self.debug and (i in error_trackers):
                    tracker_list.pop(i)

        return ret, tracker_list
    
    def comp_matches_to_salmon_matches(self, dets, cost_matrices, detidx2trkidx, frame_num, video_shape, turns):
        '''
        Find matches between salmon and detections from matches between salmon components and detections

        Args:
            - dets (np.array): Numpy array of detector detections. Shape (salmon count x ncomp x 5)
            - cost_matrices (np.array): Numpy array of IoU matrices. Shape (ncomp x salmon count x tracker count)
            - detidx2trkidx: Dictionary on the form {det_idx: np.array([trk_idx_salmon, -1, trk_idx_dorsal_fin, -1, -1, -1, -1, -1, trk_idx_body], 1: ...). 
            The keys are detection indices, the values are a numpy array of the associated tracker indices of each component.
            - frame_num (int): Frame number
            - video_shape (int): Shape of the input video
            - turns (list[int]): List of turn values
        Returns:
            - matches (list[list[int]]): A list of all matched detections and salmon trackers. On the format [[detection index, tracker_ID], ...]
            - unmatched_dets (list[int]): A list of unmatched detection indices
            - error_trackers (list[int]): A list of the indices of erroneous trackers (according to the checks in this function)
            - debug (bool): If degub is true, all checks are disabled.
            - turns (list[int]): Updated turns values
            - holdout_bps (list[list[float]]): A list of lists, where each sublist contains [component index, detection match, tracker match]
            of components that disagrees with the salmon association.
        '''
        matches = []
        unmatched_dets = []
        debug = []
        error_trackers = []
        holdout_bps = []

        for det_match in range(dets.shape[0]): 
            trk_match = None
            error = False

            if detidx2trkidx[det_match][0] != -1:
                trk_match = detidx2trkidx[det_match][0]

                # Salmon turning logic
                salmon_turn = turns[trk_match]

                # Check if salmon is in the frame (i.e. not bordering the edge of the image)
                xyxy = dets[det_match, 0, :4]
                xywh = ku.xyxy2xywh(xyxy)
                if video_shape[0] == 4242 and video_shape[1] == 4242:
                    if np.min(xyxy)<=0 or np.max(xyxy)>=4242:
                        salmon_in_frame = False
                    else:                        
                        salmon_in_frame = self.acceptable_grid[int(xyxy[1]), int(xyxy[0])] == 1 and self.acceptable_grid[int(xyxy[3]), int(xyxy[2])] == 1
                else:
                    salmon_in_frame = xyxy[0] > 5 and xyxy[1] > 5 and xyxy[2] < video_shape[0]-5 and xyxy[3] < video_shape[1]-5

                # If the salmon is bordering the edge of the image, reset the turning counter
                if not salmon_in_frame:
                    salmon_turn = 0

                # If the height of the salmon is larger than the width, increase the turning counter
                elif xywh[3] > xywh[2]:
                    salmon_turn = min(salmon_turn + 1, 10)
            
                # If the distance between the head and tail fin is shorter than twice the distance between the dorsal fin and pelvic fin, increase the turning counter
                elif dets[det_match, 1, 4] > self.eps and dets[det_match, 4, 4] > self.eps and dets[det_match, 2, 4] > self.eps and dets[det_match, 6, 4] > self.eps:
                    xyxy_head = dets[det_match, 1, :4]
                    xywh_head = ku.xyxy2xywh(xyxy_head)
                    xyxy_tailfin = dets[det_match, 4, :4]
                    xywh_tailfin = ku.xyxy2xywh(xyxy_tailfin)
                    xyxy_dfin = dets[det_match, 2, :4]
                    xywh_dfin = ku.xyxy2xywh(xyxy_dfin)
                    xyxy_pelvfin = dets[det_match, 6, :4]
                    xywh_pelvfin = ku.xyxy2xywh(xyxy_pelvfin)
                    if np.linalg.norm(np.array(xywh_head[0:2]) - np.array(xywh_tailfin[0:2])) < 2*np.linalg.norm(np.array(xywh_dfin[0:2]) - np.array(xywh_pelvfin[0:2])):
                        salmon_turn = min(salmon_turn + 1, 10)

                # If the turn counter has not been modified, and most body parts are visible, decrease the turn counter
                if salmon_turn == turns[trk_match] and sum(dets[det_match, :, 4] > self.eps) >= 7:
                    salmon_turn = max(0, salmon_turn - 1)
               
                turns[trk_match] = salmon_turn

                # Update debug information
                if salmon_turn >= 1:
                    debug.append([frame_num, detidx2trkidx[det_match][0], 'all', 'turning_' + str(salmon_turn)])

                # Only perform transfer-reducing logic of the salmon is not turning
                if salmon_turn <= 0 or not self.activate_turn:
                    all_ious = []
                    for comp_int in range(1,8):
                        iou = cost_matrices[int(comp_int),int(det_match),int(trk_match)]
                        if iou >= self.eps: all_ious.append(iou)

                    # Check if body parts are associated with a different salmon
                    agree = [i for i in detidx2trkidx[det_match][1:] if (i != -1 and i == trk_match)]
                    disagree = [i for i in detidx2trkidx[det_match][1:] if (i != -1 and i != trk_match)]
                    if len(disagree) > 1 and len(disagree) > len(agree):
                        debug.append([frame_num, detidx2trkidx[det_match][0], 'all', 'salmon_error_bp_disagreement'])
                        if (not self.debug) and self.activate_salmon_error_bp_disagreement_check:
                            error = True

                    # Check if the IOUs are low for all body parts, given the salmon association
                    if len(all_ious) == 0: #np.sum(all_ious) <= self.iou_threshold/2: #and time_since_updates[trk_match] <= 3 #and time_since_updates[trk_match] <= 3: #and len(all_ious) > 0: #self.iou_threshold/2 and len(all_ious) > 0:
                        debug.append([frame_num, detidx2trkidx[det_match][0], 'all', 'salmon_error_no_bp_overlap'])
                        if (not self.debug) and self.activate_salmon_error_no_bp_overlap_check:
                            error = True
                    
                    for comp_int in range(1,8):
                        trk_match_value = cost_matrices[comp_int, det_match, trk_match]
                        trk_costs = cost_matrices[comp_int, det_match, :]
                        top5_indices = np.argsort(trk_costs)[-5:][::-1]
                        top5_indices = [i for i in top5_indices if i != trk_match]
                        top5_values = trk_costs[top5_indices]

                        if top5_values[0]/(trk_match_value + self.eps) > 1.0:
                            debug.append([frame_num, detidx2trkidx[det_match][0], comp_int, 'bp_error_bp_disagreement'])
                            if (not self.debug) and self.activate_bp_error_bp_disagreement_check:
                                holdout_bps.append([comp_int, det_match, trk_match])

            if trk_match == None:
                unmatched_dets.append(det_match)
            else:
                matches.append([int(det_match), int(trk_match)])

            if error:
                error_trackers.append(trk_match)

        return matches, unmatched_dets, error_trackers, debug, turns, holdout_bps


    def dump_cache(self):
        if self.ecc is not None:
            self.ecc.save_cache()
    
    def get_iou_matrix(self, detections: np.ndarray, comp_int: int, buffered: bool = False) -> np.ndarray:
        tracker_objects = self.trackers[comp_int] if not self.consider_salmon_composition else self.trackers
        trackers = np.zeros((len(tracker_objects), 5))
        for t, trk in enumerate(trackers):
            pos = tracker_objects[t].get_comp_state(comp_int)[0] if self.consider_salmon_composition else tracker_objects[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], tracker_objects[t].get_confidence()[comp_int]] if self.consider_salmon_composition else [pos[0], pos[1], pos[2], pos[3], tracker_objects[t].get_confidence()]

        return iou_batch(detections, trackers) if not buffered else soft_biou_batch(detections, trackers)

    def get_mh_dist_matrix(self, detections: np.ndarray, comp_int: int, n_dims: int = 4) -> np.ndarray:
        tracker_objects = self.trackers[comp_int] if not self.consider_salmon_composition else self.trackers
        if len(tracker_objects) == 0:
            return np.zeros((0, 0))
        z = np.zeros((len(detections), n_dims), dtype=float)
        x = np.zeros((len(tracker_objects), n_dims), dtype=float)
        sigma_inv = np.zeros_like(x, dtype=float)

        f = convert_bbox_to_z
        for i in range(len(detections)):
            z[i, :n_dims] = f(detections[i, :]).reshape((-1, ))[:n_dims]
        for i in range(len(tracker_objects)):
            x[i] = tracker_objects[i].get_comp_kf_x(comp_int, n_dims) if self.consider_salmon_composition else tracker_objects[i].get_comp_kf_x(n_dims)
            # Note: we assume diagonal covariance matrix
            sigma_inv[i] = np.reciprocal(np.diag(tracker_objects[i].get_comp_kf_cv(comp_int, n_dims))) if self.consider_salmon_composition else np.reciprocal(np.diag(tracker_objects[i].get_comp_kf_cv(n_dims)))

        return ((z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2 * sigma_inv.reshape((1, -1, n_dims))).sum(axis=2)

    def duo_confidence_boost(self, detections: np.ndarray, comp_int: int) -> np.ndarray:
        n_dims = 4
        limit = 13.2767
        mahalanobis_distance = self.get_mh_dist_matrix(detections, comp_int, n_dims)

        if mahalanobis_distance.size > 0 and self.frame_count > 1:
            min_mh_dists = mahalanobis_distance.min(1)

            mask = (min_mh_dists > limit) & (detections[:, 4] < self.det_thresh)
            boost_detections = detections[mask]
            boost_detections_args = np.argwhere(mask).reshape((-1,))
            iou_limit = 0.3
            if len(boost_detections) > 0:
                bdiou = iou_batch(boost_detections, boost_detections) - np.eye(len(boost_detections))
                bdiou_max = bdiou.max(axis=1)

                remaining_boxes = boost_detections_args[bdiou_max <= iou_limit]
                args = np.argwhere(bdiou_max > iou_limit).reshape((-1,))
                for i in range(len(args)):
                    boxi = args[i]
                    tmp = np.argwhere(bdiou[boxi] > iou_limit).reshape((-1,))
                    args_tmp = np.append(np.intersect1d(boost_detections_args[args], boost_detections_args[tmp]), boost_detections_args[boxi])

                    conf_max = np.max(detections[args_tmp, 4])
                    if detections[boost_detections_args[boxi], 4] == conf_max:
                        remaining_boxes = np.array(remaining_boxes.tolist() + [boost_detections_args[boxi]])

                mask = np.zeros_like(detections[:, 4], dtype=np.bool_)
                mask[remaining_boxes] = True

            detections[:, 4] = np.where(mask, self.det_thresh + 1e-4, detections[:, 4])

        return detections

    def dlo_confidence_boost(self, detections: np.ndarray, comp_int: int, use_rich_sim: bool, use_soft_boost: bool, use_varying_th: bool) -> np.ndarray:
        sbiou_matrix = self.get_iou_matrix(detections, comp_int, True)
        if sbiou_matrix.size == 0:
            return detections
        tracker_objects = self.trackers[comp_int] if not self.consider_salmon_composition else self.trackers

        trackers = np.zeros((len(tracker_objects), 6))
        for t, trk in enumerate(trackers):
            pos = tracker_objects[t].get_comp_state(comp_int)[0] if self.consider_salmon_composition else tracker_objects[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, tracker_objects[t].time_since_update - 1]

        if use_rich_sim:
            mhd_sim = MhDist_similarity(self.get_mh_dist_matrix(detections, comp_int), 1)
            shape_sim = shape_similarity(detections, trackers)
            S = (mhd_sim + shape_sim + sbiou_matrix) / 3
        else:
            S = self.get_iou_matrix(detections, comp_int, False)

        if not use_soft_boost and not use_varying_th:
            max_s = S.max(1)
            coef = self.dlo_boost_coef
            detections[:, 4] = np.maximum(detections[:, 4], max_s * coef)

        else:
            if use_soft_boost:
                max_s = S.max(1)
                alpha = 0.65
                detections[:, 4] = np.maximum(detections[:, 4], alpha*detections[:, 4] + (1-alpha)*max_s**(1.5))
            if use_varying_th:
                threshold_s = 0.95
                threshold_e = 0.8
                n_steps = 20
                alpha = (threshold_s - threshold_e) / n_steps
                tmp = (S > np.maximum(threshold_s - trackers[:, 5] * alpha, threshold_e)).max(1)
                scores = deepcopy(detections[:, 4])
                scores[tmp] = np.maximum(scores[tmp], self.det_thresh + 1e-5)

                detections[:, 4] = scores

        return detections

