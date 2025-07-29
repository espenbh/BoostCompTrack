import cv2
import os
import sys
sys.path.append(os.path.split(os.getcwd())[0] + '\\salmon_component_tracking\\helpers')
import keybox_utils as ku
from comp_utils import get_salmon_ID_and_comp_type_from_comp_ID
import numpy as np
import math


def draw_text(img, text,
        font=cv2.FONT_HERSHEY_PLAIN,
        pos=(0, 0),
        font_scale=3,
        font_thickness=3,
        text_color=(255, 255, 255),
        text_color_bg=(0, 0, 0)
        ):
    '''
    A function to draw colored text on a colored background

    Args:
        font (cv2_fonts): Font of the text
        pos (tuple(int)): Position of the text
        font_scale (int): Size of the text
        font_thickness (int): Thickness of the text
        text_color (tuple(int)): Color of the text
        text_color_bg (tuple(int)): Background color of the text
    Operations:
        Draw text on the specified image
    Returns:
        Size of the text (in pixels?)
    '''
    # https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    x, y = pos
    y = y-text_h
    cv2.rectangle(img, (x, y), (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness, lineType = cv2.LINE_AA)

    return text_size


def draw_dashed_line(img, start_point, end_point, color, thickness, step = 10):
    '''
    Draw a dashed line.

    Args:
        img (np.array): The image to draw on
        start_point (tuple(int)): Start point of the line
        end_point (tuple(int)): End point of the line
        color (tuple(int)): Color of the line
        thickness (int): Thickness of the line
    Operations:
        Draws a dashed line on the provided image
    
    '''
    cur_point = start_point
    dir = (np.array(end_point)-np.array(start_point))/np.linalg.norm(np.array(end_point)-np.array(start_point))
    its = (np.linalg.norm(np.array(end_point)-np.array(start_point))-step)/(step*3)
    for i in range(math.ceil(its)):
        cv2.line(img, (int(cur_point[0]), int(cur_point[1])), (int(cur_point[0]+step*dir[0]), int(cur_point[1]+step*dir[1])), color, thickness)
        cur_point = cur_point + step*3*dir

def draw_dashed_box(img, xyxy, color, thickness):
    '''
    Draw a box with dashed lines.

    Args:
        img (np.array): The image to draw on
        xyxy (list(int)): Provide the two points defining the bounding box
        color (tuple(int)): Color of the line
        thickness (int): Thickness of the line
    Operations:
        Draws a dashed bounding box on the provided image
    '''
    x1, y1, x2, y2 = xyxy
    draw_dashed_line(img, (int(x1), int(y1)), (int(x2), int(y1)), color, thickness)
    draw_dashed_line(img, (int(x2), int(y1)), (int(x2), int(y2)), color, thickness)
    draw_dashed_line(img, (int(x2), int(y2)), (int(x1), int(y2)), color, thickness)
    draw_dashed_line(img, (int(x1), int(y2)), (int(x1), int(y1)), color, thickness)

def draw_complete_box(img, xyxy, color, thickness):
    '''
    Draw a box with whole lines.

    Args:
        img (np.array): The image to draw on
        xyxy (list(int)): Provide the two points defining the bounding box
        color (tuple(int)): Color of the line
        thickness (int): Thickness of the line
    Operations:
        Draws a bounding box with whole lines on the provided image
    '''
    cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, thickness)

def draw_bbox_connection(img, xyxy1, xyxy2, color = (0,0,0), thickness = 2):
    '''
    Draw whole lines between two matched bounding boxes.

    Args:
        img (np.array): The image to draw on
        xyxy1 (list(int)): Provide the two points defining the first bounding box
        xyxy2 (list(int)): Provide the two points defining the second bounding box
        color (tuple(int)): Color of the lines connecting the two bounding boxes
        thickness (int): Thickness of the lines connecting the two bounding boxes
    Operations:
        Draws whole lines between two matched bounding boxes on the provided image
    '''
    b1x1, b1y1, b1x2, b1y2 = xyxy1
    b2x1, b2y1, b2x2, b2y2 = xyxy2
    cv2.line(img, (int(b1x1), int(b1y1)), (int(b2x1), int(b2y1)), color = color, thickness=thickness)
    cv2.line(img, (int(b1x1), int(b1y2)), (int(b2x1), int(b2y2)), color = color, thickness=thickness)
    cv2.line(img, (int(b1x2), int(b1y1)), (int(b2x2), int(b2y1)), color = color, thickness=thickness)
    cv2.line(img, (int(b1x2), int(b1y2)), (int(b2x2), int(b2y2)), color = color, thickness=thickness)


def annotate_frame(targets, frame, frame_num, config, consider_salmon_composition, font_scale = 2, font_thickness = 2, skeleton_line_thickness = 2, include_id_text = True):
    '''
    Annotate a video frame.
    
    Args:
        targets (np.array): An array of shape Nobj x 7. Each object has values [x, y, x, y, ID, conf, component_type]
        frame (np.array): The video frame to be annotated
        frame_num (int): The frame count
        config (dict): Salmon config file
        consider_salmon_composition (bool): True if the link between salmon components and complete salmon is known
    Returns:
        An annotated input frame
    '''

    if consider_salmon_composition:
        salmon = {}

        # Construct a dictionary with keys corresponding to salmon IDs, and values being a list of the salmon component detections belonging to this salmon
        for t in targets:
            salmon_ID, _ = get_salmon_ID_and_comp_type_from_comp_ID(int(t[4]), len(config['components']))
            if salmon_ID in salmon:
                salmon[salmon_ID].append(t)
            else:
                salmon[salmon_ID] = [t]

        # Iterate over all salmon
        for k in salmon.keys():
            for t in salmon[k]:
                salmon_ID, salmon_comp_int = get_salmon_ID_and_comp_type_from_comp_ID(int(t[4]), len(config['components']))
                xyxy = t[0:4]
                if config['components'][int(salmon_comp_int)] in config['additional_kps']:
                    if int(xyxy[2]) < 0 or int(xyxy[3]) < 0:
                        cv2.circle(frame, (int(xyxy[0]), int(xyxy[1])), config['bp_bbox_thickness']*2, config['component_colors'][config['components'][int(salmon_comp_int)]], -1)
                    else:
                        cv2.rectangle(frame, (int(float(xyxy[0])), int(float(xyxy[1]))), (int(float(xyxy[2])), int(float(xyxy[3]))), config['component_colors'][config['components'][int(salmon_comp_int)]], -1)
                # If body part
                elif int(salmon_comp_int) != 0:
                    cv2.rectangle(frame, (int(float(xyxy[0])), int(float(xyxy[1]))), (int(float(xyxy[2])), int(float(xyxy[3]))), config['component_colors'][config['components'][int(salmon_comp_int)]], config['bp_bbox_thickness'])
                # If salmon
                else:
                    cv2.rectangle(frame, (int(float(xyxy[0])), int(float(xyxy[1]))), (int(float(xyxy[2])), int(float(xyxy[3]))), config['component_colors'][config['components'][int(salmon_comp_int)]], config['salmon_bbox_thickness'])
                    if include_id_text:
                        draw_text(frame, str(salmon_ID), pos = (int(xyxy[0]), int(xyxy[1])), font_scale=font_scale, font_thickness=font_thickness)

            # Draw skeleton
            for comp_type_1, comp_type_2 in config['skeleton']:
                comp1 = [l for l in salmon[k] if config['components'][int(l[6])] == comp_type_1]
                comp2 = [l for l in salmon[k] if config['components'][int(l[6])] == comp_type_2]
                if len(comp1) > 0 and len(comp2) > 0:
                    x1, y1 = ku.xyxy2xywh(comp1[0][:4])[0:2]
                    x2, y2 = ku.xyxy2xywh(comp2[0][:4])[0:2]               
                    frame = cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), thickness=skeleton_line_thickness)
    else:
        for t in targets:
            xyxy = t[0:4]
            if int(t[6]) != 0:
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), config['component_colors'][config['components'][int(t[6])]], config['bp_bbox_thickness'])
            else:
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), config['component_colors'][config['components'][int(t[6])]], config['salmon_bbox_thickness'])
            draw_text(frame, str(t[4]), pos = (int(xyxy[0]), int(xyxy[1])), font_scale=font_scale, font_thickness=font_thickness)

    draw_text(frame, 'Frame: ' + str(frame_num), pos = (int(50), int(config['video_shape'][1])-50), font_scale=font_scale, font_thickness=font_thickness)
    

    return frame
    