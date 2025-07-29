import random
import shutil
import os
import json
import numpy as np
import yaml
import sys
import magic

sys.path.append(os.path.split(os.getcwd())[0] + '\\helpers')
from keybox_utils import xyxy2xywh, normxywh, xyxy2xyxy_ordered, normxyxy

def get_paths_to_labelme(labelme_root: str) -> list[str]:
    '''
    This function provides a list of all available annotations for a given task
    The labels should be in the folder: labelme_root\\labels

    Args:
        - labelme_root. The root of the folder containing all data

    File operations:

    Returns:
        - The path to all manual annotations, referenced from the data_root.
    '''

    all_label_paths = []
    for fol1 in os.listdir(labelme_root):
        all_label_paths.extend([fol1 + '\\labels\\' + f for f in os.listdir(labelme_root + fol1 + '\\labels\\')])
    return all_label_paths

def get_paths_to_images(labelme_root: str) -> list[str]:
    '''
    This function provides a list of all available annotations for a given task
    The labels should be in the folder: labelme_root\\labels

    Args:
        - labelme_root. The root of the folder containing all data

    File operations:

    Returns:
        - The path to all manual annotations, referenced from the data_root.
    '''

    all_image_paths = []
    for fol1 in os.listdir(labelme_root):
        all_image_paths.extend([fol1 + '\\images\\' + f for f in os.listdir(labelme_root + fol1 + '\\images\\')])
    return all_image_paths

def generate_yaml_file(YOLO_root: str, task: str, bodyparts: list[str], additional_kps = []) -> None:
    '''
    Generates a YAML file on the YOLO format

    Args:
        - YOLO_root. Save path.
        - Task. The considered task. Can be bounding_box_detection or keybox_detection
        - bodyparts. The bodyparts of the salmon.
        - additional_kps. Specify additional key points (jaws, eye)

    File operations:
        - Creates a YOLO config file

    Returns:

    '''

    random.seed(142904)
    label_name_dict = {}
    flip_idx = [i for i in range(len(bodyparts)*2 + len(additional_kps))]
    for label_name, i in zip(['salmon'] + bodyparts, range(len(['salmon'] + bodyparts))):
        label_name_dict[str(i)] = label_name

        if task == 'bounding_box_detection':
            config = {
                # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
                'path': os.path.join(YOLO_root, 'YOLO_data'),  # dataset root dir
                'train': 'train',  # train images (relative to 'path') 4 images
                'val': 'val',  # val images (relative to 'path') 4 images
                'test': '',# test images (optional)
                # Classes dictionary
                'names': label_name_dict
            }

        elif task.startswith('keybox_detection'):
            config = {
                # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
                'path': os.path.join(YOLO_root, 'YOLO_data'),  # dataset root dir
                'train': 'train',  # train images (relative to 'path') 4 images
                'val': 'val',  # val images (relative to 'path') 4 images
                'test': '',# test images (optional)

                'kpt_shape': [len(bodyparts)*2 + len(additional_kps), 3],
                'flip_idx': flip_idx,

                # Classes dictionary
                'names': {0: 'salmon'},
                #'kpt_radius': 1,
            }
    
    with open(os.path.join(os.path.join(YOLO_root,'YOLO_data'), 'config.yml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def get_img_paths_from_label_paths(label_paths: list[str], filetype = '.jpg') -> list[str]:
    '''
    This function converts label paths to the corresponding image paths

    Args:
        - label_paths. A list of label paths, referenced from the data_root folder.
        - filetype. The file type of the labled images.

    File operations:

    Returns:
        - The path to all images having manual annotations, referenced from the data_root.
    '''
    return [('\\').join(f.split('\\')[0:1]) + '\\images\\' + f.split('\\')[-1].split('.')[0] + filetype for f in label_paths]

def get_salmon_dir(labelme_label: dict, bodyparts: list[str], max_salmon_num = 500, IW = 4242, IH = 4242):
    '''
    This function calculates the direction of a salmon.

    Parameters:
        - labelme_label (dict): A label on the labelme format
        - bodyparts. The bodyparts of the salmon.
        - max_salmon_num: Sets the size of datastructures in the code. Must be larger than the largest fish id.
        - IW: Width of labled images
        - IH: Height of labled images

    Assumptions:
        - The boudning box x values determines the anterior extents of the salmon body parts
        - Head is anterior to all fins
        - Dorsal fin is anterior to adipose fin and tail fin
        - Adipose fin is anterior to tail fin
        - Pectoral fin is anterior to the pelvic, anal and tail fin
        - Pelvic fin is anterior to the anal and tail fin

    Returns:
        - A dictionary where the keys are salmon IDs, and the direction is one of r, l and u
            - r: The salmon swims to the right in the frame
            - l: The salmon swims to the left in the frame
            - u: It is unknown whether the salmon swims to the right or to the left in the frame
    '''

    # Start with accumulating all bodypart centers in a dictionary
    group_IDs = {k:[[] for l in range(len(bodyparts))] for k in range(max_salmon_num)}
    ID_dir_dict = {k: 'u' for k in range(max_salmon_num)}
    for item in labelme_label['shapes']:
        if item['shape_type'] == 'rectangle' and (item['label'] in bodyparts):
            xywh = normxywh(xyxy2xywh(np.array(item['points']).flatten()), IW, IH)
            group_IDs[int(item['group_id'])][bodyparts.index(item['label'])] = [float(xywh[0]), float(xywh[1])]
        if 'description' in list(item.keys()) and item['description'] is not None and len(item['description']) > 0:
            ID_dir_dict[int(item['group_id'])] = item['description']

    for k, v in group_IDs.items():
        if ID_dir_dict[int(k)] != 'u':
            continue

        dir = 'u'

        # head and tail fin
        if len(group_IDs[int(k)][0])>0 and len(group_IDs[int(k)][3])>0:
            if group_IDs[int(k)][0][0] - group_IDs[int(k)][3][0] >= 0:
                dir = 'r'
            else:
                dir = 'l'


        # head and dorsal fin
        elif len(group_IDs[int(k)][0])>0 and len(group_IDs[int(k)][1])>0:
            if group_IDs[int(k)][0][0] - group_IDs[int(k)][1][0] >= 0:
                dir = 'r'
            else:
                dir = 'l'

        # head and adi fin
        elif len(group_IDs[int(k)][0])>0 and len(group_IDs[int(k)][2])>0:
            if group_IDs[int(k)][0][0] - group_IDs[int(k)][2][0] >= 0:
                dir = 'r'
            else:
                dir = 'l'
        

        # head and pec fin
        elif len(group_IDs[int(k)][0])>0 and len(group_IDs[int(k)][6])>0:
            if group_IDs[int(k)][0][0] - group_IDs[int(k)][6][0] >= 0:
                dir = 'r'
            else:
                dir = 'l'

        # head and pelv fin
        elif len(group_IDs[int(k)][0])>0 and len(group_IDs[int(k)][5])>0:
            if group_IDs[int(k)][0][0] - group_IDs[int(k)][5][0] >= 0:
                dir = 'r'
            else:
                dir = 'l'

        # head and anal fin
        elif len(group_IDs[int(k)][0])>0 and len(group_IDs[int(k)][4])>0:
            if group_IDs[int(k)][0][0] - group_IDs[int(k)][4][0] >= 0:
                dir = 'r'
            else:
                dir = 'l'
        
        # dorsal fin and adi fin
        elif len(group_IDs[int(k)][1])>0 and len(group_IDs[int(k)][2])>0:
            if group_IDs[int(k)][1][0] - group_IDs[int(k)][2][0] >= 0:
                dir = 'r'
            else:
                dir = 'l'

        # dorsal fin and tail fin
        elif len(group_IDs[int(k)][1])>0 and len(group_IDs[int(k)][3])>0:
            if group_IDs[int(k)][1][0] - group_IDs[int(k)][3][0] >= 0:
                dir = 'r'
            else:
                dir = 'l'

        # adi fin and tail fin
        elif len(group_IDs[int(k)][2])>0 and len(group_IDs[int(k)][3])>0:
            if group_IDs[int(k)][2][0] - group_IDs[int(k)][3][0] >= 0:
                dir = 'r'
            else:
                dir = 'l'

        # pec fin and pelv fin
        elif len(group_IDs[int(k)][6])>0 and len(group_IDs[int(k)][5])>0:
            if group_IDs[int(k)][6][0] - group_IDs[int(k)][5][0] >= 0:
                dir = 'r'
            else:
                dir = 'l'


        # pec fin and anal fin
        elif len(group_IDs[int(k)][6])>0 and len(group_IDs[int(k)][4])>0:
            if group_IDs[int(k)][6][0] - group_IDs[int(k)][4][0] >= 0:
                dir = 'r'
            else:
                dir = 'l'

        # pec fin and tail fin
        elif len(group_IDs[int(k)][6])>0 and len(group_IDs[int(k)][3])>0:
            if group_IDs[int(k)][6][0] - group_IDs[int(k)][3][0] >= 0:
                dir = 'r'
            else:
                dir = 'l'

        # pelv fin and anal fin
        elif len(group_IDs[int(k)][5])>0 and len(group_IDs[int(k)][4])>0:
            if group_IDs[int(k)][5][0] - group_IDs[int(k)][4][0] >= 0:
                dir = 'r'
            else:
                dir = 'l'

        # pelv fin and tail fin
        elif len(group_IDs[int(k)][5])>0 and len(group_IDs[int(k)][3])>0:
            if group_IDs[int(k)][5][0] - group_IDs[int(k)][3][0] >= 0:
                dir = 'r'
            else:
                dir = 'l'
        
        ID_dir_dict[int(k)] = dir
    return ID_dir_dict

def labelme2yolo(labelme_root: str, YOLO_root: str, task: str, bodyparts: list[str], validation_frames = 1, max_salmon_num = 500, additional_kps = []) -> None:
    '''
    A function that creates a file structure for YOLO training

    Args:
        - labelme_root (str): Root to the annotated labelme data
        - YOLO_root (str): Path to the root of where the YOLO data will be generated
        - task (str): The task the data will be constructed to solve.
            - keybox_detection
            - bounding_box_detection
        - validation_frames: Specifies the validation frames of the data.
            - If the type is int, the function will randomly select validation_frames number of validation frames
            - If the type is a list of strings, the function will use the specified frames as validation frames
        - max_salmon_num: Sets the size of datastructures in the code. Must be larger than the largest fish id.
        - additional_kps. Specify additional key points (jaws, eye)
    
    File operations:
        - Constructs a folder with YOLO data

    Returns:
        - None
    '''

    random.seed(142904)

    components = ['salmon'] + bodyparts
    if 'YOLO_data' in os.listdir(YOLO_root):
        shutil.rmtree(YOLO_root + '\\YOLO_data')

    # Make directories
    os.mkdir(YOLO_root + '\\YOLO_data')
    os.mkdir(YOLO_root + '\\YOLO_data\\train')
    os.mkdir(YOLO_root +'\\YOLO_data\\val')

    label_paths = get_paths_to_labelme(labelme_root)
    #img_paths = get_img_paths_from_label_paths(label_paths, filetype = dst_fileending)
    img_paths = get_paths_to_images(labelme_root)
    
    if type(validation_frames) == int:
        val_label_paths = random.choices(label_paths, k = validation_frames)
    else:
        val_label_paths = []
        for validation_frame in validation_frames:
            for label_path in label_paths:
                if validation_frame.split('\\')[-1].split('.')[0] == label_path.split('\\')[-1].split('.')[0]:
                    val_label_paths.append(label_path)


    generate_yaml_file(YOLO_root, task, bodyparts, additional_kps=additional_kps)

    for label_path, img_path in zip(label_paths, img_paths):
        target_file_path = '\\YOLO_data\\'
        if label_path in val_label_paths:
            target_file_path = target_file_path + 'val\\'
        else:
            target_file_path = target_file_path + 'train\\'
        target_label_path = target_file_path + label_path.split('\\')[-1].split('.')[0] + '.txt'
        target_img_path = target_file_path + label_path.split('\\')[-1].split('.')[0] + '.png'

        t = magic.from_file(labelme_root + img_path)
        if img_path.endswith('jpg'):
            IW, IH = t.split(',')[7].split('x')
        elif img_path.endswith('png'):
            IW, IH = t.split(',')[1].split('x')
        IW = int(IW)
        IH = int(IH)

        rows = []
        with open(labelme_root + label_path, 'r') as f:
            label = json.load(f)

        if task == 'bounding_box_detection':
            for item in label['shapes']:
                if item['label'] in components:
                    if item['shape_type'] == 'rectangle':
                        xywh = normxywh(xyxy2xywh(np.array(item['points']).flatten()), IW, IH)
                    rows.append([str(components.index(item['label']))] + list(np.array(xywh).astype(str)))
        elif task.startswith('keybox_detection'):
            '''
            idx 0 => salmon class
            idx [1, 2, 3, 4]  => xywh for salmon bbox
            idx [5, 6, 7]     => xyo for head_dorsal_anterior
            idx [8, 9, 10]    => xyo for head_ventral_posterior
            idx [11, 12, 13] => xyo for dfin_dorsal_anterior
            idx [14, 15, 16] => xyo for dfin_ventral_posterior
            idx [17, 18, 19] => xyo for adifin_dorsal_anterior
            idx [20, 21, 22] => xyo for adifin_ventral_posterior
            idx [23, 24, 25] => xyo for tailfin_dorsal_anterior
            idx [26, 27, 28] => xyo for tailfin_ventral_posterior
            idx [29, 30, 31] => xyo for analfin_dorsal_anterior
            idx [32, 33, 34] => xyo for analfin_ventral_posterior
            idx [35, 36, 37] => xyo for pelvfin_dorsal_anterior
            idx [38, 39, 40] => xyo for pelvfin_ventral_posterior
            idx [41, 42, 43] => xyo for pecfin_dorsal_anterior
            idx [44, 45, 46] => xyo for pecfin_ventral_posterior
            idx [47, 48, 49] => xyo for body_dorsal_anterior
            idx [50, 51, 52] => xyo for body_ventral_posterior
            '''

            keybox_estimation_array = np.zeros([max_salmon_num, 5 + len(bodyparts)*6 + len(additional_kps)*3], dtype = int)
            keybox_estimation_array[:,0] = -1
            keybox_estimation_array = keybox_estimation_array.astype(str)

            ID_dir_dict = get_salmon_dir(label, bodyparts, max_salmon_num = max_salmon_num, IW = IW, IH = IH)

            for item in label['shapes']:
                if item['shape_type'] == 'rectangle':
                    if item['label'] == 'salmon':
                        xywh = normxywh(xyxy2xywh(np.array(item['points']).flatten()), IW, IH)
                        keybox_estimation_array[int(item['group_id'])][0:5] =['0'] + [round(i, 6) for i in xywh]

                    elif item['label'] in bodyparts:
                        x1, y1, x2, y2 = normxyxy(xyxy2xyxy_ordered(np.array(item['points']).flatten()), IW, IH)
                        assert ID_dir_dict[int(item['group_id'])] != 'u', 'Direction not found for ID ' + str(item['group_id']) + ' in frame ' + label_path
                        if ID_dir_dict[int(item['group_id'])] == 'l' or ID_dir_dict[int(item['group_id'])] == 'n':
                            xyoxyo = [str(round(x1, 6)), str(round(y1, 6)), str(2), str(round(x2, 6)), str(round(y2, 6)), str(2)]
                        elif ID_dir_dict[int(item['group_id'])] == 'r':
                            xyoxyo = [str(round(x2, 6)), str(round(y1, 6)), str(2), str(round(x1, 6)), str(round(y2, 6)), str(2)]
                        keybox_estimation_array[int(item['group_id'])][5 + bodyparts.index(item['label'])*6:5 + bodyparts.index(item['label'])*6 + 6] = xyoxyo

                else:    
                    if item['label'] in additional_kps and 'eye' not in additional_kps:
                        x1, y1 = round(item['points'][0][0]/IW, 6), round(item['points'][0][1]/IH, 6)
                        xyo = [str(x1), str(y1), str(2)]
                        keybox_estimation_array[int(item['group_id'])][5 + len(bodyparts)*6 + additional_kps.index(item['label'])*3:5 + len(bodyparts)*6 + additional_kps.index(item['label'])*3 + 3] = xyo
                                
                    elif item['shape_type'] == 'line' and 'eye' in additional_kps:
                        xyxy = list(np.array(item['points']).flatten())
                        x1, y1 = round(xyxy[0]/IW, 6), round(xyxy[1]/IH, 6)
                        x2, y2 = round(xyxy[2]/IW, 6), round(xyxy[3]/IH, 6)
                        xyo_eye = [str(x1), str(y1), str(2)]
                        xyo_jaw = [str(x2), str(y2), str(2)]

                        if item['label'] == 'line_ujaw' and 'ujaw' in additional_kps:
                            keybox_estimation_array[int(item['group_id'])][5 + len(bodyparts)*6 + additional_kps.index('eye')*3:5 + len(bodyparts)*6 + additional_kps.index('eye')*3 + 3] = xyo_eye
                            keybox_estimation_array[int(item['group_id'])][5 + len(bodyparts)*6 + additional_kps.index('ujaw')*3:5 + len(bodyparts)*6 + additional_kps.index('ujaw')*3 + 3] = xyo_jaw
                        if item['label'] == 'line_ljaw' and 'ljaw' in additional_kps:
                            keybox_estimation_array[int(item['group_id'])][5 + len(bodyparts)*6 + additional_kps.index('ljaw')*3:5 + len(bodyparts)*6 + additional_kps.index('ljaw')*3 + 3] = xyo_jaw
                        continue

                    elif item['shape_type'] == 'point' and 'eye' in additional_kps:
                        x1, y1 = round(item['points'][0][0]/IW, 6), round(item['points'][0][1]/IH, 6)
                        xyo = [str(x1), str(y1), str(2)]
                        keybox_estimation_array[int(item['group_id'])][5 + len(bodyparts)*6 + additional_kps.index(item['label'])*3:5 + len(bodyparts)*6 + additional_kps.index(item['label'])*3 + 3] = xyo
                        continue
            rows = keybox_estimation_array.tolist()

        shutil.copy(labelme_root + img_path, YOLO_root + target_img_path)

        if len(rows) > 0:
            with open(YOLO_root + target_label_path, 'w') as f:
                for row in rows:
                    if row[0] == '-1':
                        continue
                    for elem in row:
                        f.write(elem + ' ')
                    f.write('\n')

