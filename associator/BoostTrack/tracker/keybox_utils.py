def get_iou(bb1: dict, bb2: dict) -> float:
    """
    Copied from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def xyoxyo2xywh(xyoxyo: list[float]) -> list[float]:
    p1 = xyoxyo[0:3]
    p2 = xyoxyo[3:]
    minx = min(p1[0], p2[0])
    miny = min(p1[1], p2[1])
    maxx = max(p1[0], p2[0])
    maxy = max(p1[1], p2[1])
    w = maxx-minx
    h = maxy-miny
    return [minx + w/2, miny + h/2, w, h]
    
def xyoxyo2xyxy(xyoxyo: list[float]) -> list[float]:
    p1 = xyoxyo[0:3]
    p2 = xyoxyo[3:]
    minx = min(p1[0], p2[0])
    miny = min(p1[1], p2[1])
    maxx = max(p1[0], p2[0])
    maxy = max(p1[1], p2[1])
    return [minx, miny, maxx, maxy]

def xywh2xyxy(xywh: list[float]) -> list[float]:
    return [xywh[0] - xywh[2]/2, xywh[1] - xywh[3]/2, xywh[0] + xywh[2]/2, xywh[1] + xywh[3]/2]

def xyxy2xywh(xyxy: list[float]) -> list[float]:
    min_x = min(xyxy[0], xyxy[2])
    x_ext = abs(xyxy[0] - xyxy[2])
    x_center = min_x + x_ext/2
    min_y = min(xyxy[1], xyxy[3])
    y_ext = abs(xyxy[1] - xyxy[3])
    y_center = min_y + y_ext/2
    xywh = [x_center, y_center, x_ext, y_ext]
    return xywh

def point_inside_box(point: list[float], box: list[float]) -> bool:
    """
    Check whether a point is inside a box.

    Args:
        point (list[float]): A point on the format (x,y)
        box (list[float]): A box on the format (x,y,w,h)
    """
    return point[0] > box[0] - box[2]/2 and point[0] < box[0] + box[2]/2 and point[1] > box[1] - box[3]/2 and point[1] < box[1] + box[3]/2

def valid_xyoxyo(xyoxyo: list[float], config: dict) -> bool:
    _, _, w, h = xyoxyo2xywh(xyoxyo)
    return xyoxyo[2] > config['bp_conf'] and xyoxyo[5] > config['bp_conf'] and w > config['bp_loc'] and h > config['bp_loc']

def normxywh(xywh: list[float], IW: int, IH: int) -> list[float]:
    x, y, w, h = xywh
    return [x/IW, y/IH, w/IW, h/IH]

def normxyxy(xyxy: list[float], IW: int, IH: int) -> list[float]:
    x1, y1, x2, y2 = xyxy
    return [x1/IW, y1/IH, x2/IW, y2/IH]

def xyxy2xyxy_ordered(xyxy: list[float]) -> list[float]:
    return [min(xyxy[0], xyxy[2]), min(xyxy[1], xyxy[3]), max(xyxy[0], xyxy[2]), max(xyxy[1], xyxy[3])]

def xyxy2bblbbtwh(xyxy: list[float]) -> list[float]:
    return [min(float(xyxy[0]), float(xyxy[2])), min(float(xyxy[1]), float(xyxy[3])), abs(float(xyxy[0]) - float(xyxy[2])), abs(float(xyxy[1]) - float(xyxy[3]))]

def bblbbtwh2xywh(bblbbtwh: list[float]) -> list[float]:
    return [float(float(bblbbtwh[0]) + float(bblbbtwh[2])/2), float(float(bblbbtwh[1]) + float(bblbbtwh[3])/2), float(float(bblbbtwh[2])), float(float(bblbbtwh[3]))]