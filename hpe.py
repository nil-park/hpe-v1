import cv2
import math
import numpy as np
import tflite_runtime.interpreter as tflite

model = tflite.Interpreter(model_path="models/tracker_v1.tflite")
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

def box_to_square(box):
    """ expand box short axis to long that leads box to square shape.

    Parameters:
    box (list like): box notation is [x1 y1 x2 y2]
    
    Returns:
    list: square box.
    """
    w = box[2] - box[0]; h = box[3] - box[1]
    e = abs(0.5 * (w - h))
    if w > h: return [box[0], box[1] - e, box[2], box[3] + e]
    else: return [box[0] - e, box[1], box[2] + e, box[3]]

def extend_box(box, scale):
    """ scale width and height of a box. from center of it.

    Parameters:
    box (list like): box notation is [x1 y1 x2 y2]
    scale (float): scale
    
    Returns:
    list: scaled box.
    """
    w = box[2] - box[0]; h = box[3] - box[1]
    e = (scale - 1) * 0.5
    ew = e * w
    eh = e * h
    return [box[0] - ew, box[1] - eh, box[2] + ew, box[3] + eh]

def crop_and_resize(im, box, dst_size):
    """ crop ROI from an image. 

    Parameters:
    im (ndarray): source image.
    box (list like): box notation is [x1 y1 x2 y2]
    dst_size (tuple): ROI will be resized to dst_size
    
    Returns:
    ndarray: cropped image.
    """
    w = box[2] - box[0]; h = box[3] - box[1]
    sx = dst_size[0] / w; sy = dst_size[1] / h
    mat = np.array([[sx, 0, -box[0]*sx], [0, sy, -box[1]*sy]], dtype=np.float32)
    return cv2.warpAffine(im, mat, dst_size)

def decode(a, box):
    """ decode data got from ML model.

    Parameters:
    a (list like): ML model output.
    box (list like): ROI box. notation: [x1 y1 x2 y2] denomalized
    
    Returns:
    list: result data. denomalized.
    """
    w = box[2] - box[0]
    cx = a[0] * w + box[0]
    cy = a[1] * w + box[1]
    radius = math.exp(a[2]) * w / 2.0
    rx = (0.5 - a[4]) * math.pi
    ry = (0.5 - a[3]) * math.pi
    rz = (1 - a[5] * 2) * math.pi
    return [cx, cy, radius, rx, ry, rz, a[6], a[7]]


def run(im, boxes, roi_scale=1.2):
    """ estimate 3D head pose, translation, scale, scores
        returns (N, 8) ndarray notation:
        - cx: center x of head
        - cy: center y of head
        - radius: radius of head
        - rx: rotation x axis
        - ry: rotation y axis
        - rz: rotation z axis
        - score: head or not. desired threshold >= 0.7
        - multiplicity: more than one head in ROI. desired threshold < 0.3

    Parameters:
    im (ndarray): RGB image. desired aspect ratio is 1.333
    boxes (list like): denomalized box got from facial detector. box notation: [x1 y1 x2 y2]
    roi_scale (float): scale to pad. refer to extend_box()
    
    Returns:
    ndarray: (N, 8) result data. denomalized.
    ndarray: (N, 4) roi used to infer head pose.
    """
    results = []
    rois = []
    for b in boxes:
        b = box_to_square(b)
        b = extend_box(b, roi_scale)
        rois.append(b)
        cropped = crop_and_resize(im, b, (96, 96))
        cropped = np.expand_dims(cropped, axis=0).astype(np.float32)
        model.set_tensor(input_details[0]['index'], cropped)
        model.invoke()
        h = model.get_tensor(output_details[0]['index'])[0]
        results.append(decode(h, b))
    return np.array(results), np.array(rois)
