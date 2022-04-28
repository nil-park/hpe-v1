import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

model = tflite.Interpreter(model_path="models/mobilenetv2_320x240_rev2_decoded.tflite")
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

def run(im):
    """ find face boxes. targeting aspect ratio is 1.333 (eg. 640x480)
        detection range is short. cascade image if long range needed.

    Parameters:
    im (ndarray): RGB image. desired aspect ratio is 1.333
    
    Returns:
    ndarray (shape=(N, 4), dtype=float32): head boxes. box notation: [x1 y1 x2 y2] denormalized.
    """
    sw = im.shape[1] / 640; sh = im.shape[0] / 480
    im = cv2.resize(im, (320, 240))
    im = np.expand_dims(im, axis=0).astype(np.float32)
    model.set_tensor(input_details[0]['index'], im)
    model.invoke()
    boxes = model.get_tensor(output_details[0]['index'])
    boxes = boxes[boxes[:,4] > 0.7]
    boxes = boxes[boxes[:,5] < 0.3]
    indices = cv2.dnn.NMSBoxes(boxes[:,:4], boxes[:,4], 0.5, 0.5)
    boxes = boxes[indices][:,:4] * [sw, sh, sw, sh]
    return boxes
