import cv2
import numpy as np
from hpe import run as hpe
from renderer import Renderer
from scipy.spatial.transform import Rotation as R
import argparse
import time
from PIL import Image

# 설정 불러오기
parser = argparse.ArgumentParser(description='Test realtime 3D head pose estimation')
parser.add_argument('--video', dest='videoFile', default='0',
                    help='video source file or webcam number')

args = parser.parse_args()

def drawTimes(im, ms):
    cv2.putText(im, ms, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

if args.videoFile.isdigit():
    cap = cv2.VideoCapture(int(args.videoFile))
else:
    cap = cv2.VideoCapture(args.videoFile)

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

render = Renderer((256, 256))


# initial box
if w > h:
    e = (w - h) / 2
    ibox = [e, 0, w - e, h]
else:
    e = (h - w) / 2
    ibox = [0, e, w, h - e]

box = ibox

while True:
    ret, frame = cap.read()
    t0 = time.time()
    y, roi = hpe(frame[...,::-1], [box])
    t1 = time.time()
    im = Image.fromarray(frame[...,::-1])
    a = y[0]
    if a[6] >= 0.7 and a[7] < 0.3: # found head
        cx = a[0]; cy = a[1]; r = a[2]; b = roi[0]
        s = 2.0 * r / (b[2] - b[0])
        m = R.from_euler('xyz', [a[3], a[4], a[5]], degrees=False).as_matrix() * s
        rendered = cv2.resize(render(m), (int(b[2] - b[0]), int(b[3] - b[1])))
        rendered[:,:,3] = (rendered[:,:,3] * 0.7).astype(np.uint8)
        rendered = Image.fromarray(rendered)
        im.paste(rendered, (int(b[0] + (b[2] + b[0]) * 0.5 - cx), int(b[1] + (b[3] + b[1]) * 0.5 - cy)), rendered)
        box = [cx - r, cy - r, cx + r, cy + r]
    else:
        box = ibox
    # drawTimes(frame, f"{t1 - t0:.3f} sec")
    cv2.imshow('Head Pose Estimation', np.array(im)[...,::-1])
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
