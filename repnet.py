# Import SixDRepNet
from sixdrepnet import SixDRepNet
import cv2
from math import cos, sin
import numpy as np


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    """
    Prints the person's name and age.

    If the argument 'additional' is passed, then it is appended after the main info.

    Parameters
    ----------
    img : array
        Target image to be drawn on
    yaw : int
        yaw rotation
    pitch: int
        pitch rotation
    roll: int
        roll rotation
    tdx : int , optional
        shift on x axis
    tdy : int , optional
        shift on y axis
        
    Returns
    -------
    img : array
    """

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img

def main():
    # Create model
    # Weights are automatically downloaded
    model = SixDRepNet()
    import os 
    import os.path as osp
    path = '/mnt/e/data/classification/image_folder/train/13'
    path = '/mnt/e/data/classification/image_folder/train/15'

    # with open('/mnt/e/data/classification/face_pose.txt', 'w') as f:
    for file in os.listdir(path):
        img = cv2.imread(osp.join(path, file))

        pitch, yaw, roll = model.predict(img)
        img = draw_axis(img, yaw, pitch, roll)

        cv2.imshow("test_window", img)
        cv2.waitKey(0)
        # break
        #     f.write(file+' '+str(pitch)+' '+str(yaw)+' '+str(roll)+'\n')
        # f.close()

if __name__=="__main__":
    main()
    