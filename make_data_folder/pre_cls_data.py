from tritoninferencer import TritonInferencer
import os
import cv2
import numpy as np
tinf = TritonInferencer("127.0.0.1:8001")

def preprocess(img, h = 224, w = 224):
    # letter box
    imh, imw = img.shape[:2]
    r = min(h / imh, w / imw)  # ratio of new/old
    h, w = round(imh * r), round(imw * r)  # resized image
    hs, ws = h, w
    top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
    im_out = np.full((h, w, 3), 0, dtype=img.dtype)
    im_out[top:top + h, left:left + w] = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    # hwc -> chw, 0-255 -> 0-1
    im_out = np.ascontiguousarray(im_out.transpose((2, 0, 1))[::-1]).astype(np.float32)  # HWC to CHW -> BGR to RGB -> contiguous
    im_out /= 255.0  # 0-255 to 0.0-1.0
    return im_out

def delete(cls):
    with open(f'./error/{cls}.txt', 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    for path in lines:
        img = cv2.imread(path)
        if img is None:
            continue
        cv2.imshow('img', img)
        res = cv2.waitKey(0)
        print(res)
        # break

        if res == 113:
            os.remove(path)
            print('Img deleted')
        while res == 119:
            img = cv2.imread(path)
            img_ = np.rot90(img, -1)
            cv2.imshow('img', img_)
            res = cv2.waitKey(0)
            cv2.imwrite(path, img_)
            if res == 113:
                os.remove(path)
                print('Img deleted')

def main(cls):
    img_folder = f'/mnt/e/data/classification/image_folder_04/val/{cls}/'

    with open(f'./error/{cls}.txt', 'w') as f:
        for img_name in os.listdir(img_folder):
            # img_name = '58533880103904757_347.jpg'
            img_path = os.path.join(img_folder, img_name)
            img = cv2.imread(img_path)
            # img = cv2.imread('/mnt/e/data/classification/toy/test/ceph.jpg')

            img_ = preprocess(img)
            pred = tinf.infer_sync('cls-yolov8', {'images': img_[None, ...]}, ['output'])
            output = pred['output'][0].copy()
            pose = output[:6]
            probs = output[6:]

            labels = np.argmax(probs)
            scores = np.max(probs)

            if labels != int(cls):
                f.writelines(img_path+'\n')
            else:
                rot, matrix = rot_from_matirx(labels, pose)
                if rot != 0:
                    f.writelines(img_path+'\n')
                img_ = np.rot90(img, -rot)
                cv2.imwrite(img_path, img_)


        f.close()

def calculate_angle(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    x_product = np.cross(vector1,vector2)
    magnitudes = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    angle = np.arccos(dot_product / magnitudes)*180/np.pi * abs(x_product) / x_product
    return angle

def ortho6d_to_rotation(ortho6d):
    x_raw = ortho6d[0:3]#batch*3
    y_raw = ortho6d[3:6]#batch*3
        
    x = x_raw / np.linalg.norm(x_raw) #batch*3
    z = np.cross(x,y_raw) #batch*3
    z = z / np.linalg.norm(z) #batch*3
    y = np.cross(z,x)#batch*3
    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[:, 0] = x
    rotation_matrix[:, 1] = y
    rotation_matrix[:, 2] = z

    return rotation_matrix
from scipy.spatial.transform import Rotation as R
def rot_from_matirx(pred_cls, ortho6d):
    
    matrix = ortho6d_to_rotation(ortho6d)
    y_axis = matrix[:,1]*-1
    diff_vertical_group = [3, 4]
    if pred_cls not in diff_vertical_group:
        angle = calculate_angle(y_axis[:2], np.array([0,-1]))

    elif pred_cls == 3:
        angle = calculate_angle(matrix[:2,2], np.array([0, 1]))
    else:
        angle = calculate_angle(matrix[:2,2], np.array([0, -1]))
    if abs(angle) // 60 > 0:
        rot = angle//90+angle%90//60
        rot_matrix = R.from_euler('xyz',[0,0,90*rot],degrees=True).as_matrix()
        matrix = rot_matrix @ matrix
        return rot, matrix
    else:
        return 0, matrix
    
if __name__=='__main__':
    cls_dict = {
        "侧位片":'00',
        "覆盖像":'01',
        "全景片":'02',
        "上颌合面像":'03',
        "下颌合面像":'04',
        "右45度微笑像":'05',
        "右45度像":'06',
        "右侧面微笑像":'07',
        "右侧面像":'08',
        "右侧咬合像":'09',
        "正面微笑像":'10',
        "正面像":'11',
        "正面咬合像":'12',
        "左45度微笑像":'13',
        "左45度像":'14',
        "左侧面微笑像":'15',
        "左侧面像":'16',
        "左侧咬合像":'17'
    }
    # for key, item in cls_dict.items():
    #     main(item)
    delete('00')

