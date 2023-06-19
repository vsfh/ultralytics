from ultralytics import YOLO
import os
from os.path import join as opj
import numpy as np
import torch
import onnxruntime

import cv2
PATH = '/mnt/e/data/classification/'
IMG_PATH = PATH+'image_folder_04/train'
project = {
            '18':['其他',0],
            '00':['侧位片',1],
            '01':['覆盖像',2],
            '02':['全景片',3],
            '03':['上颌合面像',4],
            '04':['下颌合面像',5],
            '09':['右侧咬合像',6],
            '12':['正面咬合像',7],
            '17':['左侧咬合像',8]
        }

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

def infer():
    model = onnxruntime.InferenceSession('/home/vsfh/code/gitee/ultralytics-choho/runs/custom/train5/weights/best.onnx',
                                        providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                'CPUExecutionProvider'])   

    for key in project.keys():
        img_folder = opj(IMG_PATH,key)
        img_cls = project[key][1]
        with open(f'./make_data_folder/error/{key}.txt', 'w+') as f:
            for img_name in os.listdir(img_folder):
                img_path = opj(img_folder, img_name)
                img = cv2.imread(img_path)
                img_ = preprocess(img)
                output = model.run([], {'images':img_[None]}) 
                probs = output[0][0][6:]
                labels = np.argmax(probs)
                if not labels==img_cls:
                    print(img_path)
                    f.writelines(img_path+'\n'
                                 )
        f.close()

def delete(cls):
    with open(f'./make_data_folder/error/{cls}.txt', 'r') as f:
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
if __name__=='__main__':
    delete('17')