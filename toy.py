from ultralytics import YOLO
import os
import os.path as osp
import numpy as np
import torch
import cv2
import shutil
cls_dict = {
    "侧位片":0,
    "覆盖像":1,
    "全景片":2,
    "上颌合面像":3,
    "下颌合面像":4,
    "右45度微笑像":5,
    "右45度像":6,
    "右侧面微笑像":7,
    "右侧面像":8,
    "右侧咬合像":9,
    "正面微笑像":10,
    "正面像":11,
    "正面咬合像":12,
    "左侧咬合像":13
}
    
def toy():
    model = YOLO('/home/disk/github/ultralytics/runs/yolov8-cls.yaml')
    model.train(data='/home/disk/data/classification/image_folder')
    # results = model("/home/disk/github/ultralytics/data/example/C01002721169_profile.jpg") 
    # print(results)
    pass

def infer():
    model = YOLO('/home/disk/github/ultralytics/runs/classify/train2/weights/last.pt')
    for i in range(14):
        file_path = f'/home/disk/data/classification/image_folder/train/{i:0{2}}'
        dest_path = f'/home/disk/data/classification/error_2/{i:0{2}}'
        txt_path = f'/home/disk/data/classification/error_2/error_{i}.txt'
        os.makedirs(dest_path, exist_ok=True)
        with open(txt_path,'w') as f:
            for file in os.listdir(file_path):
                pth = osp.join(file_path, file)
                res = model(pth)
                pred_cls = torch.argmax(res[0].probs)
                if not pred_cls==i:
                    shutil.copy(pth, osp.join(dest_path, file))
                    f.write(pth+'\n')
            f.close()
        
def delete():
    with open('/home/disk/data/classification/error/error_13.txt', 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    for path in lines:
        os.remove(path)
        

    
def file_copy():
    dest_path = '/home/disk/data/classification/image_folder/val'
    src_path = '/home/disk/data/classification/2023-05-04-fussen_classification'
    for name in cls_dict.keys():
        os.makedirs(os.path.join(dest_path, str(cls_dict[name])), exist_ok=True)
    for dir in os.listdir(src_path)[:20]:
        for file in os.listdir(os.path.join(src_path, dir)):
            if file.split('_')[0] in cls_dict.keys():
                dest_file = dest_path+'/'+str(cls_dict[file.split('_')[0]])+'/'+dir+'.jpg'
                shutil.copy(os.path.join(src_path, dir, file), dest_file)
                
def aug():
    from scipy import ndimage
    img = cv2.imread('/home/disk/data/classification/image_folder/train/13/58437825598523069.jpg')
    # Rotation and Scale
    R = np.eye(3, dtype=np.float32)
    a = 90
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = 1
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    img2 = ndimage.rotate(img, 45, reshape=False, cval=114)
    cv2.imshow('1', img)
    cv2.imshow('2', img2)
    cv2.waitKey(0)
    
                
# infer()
infer()
                