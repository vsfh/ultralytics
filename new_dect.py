from ultralytics import YOLO
import os
import os.path as osp
import numpy as np
import torch
import cv2
import shutil



def train():
    model = YOLO('./ultralytics/models/v8/yolov8m.yaml')
    path = '/data/shenfeihong/classification/image_folder_04/'
    path = '/mnt/e/data/classification/image_folder_04'
    model.train(data=path)
    
def test():
    model = YOLO('runs/detect/train4/weights/last.pt')
    model.predict('58382554055126628_59054.jpg')

def load_cache():
    cache_path = "/data/shenfeihong/classification/network_res/00.cache"
    import gc
    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True  # load dict
    gc.enable()
    nf, nm, ne, nc, n = cache.pop('results')
    pass
if __name__=='__main__':
    train()