from ultralytics_ori import YOLO
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os.path as osp
import numpy as np
import torch
import cv2
import shutil



def train():
    model = YOLO('./ultralytics/models/v8/yolov8m.yaml')
    path = '/data/shenfeihong/classification/image_folder_04/'
    # path = '/mnt/e/data/classification/image_folder_04/'

    # model.train(data=path, device='1,2,3')
    model.train(data=path)
    
def test():
    path = 'runs/detect/train7/weights/last.pt'
    path = '/mnt/e/share/last.pt'
    model = YOLO(path)
    model.predict('/mnt/e/data/classification/else/error/1.jpg')
    
def export():
    path = '/home/gregory/code/ultralytics/runs/detect/train3/weights/best.pt'
    # path = '/mnt/e/share/last.pt'
    model = YOLO(path)
    # model.export(format='torchs1
    # cript')
    model.export(format='onnx')

def load_cache():
    cache_path = "/data/shenfeihong/classification/network_res/00.cache"
    import gc
    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True  # load dict
    gc.enable()
    nf, nm, ne, nc, n = cache.pop('results')
    pass

def check():
    from ultralytics_ori.yolo.utils.checks import check_font
    check_font('Arial.ttf')
    
if __name__=='__main__':
    train()