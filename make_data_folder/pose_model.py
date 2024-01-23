

from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.utils import yaml_load
# from ultralytics import YOLO
import cv2
import sys
sys.path.append('.')
from cls.trainer import ClassificationTrainerNew
from yolo import YOLO


def train_new():
    model = YOLO('/home/gregory/code/ultralytics/make_data_folder/det.yaml')
    model.train()
    
def predict_new():
    import os
    import numpy as np
    model = YOLO('/mnt/e/wsl/code/ultralytics/make_data_folder/runs/classify/train7/weights/best.pt')
    path = '/mnt/e/data/classification/image_folder_04/val/03/'
    for name in os.listdir(path):
        res = model.predict(os.path.join(path, name))
        cls = res[0].probs.top1
        if cls:
            img = cv2.imread(os.path.join(path, name))
            img = np.rot90(img, k=cls)
            cv2.imwrite(os.path.join(path, name), img)
            print(name)
        # break
    

def export_new():
    model = YOLO('/home/gregory/code/ultralytics/make_data_folder/runs/detect/train2/weights/best.pt')
    model.export(format="onnx")

if __name__=='__main__':
    train_new()
