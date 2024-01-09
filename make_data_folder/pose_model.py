

from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.utils import yaml_load
# from ultralytics import YOLO
import cv2
import sys
sys.path.append('.')
from cls.trainer import ClassificationTrainerNew
from cls.yolo import YOLO

         
def train_cls():
    model_cfg_dict = yaml_model_load('/mnt/e/wsl/code/ultralytics/make_data_folder/cls.yaml')
    default_cfg_dict = yaml_load('/mnt/e/wsl/code/ultralytics/make_data_folder/cfg.yaml')
    
    trainer = ClassificationTrainerNew(default_cfg_dict)
    trainer.model = trainer.get_model(cfg=model_cfg_dict)
    trainer.train()
    pass

def train_new():
    model = YOLO('/mnt/e/wsl/code/ultralytics/make_data_folder/cls.yaml')
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
    

def export_cls():
    model = YOLO('/mnt/e/wsl/code/ultralytics/make_data_folder/runs/classify/train6/weights/last.pt')
    model.export(format="onnx")
    
def pred_cls():
    import onnxruntime
    import numpy as np
    import os
    def preprocess(img_path):
        img = pad_and_resize(cv2.imread(img_path),(640,640))
        input = np.ascontiguousarray(img[..., ::-1].transpose((2, 0, 1))) / 255
        # im = torch.from_numpy(im).float()
        return input.astype(np.float32), img
    img_dir = '/mnt/e/data/classification/image_folder_04/val/03/'
    onnx_path = '/mnt/e/wsl/code/ultralytics/make_data_folder/runs/classify/train6/weights/last.onnx'
    onnx_sess = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    
    for img_name in os.listdir(img_dir):
        # img_name = '58146373763750710_15950.jpg'
        img_path = os.path.join(img_dir, img_name)
        input, img = preprocess(img_path)
        res = onnx_sess.run([], {'images': input[None]})[0]
        print(res)
        cls = np.argmax(res)

        new_img = np.rot90(img, k=4-cls)
        cv2.imshow('sa', new_img)
        cv2.imshow('sb', img)
        cv2.waitKey(0)
        # break
    pass
    
if __name__=='__main__':
    predict_new()
