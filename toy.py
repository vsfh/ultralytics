from ultralytics import YOLO
import os
import os.path as osp
import numpy as np
import torch
import cv2
import shutil
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
    "左侧咬合像":'17',
    "其他":'18'
}
    

def export():
    model = YOLO('./runs/custom/train2/weights/best.pt')
    success = model.export(dynamic=True,format="onnx")

def train():
    model = YOLO('./ultralytics/models/v8/custom/yolov8m-cus.yaml')
    # model = YOLO('./runs/custom/train/weights/last.pt')
    # model._load('')
    model.train(data='/data/shenfeihong/classification/image_folder_04/')
    # results = model("/home/disk/github/ultralytics/data/example/C01002721169_profile.jpg") 
    # print(results)
    pass

def infer():
    model = YOLO('./runs/custom/train/weights/last.pt')
    for i in range(14):
        file_path = f'/data/shenfeihong/classification/image_folder_04/train/{i:0{2}}'
        dest_path = f'/home/disk/data/classification/error_2/{i:0{2}}'
        txt_path = f'/home/disk/data/classification/error_2/error_{i}.txt'
        # os.makedirs(dest_path, exist_ok=True)

        for file in os.listdir(file_path):
            pth = osp.join(file_path, file)
            res = model(pth)
            pred_cls = torch.argmax(res[0].probs)
            print(pred_cls, i)
            break
            # if not pred_cls==i:
            #     cv2.imshow('img', pth)
            #     key = cv2.waitKey(0)
                # if key==127:
                #     os.remove(pth)
                #     print('Img deleted')
    cv2.destroyAllWindows()

def delete():
    with open('/home/disk/data/classification/error/error_13.txt', 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    for path in lines:
        os.remove(path)
    
def file_copy(mode='train'):
    dest_path = f'/mnt/e/data/classification/image_folder_04/{mode}'
    src_path = '/mnt/e/data/classification/2023-05-04-fussen_classification'
    for name in cls_dict.keys():
        os.makedirs(os.path.join(dest_path, str(cls_dict[name])), exist_ok=True)
    img_list = os.listdir(src_path)
    if mode=='val':
        img_list = img_list[-500:]
    else:
        img_list = img_list[:-500]
    for dir in os.listdir(src_path):
        for file in os.listdir(os.path.join(src_path, dir)):
            if file.split('_')[0] in cls_dict.keys() and file.split('_')[1]=="正畸检查":
                dest_file = dest_path+'/'+str(cls_dict[file.split('_')[0]])+'/'+dir+'_'+file.split('_')[2]
                shutil.copy(os.path.join(src_path, dir, file), dest_file)

def copy_json():
    src_path = '/mnt/e/data/classification/label'
    inner_path = '/mnt/e/data/classification/label_inner'
    inner_cls = ['03','04','09','12','17']
    face_path = '/mnt/e/data/classification/label_face'
    face_cls = ['05','06','07','08','10','11','13','14','15','16']
    for cls in inner_cls:
        folder_path = f'/mnt/e/data/classification/image_folder_04/val/{cls}'
        for img_file in os.listdir(folder_path):
            file_name = img_file.replace('jpg','json')
            shutil.copy(os.path.join(src_path, file_name), os.path.join(inner_path, file_name))
    for cls in face_cls:
        folder_path = f'/mnt/e/data/classification/image_folder_04/val/{cls}'
        for img_file in os.listdir(folder_path):
            file_name = img_file.replace('jpg','json')
            shutil.copy(os.path.join(src_path, file_name), os.path.join(face_path, file_name))
def aug():
    from scipy import ndimage
    img = cv2.imread('/mnt/e/data/classification/toy/front.jpg')
    # Rotation and Scale
    R = np.eye(3, dtype=np.float32)
    a = 90
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = 1
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    img2 = ndimage.rotate(img, 45, reshape=False, cval=114)
    cv2.imwrite('/mnt/e/data/classification/toy/rot.jpg', img2)
    # cv2.imshow('1', img)
    # cv2.imshow('2', img2)
    # cv2.waitKey(0)
    
def read_json():
    import json
    with open('/mnt/e/data/classification/image_folder/train/05/58497756069232932_998.json', 'r') as f:
        a = json.load(f)
    print(a)
    
from natsort import natsorted
def chose_folder():
    path = '/mnt/e/data/classification/label_inner'
    path_list = natsorted(os.listdir(path))
    name = ''
    count=0
    with open('a.txt', 'w') as f:
        for idx, file in enumerate(path_list):
            folder_name = file.split('_')[0]
            if name != folder_name:
                count = 1
                name = folder_name
            else:
                count+=1
            if count==5:
                f.writelines(folder_name+'\n')
    f.close()
            
def copy_folder():
    with open('a.txt', 'r') as f:
        file_list = f.readlines()
    filename_list = [a.strip() for a in file_list]
    path = '/mnt/e/data/classification/2023-05-06-fussen-cls-data'
    folder_path_list = [osp.join(path, foldername) for foldername in filename_list]
    print(len(folder_path_list))
    i = 0
    for folder_path in folder_path_list:
        # if i < 250:
        #     continue
        if osp.exists(folder_path):
            i += 1

            os.makedirs(osp.join('/mnt/e/data/classification/five_inner',osp.basename(folder_path)), exist_ok=True)
            for file in os.listdir(folder_path):
                if '上颌合面像_正畸检查' in file:
                    shutil.copy(osp.join(folder_path, file), osp.join('/mnt/e/data/classification/five_inner',osp.basename(folder_path),'upper.jpg'))
                if '下颌合面像_正畸检查' in file:
                    shutil.copy(osp.join(folder_path, file), osp.join('/mnt/e/data/classification/five_inner',osp.basename(folder_path),'lower.jpg'))
                if '左侧咬合像_正畸检查' in file:
                    shutil.copy(osp.join(folder_path, file), osp.join('/mnt/e/data/classification/five_inner',osp.basename(folder_path),'left.jpg'))
                if '右侧咬合像_正畸检查' in file:
                    shutil.copy(osp.join(folder_path, file), osp.join('/mnt/e/data/classification/five_inner',osp.basename(folder_path),'right.jpg'))
                if '正面咬合像_正畸检查' in file:
                    shutil.copy(osp.join(folder_path, file), osp.join('/mnt/e/data/classification/five_inner',osp.basename(folder_path),'front.jpg'))
                if '全景片' in file:
                    shutil.copy(osp.join(folder_path, file), osp.join('/mnt/e/data/classification/five_inner',osp.basename(folder_path),'pano.jpg'))

            # break
    
if __name__=='__main__':
    copy_folder()
                