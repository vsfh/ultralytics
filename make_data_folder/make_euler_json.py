import os
from os.path import join as opj
import numpy as np
import json
import cv2
# from chohocv.models import YoloModel, KeypointsModel
# from sixdrepnet import SixDRepNet
from math import cos, sin
from scipy.spatial.transform import Rotation as R
import shutil

PATH = '/mnt/e/data/classification'
LABEL_DIR = PATH+'/new_label/network_res'
IMG_DIR = PATH+'/image_folder_04'
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
def make_inner_euler(mode='val'):
    cls_dict = {
        '03':'upper',
        '04':'lower',
        '09':'right',
        '17':'left',
        '12':'front'
    }
    id_dict = {
        'upper': [1,2],
        'right': [1,4],
        'left': [2,3],
        'lower': [3,4],
        'front': [1,2,3,4]
    }
    def calculate_angle(vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        magnitudes = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        angle = np.arccos(dot_product / magnitudes)*180/np.pi
        return angle
    train_folder = opj(PATH, 'image_folder_04',mode)
    case_folder = '/mnt/e/data/classification/2023-05-04-fussen_classification'

    for key, group_type in cls_dict.items():
        cls_folder = opj(train_folder, key)
        for img_file in os.listdir(cls_folder):
            # if os.path.isfile(opj(LABEL_PATH ,img_file.replace('jpg','json'))):
            #     continue
            case_id = img_file.split('_')[0]
            case_path = opj(case_folder, case_id)
            # case_path = PATH+'/2023-05-06-fussen-cls-data/58533235385257642'
            # print(case_path)
            try:
                with open(opj(case_path,'segmentation_2d.json'),'r') as f:
                    context = json.load(f)   
                if 'result' in context.keys():
                    img_context = context['result']['image'][group_type]
                else:
                    img_context = context['image'][group_type]
                roi = img_context['roi']
                if group_type in ['upper', 'lower']:
                    mid_x = (roi[0]+roi[2])/2
                    mid_y = (roi[1]+roi[3])/2
                    min_teeth_id = 1e5

                    for teeth_id, teeth in img_context['teeth'].items():
                        pts = teeth['points']
                        vector = np.array([0, 1])

                        cpoint = np.mean(np.array(pts),0).astype(int)
                        if min_teeth_id > int(teeth_id) and int(teeth_id)//10 in id_dict[group_type]:
                            min_teeth_id = int(teeth_id)
                            angle = 180 - calculate_angle(vector, np.array([mid_x-cpoint[0], mid_y-cpoint[1]]))
                    if group_type=='lower':
                        init_matrix = R.from_euler('xyz', [90,0,0], degrees=True).as_matrix()
                        rot_matrix = R.from_euler('xyz', [0,0,angle], degrees=True).as_matrix()
                    else:
                        init_matrix = R.from_euler('xyz', [-90,0,0], degrees=True).as_matrix()
                        rot_matrix = R.from_euler('xyz', [0,0,180-angle], degrees=True).as_matrix()
                    matrix = rot_matrix@init_matrix
                if group_type in ['right', 'left']:
                    close_teeth_id = None
                    close_dis = 1e5
                    mid_x = (roi[0]+roi[2])/2
                    for teeth_id, teeth in img_context['teeth'].items():
                        pts = teeth['points']
                        cpoint = np.mean(np.array(pts),0).astype(int)
                        if abs(cpoint[0]-mid_x)<close_dis:
                            close_teeth_id = teeth_id
                            close_dis = abs(cpoint[0]-mid_x)

                    lr_degree = 0
                    if group_type=='right':
                        lr_degree = -90/7*(int(close_teeth_id)%10-0.5)
                    elif group_type=='left':
                        lr_degree = 90/7*(int(close_teeth_id)%10-0.5)
                    matrix = R.from_euler('xyz', [0,lr_degree,0], degrees=True).as_matrix()
                if group_type=='front':
                    matrix = R.from_euler('xyz', [0,-90,0], degrees=True).as_matrix()

                context={
                    'img_path':opj(cls_folder, img_file),
                    'group_id': key,
                    'roi': roi,
                    'euler': [matrix.tolist()]
                }
                with open(opj(LABEL_PATH ,img_file.replace('jpg','json')), 'w') as f:
                    json.dump(context, f)
            except:
                print(cls_folder, img_file)
                # break


def make_certain_euler(mode='val', show=True):
    cls_dict = {
        '00':[0, -90, 0],
        '01':[0, -90, 0],
        '02':[0, 0, 0],
        '12':[0, 0, 0],
    }

    train_folder = opj(PATH,'image_folder',mode)
    for key, item in cls_dict.items():
        cls_folder = opj(train_folder, key)
        for img_file in os.listdir(cls_folder):
            img = cv2.imread(opj(cls_folder, img_file))
            matrix = R.from_euler('xyz', item, degrees=True).as_matrix()
            if show:
                img_ = draw_pose(img, matrix)
                cv2.imshow('img', img_)
                cv2.waitKey(0)
            context={
                'img_path': opj(cls_folder, img_file),
                'group_id': key,
                'euler': [matrix.tolist()],
            }
            with open(opj(LABEL_PATH ,img_file.replace('jpg','json')), 'w') as f:
                json.dump(context, f)
            # break

def draw_pose(img, pose_ori, tdx=None, tdy=None, size = 100):
    img = img.copy()
    height, width = img.shape[:2]
    if len(pose_ori.shape) == 3:
        pose = pose_ori[0].copy()
    else:
        pose = pose_ori.copy()
    pose[:,1] *= -1
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2


    # X-Axis pointing to the right. Drawn in red
    x_axis = pose[:, 0]
    x1 = size * x_axis[0] + tdx
    y1 = size * x_axis[1] + tdy

    # Y-Axis | Drawn in green
    #        v
    y_axis = pose[:, 1]
    x2 = size * y_axis[0] + tdx
    y2 = size * y_axis[1] + tdy

    # Z-Axis (out of the screen). Drawn in blue
    z_axis = pose[:, 2]
    x3 = size * z_axis[0] + tdx
    y3 = size * z_axis[1] + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(255,0,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(0,255,0),4)
    return img
from tqdm import tqdm
def vis_json(sub_folder):
    number = sub_folder[-2:]
    with open(f'/mnt/e/data/classification/{number}.txt', 'w') as err_f:
        for file_name in tqdm(os.listdir(f'/mnt/e/data/classification/image_folder_04/{sub_folder}')):
            try:
                file = f'{sub_folder}/{file_name}'
                label_path = opj(LABEL_DIR,sub_folder[-2:] ,file_name.replace('jpg', 'json'))
                img_path = opj(IMG_DIR ,file)
                with open(label_path, 'r') as f:
                    context = json.load(f)

                k = np.random.randint(-3,3)
                img = cv2.imread(img_path)
                # if (context['xyxy'][2]-context['xyxy'][0])*(context['xyxy'][3]-context['xyxy'][1])/(img.shape[0]*img.shape[1]) > 0.95:
                    # os.remove(img_path)  
                cv2.rectangle(img, (int(context['xyxy'][0]), int(context['xyxy'][1])), (int(context['xyxy'][2]), int(context['xyxy'][3])), (255,0,0), 2)
                img = cv2.resize(img, (640, int(img.shape[0]/img.shape[1]*640)), interpolation=cv2.INTER_LINEAR)
                # img = np.rot90(img, k)
                euler = np.array(context['euler'])[0]
                if np.abs(euler[1])<90 or np.abs(euler[1])>100:
                    err_f.write(file_name+'\n')
                # rot_matrix = R.from_euler('xyz', euler, degrees=True).as_matrix()
                # img_ = draw_pose(img, rot_matrix)
                # cv2.imshow('img', img_)
                # cv2.waitKey(0)
            except:
                print(file)
                continue
        err_f.close()
    return

def vis_error(sub_folder):
    # sub_folder = 'train/16'
    number = sub_folder[-2:]
    
    with open(f'/mnt/e/data/classification/{number}.txt', 'r') as err_f:
        file_name_list = err_f.readlines()
        for file_name in file_name_list:
            file_name = file_name.strip()
            file = f'{sub_folder}/{file_name}'
            label_path = opj(LABEL_DIR,sub_folder[-2:] ,file_name.replace('jpg', 'json'))
            img_path = opj(IMG_DIR ,file)
            with open(label_path, 'r') as f:
                context = json.load(f)

            img = cv2.imread(img_path)
            cv2.rectangle(img, (int(context['xyxy'][0]), int(context['xyxy'][1])), (int(context['xyxy'][2]), int(context['xyxy'][3])), (255,0,0), 2)
            img = cv2.resize(img, (640, int(img.shape[0]/img.shape[1]*640)), interpolation=cv2.INTER_LINEAR)

            euler = np.array(context['euler'])[0]

            rot_matrix = R.from_euler('xyz', euler, degrees=True).as_matrix()
            img_ = draw_pose(img, rot_matrix)
            cv2.imshow('img', img_)
            cv2.waitKey(0)
        err_f.close()
def make_face_euler(mode='val', points=True):
    YOLO_CLASS_NAMES = ['face', 'tmp', 'mouth', 'nose']
    backend = 'native'
    yolo_model = YoloModel('weights', 'face-detector', (512, 512), backend=backend)

    ori_model = SixDRepNet()

    ms_kps_model = KeypointsModel(
        'weights',
        'tmp-lapa_landmark', (256, 256), resize_ratio=0.9, )
    def predict_face_detec(path):
        img = cv2.imread(path)
        img = img[..., ::-1]

        objs = yolo_model.predict(
            img,
            class_names=YOLO_CLASS_NAMES,
            show=False,
            class_agnostic=False,
        )

        face_bbox = objs.get_bboxes('face', first_only=True, expand=.15)
        face = objs.get_bboxes('face', first_only=True, expand=.0)

        if face_bbox is None:
            print(path)

        return face_bbox

    cls_list = ['05','06','07','08','10','11','13','14','15','16']
    # cls_list = ['15']

    train_folder = opj(PATH,'image_folder_04',mode)
    for cls in cls_list:
        cls_folder = opj(train_folder, cls)
        for img_file in os.listdir(cls_folder):
            img_path = opj(cls_folder, img_file)
            face_bbox = predict_face_detec(img_path)
            if face_bbox is None:
                context={

                    'img_path':img_path,
                    'group_id': cls,
                    'euler': [R.from_euler('xyz', [0,0,0], degrees=True).as_matrix().tolist()],
                }
                with open(opj(LABEL_PATH ,img_file.replace('jpg','json')), 'w') as f:
                    json.dump(context, f)
                    continue
            x1, y1, x2, y2 = face_bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
            img = cv2.imread(img_path)
            kps = ms_kps_model.predict(img[..., ::-1], bbox=face_bbox, show=False)
            res = kps[15] - kps[51]
            rot = 0
            if abs(res[1]) > abs(res[0]) and res[1]<0:
                rot  = 2
            if abs(res[1]) < abs(res[0]):
                if res[0] > 0:
                    rot = 1
                else:
                    rot = -1
            def rotate_pred(pred, rot):
                if len(pred.shape) == 3:
                    pred = pred[0]
                rotation_matrix = R.from_euler('xyz', rot, degrees=True).as_matrix()

                new_pred = (rotation_matrix @ pred).copy()
                return new_pred
            
            img_ = np.rot90(img[y1:y2,x1:x2].copy(), -rot)
            pred = ori_model.predict(img_) 
            new_pred = rotate_pred(pred, [0,0,-90*rot])

            context={
                'img_path':img_path,
                'group_id': cls,
                'face_points': [[x1, y1], [x2, y2]],
                'euler': [new_pred.tolist()],
                'kps': {f'{i}': [kps[i].tolist()] for i in range(len(kps))}
            }

            with open(opj(LABEL_PATH ,img_file.replace('jpg','json')), 'w') as f:
                json.dump(context, f)
            # break

def file_copy(mode='val'):

    dest_path = PATH+f'/image_folder/{mode}'
    src_path = PATH+'/2023-05-06-fussen-cls-data'
    for name in cls_dict.keys():
        os.makedirs(os.path.join(dest_path, str(cls_dict[name])), exist_ok=True)
    img_list = os.listdir(src_path)
    if mode=='val':
        img_list = img_list[-100:]
    else:
        img_list = img_list[:-100]
    for dir in img_list:
        for file in os.listdir(os.path.join(src_path, dir)):
            if file.split('_')[0] in cls_dict.keys() and file.split('_')[1]=="正畸检查":
                dest_file = dest_path+'/'+str(cls_dict[file.split('_')[0]])+'/'+dir+'_'+file.split('_')[2]
                shutil.copy(os.path.join(src_path, dir, file), dest_file)
                
if __name__=='__main__':
    # file_copy('val')
    # file_copy('train')
    # make_certain_euler('train', False)
    # make_certain_euler('val', False)
    # make_inner_euler('train')
    # make_inner_euler('val')
    # make_face_euler('train')
    # make_inner_euler('val')
    ceph_face = ['07','08','15','16']
    face_45 = ['05','06','13','14']
    face_smile = ['10','11']
    for face in face_smile:
        vis_json(f'train/{face}')
