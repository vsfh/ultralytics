import cv2
import numpy as np
from ultralytics.yolo.data.augment import LetterBox
from make_data_folder.make_euler_json import draw_pose, R
import onnxruntime as ort
ort_session = ort.InferenceSession("/mnt/e/share/last.onnx", providers=["CUDAExecutionProvider"])

img = cv2.imread("/mnt/e/data/classification/else/error/1.jpg")
resized = np.stack([LetterBox(640, auto=False)(image=img)])
resized = resized[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
resized = np.ascontiguousarray(resized).astype(np.float32)  # contiguous
resized /= 255

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: resized}
ort_outs = ort_session.run(None, ort_inputs)
prediction = ort_outs[0]
nc = 19
nm = prediction.shape[1] - 4 - nc
conf_thres = 0.25
mi = 4+nc
xc = np.amax(prediction[:, 4:mi], 1) > conf_thres  # candidates
output = np.zeros((prediction.shape[0], 5))
for xi, x in enumerate(prediction):  # image index, image inference
    # Apply constraints
    # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
    x = x.transpose()[xc[xi]]  # confidence

    # If none remain process next image
    if not x.shape[0]:
        continue

    # Detections matrix nx6 (xyxy, conf, cls)
    # box, cls, mask = x.split((4, nc, nm), 1)
    box = x[:,:4]
    cls = x[:,4:4+nc]
    mask = x[:,4+nc:]
    
    # box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)

    conf = np.max(cls, 1, keepdims=True)
    j = np.argmax(cls, 1, keepdims=True)
    
    final_pred_index = np.argmax(conf)
    final_pred_cls = j[final_pred_index]
    final_pred_pose = mask[final_pred_index]
    img_ = draw_pose(img, R.from_quat(final_pred_pose).as_matrix())
    cv2.imshow('img', img_)
    cv2.waitKey(0)
    output[xi][0] = final_pred_cls
    output[xi][1:] = final_pred_pose

pass