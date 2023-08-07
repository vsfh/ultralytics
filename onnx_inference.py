import cv2
import numpy as np

# img = cv2.imread("image.jpg", cv2.IMREAD_UNCHANGED)
# resized = cv2.resize(img, (640,640), interpolation = cv2.INTER_AREA).astype(np.float32)
# resized = resized.transpose((2, 0, 1)) # convert to ONNX model format
# resized = np.expand_dims(resized, axis=0)  # Add batch dimension

import onnxruntime as ort
ort_session = ort.InferenceSession("/home/gregory/code/github/ultralytics/runs/detect/train4/weights/last.onnx", providers=["CUDAExecutionProvider"])
# compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: resized}
# ort_outs = ort_session.run(None, ort_inputs)