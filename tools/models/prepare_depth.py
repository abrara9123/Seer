#!/usr/bin/env python3
"""Prepare the small local depth specialist; values are uncalibrated estimates."""
import hashlib
import json
import os
from pathlib import Path
root = Path(__file__).resolve().parents[2]
models = root / 'models'
models.mkdir(exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(root / '.model-tools' / 'config'))
os.environ.setdefault('MPLCONFIGDIR', str(root / '.model-tools' / 'matplotlib'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.chdir(root)
from ultralytics import YOLO
import ultralytics
import onnx
import cv2
import numpy as np
model = YOLO('yolo26n-depth.pt')
exported = Path(model.export(format='onnx', imgsz=512, device='cpu', batch=1,
                             opset=12, simplify=False, dynamic=False))
graph = onnx.load(exported)
onnx.checker.check_model(graph)
net = cv2.dnn.readNetFromONNX(str(exported))
net.setInput(np.zeros((1, 3, 512, 512), dtype=np.float32))
output = net.forward()
if output.ndim != 4 or output.shape[:2] != (1, 1) or not np.isfinite(output).all() or not (output > 0).all():
    raise RuntimeError('OpenCV depth output failed validation.')
exported.replace(models / 'depth_estimator.onnx')
manifest = {'checkpoint': 'yolo26n-depth.pt', 'source': 'https://docs.ultralytics.com/tasks/depth',
            'ultralytics_version': ultralytics.__version__, 'input_shape': [1, 3, 512, 512],
            'output_shape': list(output.shape), 'opset': 12, 'calibrated': False,
            'sha256': hashlib.sha256((models / 'depth_estimator.onnx').read_bytes()).hexdigest()}
(models / 'depth_model.json').write_text(json.dumps(manifest, indent=2) + '\n')
(root / 'yolo26n-depth.pt').unlink(missing_ok=True)
print('Local depth specialist prepared and validated.')
