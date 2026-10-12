#!/usr/bin/env python3
"""One-time download/export of a verified 365-class detector; not used at runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.model-tools' / 'config'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.model-tools' / 'matplotlib'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=list('nsmlx'), default='n')
    parser.add_argument('--stage', action='store_true', help='Prepare a candidate for testing before replacement.')
    args = parser.parse_args()
    from ultralytics import YOLO
    import ultralytics
    import onnx

    checkpoint = f'yolo26{args.variant}-objv1-150.pt'
    model = YOLO(checkpoint)
    if len(model.names) != 365 or set(model.names) != set(range(365)):
        raise RuntimeError('Checkpoint must have exactly 365 categories in contiguous ID order.')
    exported = Path(model.export(format='onnx', imgsz=640, batch=1, device='cpu',
                                 opset=12, simplify=False, dynamic=False, nms=None))
    graph = onnx.load(exported)
    onnx.checker.check_model(graph)
    shape = [dimension.dim_value for dimension in graph.graph.output[0].type.tensor_type.shape.dim]
    if shape != [1, 369, 8400]:
        raise RuntimeError(f'Unexpected detector output: {shape}')
    # Verify the actual OpenCV runtime before replacing a working model.
    import cv2
    import numpy as np
    net = cv2.dnn.readNetFromONNX(str(exported))
    net.setInput(np.zeros((1, 3, 640, 640), dtype=np.float32))
    if list(net.forward().shape) != shape:
        raise RuntimeError('OpenCV could not run the exported detector correctly.')
    folder = ROOT / '.model-tools' / 'candidates' / args.variant if args.stage else ROOT / 'models'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'objects365_detector.onnx'
    if exported != output:
        exported.replace(output)
    labels = [str(model.names[index]).strip().lower() for index in range(365)]
    (folder / 'objects365.names').write_text('\n'.join(labels) + '\n')
    manifest = {
        'checkpoint': checkpoint,
        'source': 'https://docs.ultralytics.com/datasets/detect/objects365',
        'ultralytics_version': ultralytics.__version__,
        'input_shape': [1, 3, 640, 640], 'output_shape': shape,
        'class_count': len(labels), 'preprocessing': 'RGB, 1/255, letterbox padding 114',
        'opset': 12, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (folder / 'objects365_model.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not args.stage:
        # Only known superseded project assets are removed. The exported ONNX
        # file is the single active runtime model; raw export weights are not needed.
        obsolete = ['yolov8n.onnx', 'yolov8s.onnx']
        obsolete += [f'objects365_yolo26{variant}.onnx' for variant in 'nsmlx']
        obsolete += [f'yolo26{variant}-objv1-150.pt' for variant in 'nsmlx']
        for name in obsolete:
            (ROOT / name).unlink(missing_ok=True)
    print('Objects365 detector ready:', output, 'with', len(labels), 'categories.')


if __name__ == '__main__':
    main()
