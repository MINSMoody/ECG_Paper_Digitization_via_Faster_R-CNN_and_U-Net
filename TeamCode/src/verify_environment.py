## Methods to verify that everything is installed as expected
import os

def verify_environment():
    # the following should be importable
    import cv2
    import torch
    from mmdet.apis import init_detector
    import mmcv
    import mmcv.ops

    # check if cuda is avaialble
    # torch.cuda.is_available() 
    device = os.environ.get("EXPECTEDDEVICE", "unspecified")
    if device == 'unspecified':
        print("Not running in a docker container")
    elif device == 'cpu':
        print("Running with cpu")
    elif device == 'gpu':
        cuda_avail = torch.cuda.is_available() 
        if not cuda_avail:
            raise RuntimeError("Expected CUDA to be avialable but torch.cuda.is_available() returned False") 