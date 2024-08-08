

from TeamCode.src.interface import AbstractDigitizationModel, AbstractClassificationModel
from TeamCode.src.verify_environment import verify_environment
import helper_code as hc
import numpy as np
import os

import torch

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo
import mmcv
from mmdet.apis import init_detector, inference_detector
from helper_code import get_num_samples, get_signal_names, get_image_files
from scipy import interpolate
device = os.environ.get("EXPECTEDDEVICE", "unspecified")
if device == 'unspecified':
    print("Not running in a docker container")
    dev = "cpu"
elif device == 'cpu':
    print("Running with cpu")
    dev = "cpu"
elif device == 'gpu':
    cuda_avail = torch.cuda.is_available() 
    if not cuda_avail:
        raise RuntimeError("Expected CUDA to be avialable but torch.cuda.is_available() returned False") 
    dev = "cuda:0"


## helper functions
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_nan(signal):
    if np.isnan(signal[0]):
        signal[0] = 0
    if np.isnan(signal[-1]):
        signal[-1] = 0
    nans, x= nan_helper(signal)
    signal[nans]= np.interp(x(nans), x(~nans), signal[~nans])
    return signal

def upsample(signal, num_samples):
    x = np.arange(0, len(signal))
    f = interpolate.interp1d(x, signal)
    xnew = np.linspace(0, len(signal)-1, num_samples)
    signal = f(xnew)
    return signal    

def downsample(signal, num_samples):
    x = np.arange(0, len(signal))
    f = interpolate.interp1d(x, signal)
    xnew = np.linspace(0, len(signal)-1, num_samples)
    signal = f(xnew)
    return signal


def filter_boxes(pred_bboxes, pred_labels, pred_scores, pred_masks):
    """
    Filter out the boxes with low confidence score and remove the boxes with high IoU
    Args:
        pred_bboxes (np): list of bounding boxes
        pred_labels (np): list of labels
        pred_scores (np): list of confidence scores
        pred_masks (np): list of masks
    Returns:
        np: filtered bounding boxes
        np: filtered labels
        np: filtered confidence scores
        np: filtered masks
    """
    def bbox_iou(box1, box2):
        """
        Calculate the Intersection of Unions (IoUs) between bounding boxes.
        Args:
            box1 (list): bounding box formatted as [x1, y1, x2, y2]
            box2 (list): bounding box formatted as [x1, y1, x2, y2]
        Returns:
            float: IoU value
        """
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1
        b2_x1, b2_y1, b2_x2, b2_y2 = box2

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = max(b1_x1, b2_x1)
        inter_rect_y1 = max(b1_y1, b2_y1)
        inter_rect_x2 = min(b1_x2, b2_x2)
        inter_rect_y2 = min(b1_y2, b2_y2)

        # Intersection area
        inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * max(inter_rect_y2 - inter_rect_y1 + 1, 0)

        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area)

        return iou
    # # loop over bounding boxes, if find some boxes have iou > 0.5, filter out the one with lower score


    for i in range(len(pred_bboxes)):
        for j in range(i+1, len(pred_bboxes)):
            iou = bbox_iou(pred_bboxes[i], pred_bboxes[j])
            if iou > 0.3:
                if pred_scores[i] > pred_scores[j]:
                    pred_scores[j] = 0
                else:
                    pred_scores[i] = 0
    pred_bboxes = pred_bboxes[pred_scores > 0]
    pred_labels = pred_labels[pred_scores > 0]
    pred_masks = pred_masks[pred_scores > 0]
    pred_scores = pred_scores[pred_scores > 0]
    
    if len(pred_scores) > 13:
        # sort the scores and get the top 13
        indices = np.argsort(pred_scores)[::-1][:13]
        pred_bboxes = pred_bboxes[indices]
        pred_labels = pred_labels[indices]
        pred_masks = pred_masks[indices]
        pred_scores = pred_scores[indices]
    
    return pred_bboxes, pred_labels, pred_scores, pred_masks



def bboxes_matching(bboxes, masks):
    # bboxes = bboxes.detach().numpy()
    # masks = masks.detach().numpy()
    # bboxes = bboxes[:13]
    # masks = masks[:13]
    #sort the bboxes by the average of H and W
    bboxes_avg_H = list((bboxes[:,1]+bboxes[:,3])/2)
    bboxes_avg_W = list((bboxes[:,0]+bboxes[:,2])/2)
    bboxes_avg = np.array([bboxes_avg_H, bboxes_avg_W]).transpose()
    #first take the long leads and 3x4
    sortH_idx = bboxes_avg[:, 0].argsort()
    mask_last = masks[sortH_idx[12]]
    bbox_last = bboxes[sortH_idx[12]]
    bboxes_avg = bboxes_avg[sortH_idx[0:12]]
    masks = masks[sortH_idx[0:12]]
    bboxes = bboxes[sortH_idx[0:12]]
    
    #then sort by W
    sortW_idx = bboxes_avg[:, 1].argsort()
    bboxes_avg = bboxes_avg[sortW_idx]
    masks = masks[sortW_idx]
    bboxes = bboxes[sortW_idx]
    col1 = list(bboxes_avg[0:3, 0].argsort())
    col2 = list(bboxes_avg[3:6, 0].argsort()+3)
    col3 = list(bboxes_avg[6:9, 0].argsort()+6)
    col4 = list(bboxes_avg[9:12, 0].argsort()+9)
    idx = col1+col2+col3+col4
    masks = masks[idx]
    bboxes = bboxes[idx]
    bboxes = np.append(bboxes, bbox_last.reshape((1,-1)), axis=0)
    masks = np.append(masks, mask_last.reshape((1,mask_last.shape[0],mask_last.shape[1])), axis=0)
    return bboxes, masks


## helper functions end 

#format = [['I', 'aVR', 'V1', 'V4'], ['II', 'aVL', 'V2', 'V5'], ['III', 'aVF', 'V3', 'V6'], ['II']] # format is hardcoded for now
#format = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6']
#fullmode = 'II'

def crop_from_bbox( bbox, mask, mV_pixel):
    ## input: leadname, string, the name of the lead
    ## input: bbox, tensor of shape (4,), the bounding box of the lead
    ## input: mask, tensor of shape (H, W), the binary mask of the lead
    bbox = bbox.astype(int)
    ecg_segment = mask[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]

    weighting_column = np.linspace(start = (bbox[3] - bbox[1])*mV_pixel/2, stop=-1*(bbox[3] - bbox[1])*mV_pixel/2, num=ecg_segment.shape[0]) #start and end included
    weighting_column = weighting_column.reshape((ecg_segment.shape[0],-1))
    weighting_matrix = np.tile(weighting_column, (1,ecg_segment.shape[1]))
    # weighting_matrix = weighting_column.repeat(1,ecg_segment.shape[1])

    weighted_ecg_segment = np.multiply(weighting_matrix, ecg_segment)
    # signal = torch.zeros(ecg_segment.shape[1])
    # np.array([0.0]*ecg_segment.shape[1])
    
    denominator = np.sum(ecg_segment,axis = 0)
    idx = np.where(denominator >= 1)
    numerator = np.sum(weighted_ecg_segment,axis=0)
    signal = np.empty(denominator.shape)
    signal[:] = np.nan
    signal[idx] = np.divide(numerator[idx],denominator[idx])

    # idx = (denominator != 0)
    # signal[idx] = torch.div(numerator[idx],denominator[idx])

    return signal




def readOut(header_path, masks, bboxes, mV_pixel, format):
    with open(header_path, 'r') as f:
        input_header = f.read()

    # get_sampling_frequency(input_header)
    num_samples = get_num_samples(input_header)
    # leadnames_all = get_signal_names(input_header)
    bboxes, masks = bboxes_matching(bboxes, masks)

    # signals_dict = {}
    ## first do the full modes
     #[leadname[0] for leadname in format[3:] ]
    signals_np = np.empty(shape=(12, num_samples))

    for i in range(12):
        if i == 1:
            signal = crop_from_bbox( bboxes[12], masks[12], mV_pixel)
            # signal = signal.detach().numpy()
            signal = interpolate_nan(signal)
            signal = signal - np.mean(signal)
            signallen = num_samples
            if len(signal) < signallen:
                signal = upsample(signal, signallen)
            elif len(signal) > signallen:
                signal = downsample(signal, signallen)
            signals_np[i] = signal
        else:
            signal = crop_from_bbox( bboxes[i], masks[i], mV_pixel)
            # signal = signal.detach().numpy()
            signal = interpolate_nan(signal)
            signal = signal - np.mean(signal)
            signallen = num_samples // 4
            if len(signal) < signallen:
                signal = upsample(signal, signallen)
            elif len(signal) > signallen:
                signal = downsample(signal, signallen)
            start_idx = (num_samples // 4)* (i//3)
            end_idx = start_idx + (num_samples // 4)
            signals_np[i,start_idx:end_idx] = signal
    
    np.clip(signals_np, -3, 3, out=signals_np)
    # nans = np.where(np.isnan(signals_np))
    # print the signal min max without nans
    signals_np = np.nan_to_num(signals_np)
    
    print(header_path)
    print(f'min max: {np.min(signals_np)}, {np.max(signals_np)}')
    if signals_np.shape[1] > signals_np.shape[0]:
        return signals_np.transpose()
    else: 
        return signals_np






class VoidClassificationModel(AbstractClassificationModel):
    def __init__(self):
        pass

    @classmethod
    def from_folder(cls, model_folder, verbose):
        return cls()

    def train_model(self, data_folder, model_folder, verbose):
        pass

    def run_classification_model(self, record, signal, verbose):
        return None
    
class OurDigitizationModel(AbstractDigitizationModel):
    def __init__(self):
        verify_environment()
        work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work_dir')
        self.work_dir = work_dir
        self.config = os.path.join(work_dir, "maskrcnn_config.py")
        self.model = None


    @classmethod
    def from_folder(cls, model_folder, verbose):
         # Create an instance of the class
        instance = cls()

        # Construct checkpoint path based on the model_folder parameter
        checkpoint_file = os.path.join(instance.work_dir, 'last_checkpoint.pth')

        # Initialize the model using instance-specific variables
        instance.model = init_detector(instance.config, checkpoint_file, device=dev)

        if verbose:
            print(f"Model loaded from {checkpoint_file}")

        return instance

    def train_model(self, data_folder, model_folder, verbose):
        print("We did not implement training the digitization model from (Image, Signal) pairs. Since we need extra context information generated by ecg-toolkit for training.")
        pass
        # if verbose:
        #     print('Training the digitization model...')
        #     print('Finding the Challenge data...')

        # # Reduce the number of repeated compilations and improve
        # # training speed.
        # setup_cache_size_limit_of_dynamo()

        # # load config
        # base_dir = os.path.dirname(os.path.abspath(__file__))  # Path to the directory containing your_script.py
        # work_dir = os.path.join(base_dir, 'work_dir')
        # config_file_path = os.path.join(work_dir, 'maskrcnn_config.py')
        # cfg = Config.fromfile(config_file_path)


        # cfg.work_dir = model_folder
        # # "moody/official-phase-mins-eth/TeamCode/work_dir"
        
        # cfg.data_root = data_folder
        # assert os.path.exists(os.path.join(base_dir,'checkpoints')), f'ckpt_root is not found'
        # cfg.load_from = os.path.join(base_dir,'checkpoints', 'mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth')
        


        # # # enable automatic-mixed-precision training
        # # if args.amp is True:
        # #     cfg.optim_wrapper.type = 'AmpOptimWrapper'
        # #     cfg.optim_wrapper.loss_scale = 'dynamic'

        # # enable automatically scaling LR
        # # if args.auto_scale_lr:
        # #     if 'auto_scale_lr' in cfg and \
        # #             'enable' in cfg.auto_scale_lr and \
        # #             'base_batch_size' in cfg.auto_scale_lr:
        # #         cfg.auto_scale_lr.enable = True
        # #     else:
        # #         raise RuntimeError('Can not find "auto_scale_lr" or '
        # #                         '"auto_scale_lr.enable" or '
        # #                         '"auto_scale_lr.base_batch_size" in your'
        # #                         ' configuration file.')

        # # resume is determined in this priority: resume from > auto_resume
        # # if args.resume == 'auto':
        # #     cfg.resume = True
        # #     cfg.load_from = None
        # # elif args.resume is not None:
        # #     cfg.resume = True
        # #     cfg.load_from = args.resume

        # # build the runner from config
        # if 'runner_type' not in cfg:
        #     # build the default runner
        #     runner = Runner.from_cfg(cfg)
        # else:
        #     # build customized runner from the registry
        #     # if 'runner_type' is set in the cfg
        #     runner = RUNNERS.build(cfg)

        # # start training
        # runner.train()


    
    def run_digitization_model(self, record, verbose):
        # image = np.array(load_image(record)[0])/255.0
        
        # config=f'/config/mask-rcnn_r50-caffe_fpn_ms-poly-3x_ecg.py'
        # img_dir = '/scratch/hshang/DLECG_Data/data/00000/val/00900_lr-0.png'

        # load image paths
        path = os.path.split(record)[0]
        image_files = get_image_files(record)

        images = list()
        for image_file in image_files:
            image_file_path = os.path.join(path, image_file)
            if os.path.isfile(image_file_path):
                images.append(image_file_path)
                
        # assume there is only one image per record
        img = images[0]

        img = mmcv.imread(img,channel_order='rgb')
        result = inference_detector(self.model, img)
        result_dict = result.to_dict()
        pred = result_dict['pred_instances']
        bboxes = pred['bboxes'].to(torch.int)
        masks = pred['masks']
        
        mV_pixel = (25.4 *8.5*0.5)/(masks[0].shape[0]*5) #hardcoded for now
        header_path = hc.get_header_file(record)
        signal=readOut(header_path, masks, bboxes, mV_pixel, format)
        return signal
    
# class OurDigitizationModel(AbstractDigitizationModel):
#     def __init__(self):
#         pass

#     def train_model(self, data_folder, model_folder, verbose):
#         print("We did not implement training the digitization model from (Image, Signal) pairs. Since we need extra context information generated by ecg-toolkit for training.")
#         pass

#     @classmethod
#     def from_folder(cls, model_folder, verbose):
#         # load whatever is in the model folder of this git repo
#         # TODO
#         return cls()
    
#     def run_digitization_model(self, record, verbose):
#         header_file = hc.get_header_file(record)
#         header = hc.load_text(header_file)

#         num_samples = hc.get_num_samples(header)
#         num_signals = hc.get_num_signals(header)

#         images = hc.load_images(record)
#         # I think there can be more than one image per record

#         if verbose:
#             print(f'Running the digitization model on {record}, it has {len(images)} images.')

#         signal_collector = np.zeros((len(images), num_samples, num_signals))
#         for i, img in enumerate(images):
#             # img is a PIL object at this point

#             np_img = np.asarray(img)



#             if verbose:
#                 print(f'Image shape: {np_img.shape}')

#             seed = int(np.round(np.mean(img)))

#             this_signal = np.random.default_rng(seed=seed).uniform(low=-0.1, high=0.1, size=(num_samples, num_signals))
#             signal_collector[i, :, :] = this_signal
        
#         signal = np.mean(signal_collector, axis=0)
#         return signal
        