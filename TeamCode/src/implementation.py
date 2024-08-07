

from TeamCode.src.interface import AbstractDigitizationModel, AbstractClassificationModel
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

## helper functions
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_nan(signal):
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

def bboxes_matching(bboxes, masks):
    bboxes = bboxes.detach().numpy()
    masks = masks.detach().numpy()
    # bboxes = bboxes[:13]
    # masks = masks[:13]
    #sort the bboxes by the average of H and W
    bboxes_avg_H = list((bboxes[:,1]+bboxes[:,3])/2)
    bboxes_avg_W = list((bboxes[:,0]+bboxes[:,2])/2)
    bboxes_avg = np.array([bboxes_avg_H, bboxes_avg_W]).transpose()
    #first sort by H
    sortrow_idx = bboxes_avg[:, 0].argsort()
    masks = masks[sortrow_idx]
    bboxes = bboxes[sortrow_idx]
    #then sort by W
    bboxes_avg = bboxes_avg[sortrow_idx[0:12]]
    row1 = list(bboxes_avg[0:4, 1].argsort())
    row2 = list(bboxes_avg[4:8, 1].argsort()+4)
    row3 = list(bboxes_avg[8:12, 1].argsort()+8)
    idx = row1+row2+row3
    masks[0:12] = masks[idx]
    bboxes[0:12] = bboxes[idx]
    bboxes[4] = bboxes[12]
    masks[4] = masks[12]
    return bboxes[0:12], masks[0:12]

## helper functions end 

#format = [['I', 'aVR', 'V1', 'V4'], ['II', 'aVL', 'V2', 'V5'], ['III', 'aVF', 'V3', 'V6'], ['II']] # format is hardcoded for now
#format = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6']
#fullmode = 'II'

def crop_from_bbox( bbox, mask, mV_pixel):
    ## input: leadname, string, the name of the lead
    ## input: bbox, tensor of shape (4,), the bounding box of the lead
    ## input: mask, tensor of shape (H, W), the binary mask of the lead
    ecg_segment = torch.from_numpy(mask[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1])

    weighting_column = torch.linspace(start = (bbox[3] - bbox[1])*mV_pixel/2, end=-1*(bbox[3] - bbox[1])*mV_pixel/2, steps=ecg_segment.shape[0]) #start and end included
    weighting_column = weighting_column.reshape((ecg_segment.shape[0],-1))
    weighting_matrix = weighting_column.repeat(1,ecg_segment.shape[1])

    weighted_ecg_segment = torch.mul(weighting_matrix, ecg_segment)
    # signal = torch.zeros(ecg_segment.shape[1])
    # np.array([0.0]*ecg_segment.shape[1])
    denominator = torch.sum(ecg_segment,axis = 0)
    numerator = torch.sum(weighted_ecg_segment,axis=0)
    signal = torch.div(numerator,denominator)
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
        signal = crop_from_bbox( bboxes[i], masks[i], mV_pixel)
        signal = signal.detach().numpy()
        signal = interpolate_nan(signal)
        signal = signal - np.mean(signal)
        if i == 4:
            signallen = num_samples
        else:
            signallen = num_samples // 4
        if len(signal) < signallen:
            signal = upsample(signal, signallen)
        elif len(signal) > signallen:
            signal = downsample(signal, signallen)
        if i == 4:
            signals_np[i] = signal
        else:
            start_idx = (num_samples // 4)* (i%4)
            end_idx = start_idx + (num_samples // 4)
            signals_np[i,start_idx:end_idx] = signal
            np.where(signals_np[i] > 1, signals_np[i], 1)
            np.where(signals_np[i] < -1, signals_np[i], -1)
            # min_value = np.min(signals_np)
            # max_value = np.max(signals_np)

            # print(f"Min Value: {min_value}")
            # print(f"Max Value: {max_value}")
            # assert min_value >= -32.768 and max_value <= 32767, f"Signal values are out of range: {min_value} - {max_value}"
    return np.transpose(signals_np)
        



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
        work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work_dir')
        self.work_dir = work_dir
        self.config = os.path.join(work_dir, "maskrcnn_config.py")

    @classmethod
    def from_folder(cls, model_folder, verbose):
        # load last_checkpoint.pth from work_dir
        # load config from work_dir
        # config = os.path.join(model_folder, "maskrcnn_config.py")
        return cls()

    def train_model(self, data_folder, model_folder, verbose):
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
        print(type(img))
        img = mmcv.imread(img,channel_order='rgb')
        checkpoint_file = os.path.join(self.work_dir, 'last_checkpoint.pth')
        model = init_detector(self.config, checkpoint_file, device='cpu')
        result = inference_detector(model, img)
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
        