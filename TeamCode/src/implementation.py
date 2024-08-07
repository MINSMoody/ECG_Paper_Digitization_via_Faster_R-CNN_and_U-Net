
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
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.config = os.path.join(work_dir, "maskrcnn_config.py")

    @classmethod
    def from_folder(cls, model_folder, verbose):
        # load last_checkpoint.pth from work_dir
        # load config from work_dir
        config = os.path.join(model_folder, "maskrcnn_config.py")
        return cls(config)

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
        img = mmcv.imread(record,channel_order='rgb')
        checkpoint_file = os.path.join(self.work_dir, 'last_checkpoint.pth')
        model = init_detector(self.config, checkpoint_file, device='cpu')
        result = inference_detector(model, img)
        result_dict = result.to_dict()
        pred = result_dict['pred_instances']
        bboxes = pred['bboxes'].to(torch.int)
        masks = pred['masks']
        
        mV_pixel = 0.5 #read from printout
        signals_dict = {}

        for i in range(len(bboxes)):
            #bbox = bboxes[i].item()

            ecg_segment = masks[i][bboxes[i][1].item():bboxes[i][3].item()+1, bboxes[i][0].item():bboxes[i][2].item()+1]

            weighting_column = torch.linspace(start = (bboxes[i][3] - bboxes[i][1])*mV_pixel/2, end=-1*(bboxes[i][3] - bboxes[i][1])*mV_pixel/2, steps=ecg_segment.shape[0]) #start and end included
            weighting_column = weighting_column.reshape((ecg_segment.shape[0],-1))
            weighting_matrix = weighting_column.repeat(1,ecg_segment.shape[1])

            weighted_ecg_segment = torch.mul(weighting_matrix, ecg_segment)
            signal = torch.zeros(ecg_segment.shape[1])
            # np.array([0.0]*ecg_segment.shape[1])
            denominator = torch.sum(ecg_segment,axis = 0)
            numerator = torch.sum(weighted_ecg_segment,axis=0)
            mask = (denominator != 0)
            signal[mask] = torch.div(numerator[mask],denominator[mask])
            signals_dict[i] = {'signal':signal, 'left_pxl': round(bboxes[i][0].item()),'right_pxl':round(bboxes[i][2].item())}


        ## from here work with numpy array
        fullmode = True
        bboxes = bboxes.detach().numpy()

        fullmode_idx = np.argmax(bboxes[:,2] - bboxes[:,0])
        leftmost, rightmost = bboxes[fullmode_idx, [0,2]]
        bboxes = bboxes-leftmost
        rightmost = rightmost-leftmost

        signals_mat = np.empty((len(bboxes), rightmost+1))
        signals_mat[:] = np.nan
        for i in range(len(bboxes)):
            if bboxes[i][0]<0:
                start_idx = 0
                end_idx = bboxes[i,2]+abs(bboxes[i][0])
            else:
                start_idx = bboxes[i,0]
                end_idx = bboxes[i,2]
            signals_mat[i,start_idx:end_idx+1] = signals_dict[i]['signal'].detach().numpy()[:]

        return signals_mat
    
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
        