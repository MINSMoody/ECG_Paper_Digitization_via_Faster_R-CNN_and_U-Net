import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# Get the directory one level up
import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to the Python path
sys.path.insert(0, parent_dir)
from src.interface import AbstractDigitizationModel, AbstractClassificationModel

from src import helper_code as hc
import os
# import joblib
import pickle

import torch
from tqdm import tqdm

import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo
import mmcv
from mmdet.apis import init_detector, inference_detector



class MaskRCNNDigitizationModel(AbstractDigitizationModel):
    def __init__(self):
        pass

    @classmethod
    def from_folder(cls, model_folder, verbose):
        pass
        instance = cls()
        
        filename = os.path.join(model_folder, 'digitization_model.pkl')
        model_instance = load_pickle(filename)
        return model_instance

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
        # config_file_path = os.path.join(base_dir, 'config', 'mask-rcnn_r50-caffe_fpn_ms-poly-3x_ecg.py')
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
        
        config=f'/config/mask-rcnn_r50-caffe_fpn_ms-poly-3x_ecg.py'
        # img_dir = '/scratch/hshang/DLECG_Data/data/00000/val/00900_lr-0.png'
        img = mmcv.imread(record,channel_order='rgb')
        checkpoint_file = '/model/combo_epoch_12.pth'
        model = init_detector(config, checkpoint_file, device='cpu')
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
        
        return signal

class ExampleClassificationModel(AbstractClassificationModel):

    @classmethod
    def from_folder(cls, model_folder, verbose):
        pass
        # filename = os.path.join(model_folder, 'dx_model.pkl')
        # model_instance = load_pickle(filename)
        # return model_instance


    # Train your dx classification model.
    def train_model(self, data_folder, model_folder, verbose):
        pass
        # Find data files.
        # if verbose:
        #     print('Training the dx classification model...')
        #     print('Finding the Challenge data...')

        # records = hc.find_records(data_folder)
        # num_records = len(records)

        # if num_records == 0:
        #     raise FileNotFoundError('No data was provided.')

        # # Extract the features and labels.
        # if verbose:
        #     print('Extracting features and labels from the data...')

        # features = list()
        # dxs = list()

        # for i in range(num_records):
        #     if verbose:
        #         width = len(str(num_records))
        #         print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        #     record = os.path.join(data_folder, records[i])

        #     # Extract the features from the image, but only if the image has one or more dx classes.
        #     dx = hc.load_dx(record)
        #     if dx:
        #         current_features = extract_features(record)
        #         features.append(current_features)
        #         dxs.append(dx)

        # if not dxs:
        #     raise Exception('There are no labels for the data.')

        # features = np.vstack(features)
        # self.classes = sorted(set.union(*map(set, dxs)))
        # dxs = hc.compute_one_hot_encoding(dxs, self.classes)

        # # Train the model.
        # if verbose:
        #     print('Training the model on the data...')

        # # Define parameters for random forest classifier and regressor.
        # n_estimators   = 12  # Number of trees in the forest.
        # max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
        # random_state   = 56  # Random state; set for reproducibility.

        # # Fit the model.
        # self.model = RandomForestClassifier(
        #     n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, dxs)

        # # Create a folder for the model if it does not already exist.
        # os.makedirs(model_folder, exist_ok=True)

        # # Save the model.
        # path = os.path.join(model_folder, 'dx_model.pkl')
        # save_pickle(self, path)

        # if verbose:
        #     print('Done.')
        #     print()

    def run_classification_model(self, record, signal, verbose):
        model = self.model
        classes = self.classes

        # Extract features.
        features = extract_features(record)
        features = features.reshape(1, -1)

        # Get model probabilities.
        probabilities = model.predict_proba(features)
        probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

        # Choose the class(es) with the highest probability as the label(s).
        max_probability = np.nanmax(probabilities)
        labels = [classes[i] for i, probability in enumerate(probabilities) if probability == max_probability]

        return labels


# # Save your trained digitization model.
# def save_digitization_model(model_folder, model):
#     d = {'model': model}
#     filename = os.path.join(model_folder, 'digitization_model.sav')
#     joblib.dump(d, filename, protocol=0)

# # Save your trained dx classification model.
# def save_dx_model(model_folder, model, classes):
#     d = {'model': model, 'classes': classes}
#     filename = os.path.join(model_folder, 'dx_model.sav')
#     joblib.dump(d, filename, protocol=0)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):  
    with open(path, 'rb') as f:
        return pickle.load(f)



################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features(record):
    images = hc.load_images(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])

