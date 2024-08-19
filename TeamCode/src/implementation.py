

from TeamCode.src import our_paths
from TeamCode.src.interface import AbstractDigitizationModel, AbstractClassificationModel
from TeamCode.src.verify_environment import verify_environment
import helper_code as hc
import numpy as np
import os
import cv2
import pickle

# from mmseg.registry import DATASETS
# from mmseg.datasets import BaseSegDataset
# # from mmseg.apis import MMSegInferencer

# # from mmdet.apis import DetInferencer
# from mmseg.apis import inference_model, init_model

from concurrent.futures import ThreadPoolExecutor

# import threading
import multiprocessing

import torch
import warnings

from TeamCode.src.ecg_predict import ECGPredictor
from TeamCode.src.ecg_main import ECGSegment

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo
import mmcv
from mmdet.apis import init_detector, inference_detector
from helper_code import get_num_samples, get_signal_names, get_image_files
from scipy import interpolate

import random
from TeamCode.src.ecg_image_generator.helper_functions import find_records
from TeamCode.src.ecg_image_generator.gen_ecg_image_from_data import run_single_file
import warnings
import json
from argparse import Namespace

import os.path as osp
import numpy as np
from mmcv import imread, imwrite
import mmengine
from tqdm.auto import tqdm


from pycocotools import mask as maskutils
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings("ignore")

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

# def interpolate_nan(signal):
#     if np.isnan(signal[0]):
#         signal[0] = 0
#     if np.isnan(signal[-1]):
#         signal[-1] = 0
#     nans, x = nan_helper(signal)
#     if len(nans) == 0:
#         return signal
#     signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
#     return signal

from scipy.interpolate import CubicSpline

def interpolate_nan(signal):
    nans, x = np.isnan(signal), lambda z: z.nonzero()[0]
    if np.isnan(signal).all():
        warnings.warn("Signal is all nan", UserWarning)
        return np.zeros_like(signal)
    signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
    # For sharper interpolation:
    cs = CubicSpline(np.arange(len(signal)), signal, bc_type='natural')
    return cs(np.arange(len(signal)))

from scipy.signal import resample

def upsample(signal, target_length):
    return resample(signal, target_length)

def downsample(signal, target_length):
    return resample(signal, target_length)

def dist(tuple1, tuple2):
    return np.sqrt((tuple1[0]-tuple2[0])**2 + (tuple1[1]-tuple2[1])**2)

def group_bboxes_row(bboxes, vertical_distance_threshold):
    # bboxes_avg_H = list((bboxes[:,1]+bboxes[:,3])/2)
    # bboxes_avg_W = list((bboxes[:,0]+bboxes[:,2])/2)
    # bboxes_avg = bboxes[:,4:6]
    row_dict = {}
    for i in range(bboxes.shape[0]):
        h, w = bboxes[i,4:6]
    #for h, w  in bboxes_avg:
        if not bool(row_dict):
            row_dict[(h,w)] = [bboxes[i]]
        else:
            paired = False
            for key in row_dict.keys():
                if abs(key[0] - h) < vertical_distance_threshold:
                    row_dict[key].append(bboxes[i])
                    paired = True
                    break
            if not paired:
                row_dict[(h,w)] = [bboxes[i]]
    row_dict_new = {}
    row_dict_numcol = {}
    i = 0
    for key in row_dict.keys():
        row_dict_new[i]  = np.stack(row_dict[key], axis=0)
        row_dict_numcol[i] = len(row_dict[key])
        i += 1
    return row_dict_new, row_dict_numcol





def group_bboxes(bboxes, vertical_distance_threshold, horizontal_distance_threshold):
    bboxes_avg_H = list((bboxes[:,1]+bboxes[:,3])/2)
    bboxes_avg_W = list((bboxes[:,0]+bboxes[:,2])/2)
    bboxes_avg = np.array([bboxes_avg_H, bboxes_avg_W]).transpose()
    # sort by W
    row_dict = {}
    for h, w  in bboxes_avg:
        if not bool(row_dict):
            row_dict[(h,w)] = [(h, w)]
        else:
            paired = False
            for key in row_dict.keys():
                if abs(key[0] - h) < vertical_distance_threshold:
                    row_dict[key].append((h,w))
                    paired = True
                    break
            if not paired:
                row_dict[(h,w)] = [(h,w)]
    col_dict = {}
    for h, w in bboxes_avg:
        if not bool(col_dict):
            col_dict[(h,w)] = [(h, w)]
        else:
            paired = False
            for key in col_dict.keys():
                if abs(key[1] - w) < horizontal_distance_threshold:
                    col_dict[key].append((h,w))
                    paired = True
                    break
            if not paired:
                col_dict[(h,w)] = [(h,w)]
    return row_dict, col_dict


def filter_boxes(pred_bboxes, pred_labels, pred_scores, pred_masks):
    """
    Filter out the boxes with low confidence score and remove the boxes with high IoU
    Args:
        pred_bboxes (np): list of bounding boxes
        pred_labels (np): list of labels
        pred_scores (np): list of confidence scores
        pred_masks (np): list of maskss
    Returns:
        np: filtered bounding boxes
        np: filtered labels
        np: filtered confidence scores
        np: filtered masks
    """
    def bbox_intersection_over_smaller_area(box1, box2):
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

        # boxes Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        
        return inter_area / min(b1_area, b2_area)
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


    if len(pred_bboxes) >= 13:
        for i in range(len(pred_bboxes)):
            for j in range(i+1, len(pred_bboxes)):
                iou = bbox_intersection_over_smaller_area(pred_bboxes[i], pred_bboxes[j])
                if iou > 0.4:
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

from scipy.signal import savgol_filter

def apply_savgol_filter(signal):
    return savgol_filter(signal, window_length=5, polyorder=3)  # Adjust parameters as needed

def bboxes_sorting(bboxes, img_width): # input: bboxes output by maskrcnn
    # print(bboxes)
    # first sort by H
    bboxes_avg_H = (bboxes[:,1]+bboxes[:,3])/2
    bboxes_avg_W = (bboxes[:,0]+bboxes[:,2])/2
    bboxes = np.append(bboxes, bboxes_avg_H.reshape((-1,1)), axis=1)
    bboxes = np.append(bboxes, bboxes_avg_W.reshape((-1,1)), axis=1)
    sortH_idx = bboxes[:, 4].argsort()
    bboxes = bboxes[sortH_idx]

    row_dict, row_dict_numcol = group_bboxes_row(bboxes, vertical_distance_threshold = img_width/10)
    if len(row_dict) == 4:
        ncol = max(list(row_dict_numcol.values()))
        if ncol == 4:
            print('row=4, col=4')
            ## step 1, adjust the bboxes
            rowswith4cols = [k for k,v in row_dict_numcol.items() if int(v) == 4]           
            left = min([np.min(row_dict[row][:,0]) for row in rowswith4cols])
            right = max([np.max(row_dict[row][:,2]) for row in rowswith4cols])
            leadwidth = (right - left)/4
            leftmid = left + leadwidth
            mid = left + 2*leadwidth
            rightmid = left + 3*leadwidth
            for key in [0,1,2]:
                if row_dict_numcol[key] == 4:
                    row_dict[key] = row_dict[key][row_dict[key][:,5].argsort()]
                    row_dict[key][0,0] = left
                    row_dict[key][0,2] = leftmid
                    row_dict[key][1,0] = leftmid
                    row_dict[key][1,2] = mid
                    row_dict[key][2,0] = mid
                    row_dict[key][2,2] = rightmid
                    row_dict[key][3,0] = rightmid
                    row_dict[key][3,2] = right
                else:
                    new_bboxes = np.nan*np.ones((4,4))
                    for i in range(row_dict[key].shape[0]):
                        h, w = row_dict[key][i,4:6]
                        if w < leftmid:
                            new_bboxes[0] = [left, row_dict[key][i,1], leftmid, row_dict[key][i,3]]
                        elif w < mid:
                            new_bboxes[1] = [leftmid, row_dict[key][i,1], mid, row_dict[key][i,3]]
                        elif w < rightmid:
                            new_bboxes[2] = [mid, row_dict[key][i,1], rightmid, row_dict[key][i,3]]
                        else:
                            new_bboxes[3] = [rightmid, row_dict[key][i,1], right, row_dict[key][i,3]]
                    row_dict[key] = new_bboxes
                    
            row_dict[3][0,0] = row_dict[3][0,0] if row_dict[3][0,0] < left else left
            row_dict[3][0,2] = row_dict[3][0,2] if row_dict[3][0,2] > right else right
            bboxes = np.concatenate([row_dict[0][:,0:4], row_dict[1][:,0:4], row_dict[2][:,0:4], row_dict[3][:,0:4]], axis=0)
            standard_format = [0, 12, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
            bboxes = bboxes[standard_format]
            # print(bboxes)
            return bboxes, 4
    if len(row_dict) == 3:
        ncol = max(list(row_dict_numcol.values()))
        if ncol == 4:
            print('row=3, col=4')
            rowswith4cols = [k for k,v in row_dict_numcol.items() if int(v) == 4]           
            left = min([np.min(row_dict[row][:,0]) for row in rowswith4cols])
            right = max([np.max(row_dict[row][:,2]) for row in rowswith4cols])
            leadwidth = (right - left)/4
            leftmid = left + leadwidth
            mid = left + 2*leadwidth
            rightmid = left + 3*leadwidth
            for key in [0,1,2]:
                if row_dict_numcol[key] == 4:
                    row_dict[key] = row_dict[key][row_dict[key][:,5].argsort()]
                    row_dict[key][0,0] = left
                    row_dict[key][0,2] = leftmid
                    row_dict[key][1,0] = leftmid
                    row_dict[key][1,2] = mid
                    row_dict[key][2,0] = mid
                    row_dict[key][2,2] = rightmid
                    row_dict[key][3,0] = rightmid
                    row_dict[key][3,2] = right
                else:
                    new_bboxes = np.nan*np.ones((4,4))
                    for i in range(row_dict[key].shape[0]):
                        h, w = row_dict[key][i,4:6]
                        if w < leftmid:
                            new_bboxes[0] = [left, row_dict[key][i,1], leftmid, row_dict[key][i,3]]
                        elif w < mid:
                            new_bboxes[1] = [leftmid, row_dict[key][i,1], mid, row_dict[key][i,3]]
                        elif w < rightmid:
                            new_bboxes[2] = [mid, row_dict[key][i,1], rightmid, row_dict[key][i,3]]
                        else:
                            new_bboxes[3] = [rightmid, row_dict[key][i,1], right, row_dict[key][i,3]]
                    # bottom_max = np.max(row_dict[key][:,4])
                    # top_min = np.min(row_dict[key][:,1])
                    # new_bboxes = np.array([[left, top_min, leftmid, bottom_max], [leftmid, top_min, mid, bottom_max], [mid, top_min, rightmid, bottom_max], [rightmid, top_min, right, bottom_max]])
                    # row_dict[key] = new_bboxes
            bboxes = np.concatenate([row_dict[0][:,0:4], row_dict[1][:,0:4], row_dict[2][:,0:4]], axis=0)
            standard_format = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
            bboxes = bboxes[standard_format]
            # print(bboxes)
            return bboxes, 3
    print('none!')
    return None, None



## helper functions end 

#format = [['I', 'aVR', 'V1', 'V4'], ['II', 'aVL', 'V2', 'V5'], ['III', 'aVF', 'V3', 'V6'], ['II']] # format is hardcoded for now
#format = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6']
#fullmode = 'II'
import matplotlib.pyplot as plt
def crop_from_bbox(bbox, mask, mV_pixel):
    bbox = bbox.astype(int)
    # draw bbox on to the mask and save it
    # Assuming mask and bbox are defined
    # mask = (mask * 255).astype(np.uint8)

    # # Convert grayscale mask to RGB to draw a colored rectangle
    # mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # # Draw rectangle on the RGB image
    # cv2.rectangle(mask_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

    # Display the image
    # plt.imshow(mask_rgb)
    # plt.title(f'bbox: {bbox}')
    # plt.savefig('bbox.png')
    # plt.show()
    ecg_segment = mask[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
    if np.sum(ecg_segment, axis=1).all() == 0:
        warnings.warn("All empty in ecg segment, wrong bbox", UserWarning)
        # raise ValueError("All empty in ecg segment, wrong bbox")
        # return np.zeros(ecg_segment.shape[1])

    weighting_matrix = np.linspace((bbox[3] - bbox[1])*mV_pixel/2, -1*(bbox[3] - bbox[1])*mV_pixel/2, num=ecg_segment.shape[0]).reshape(-1, 1)
    weighted_ecg_segment = ecg_segment * weighting_matrix

    denominator = np.sum(ecg_segment, axis=0)
    numerator = np.sum(weighted_ecg_segment, axis=0)

    signal = np.full(denominator.shape, np.nan)
    valid_idx = denominator >= 1
    signal[valid_idx] = numerator[valid_idx] / denominator[valid_idx]
    # check if signal is all nan
    if np.isnan(signal).all():
        warnings.warn("Signal is all nan", UserWarning)

    return signal

# median method
# def crop_from_bbox(bbox, mask, mV_pixel):
#     bbox = bbox.astype(int)
#     ecg_segment = mask[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]

#     height = ecg_segment.shape[0]

#     a = -1*mV_pixel*height/(height-1)
#     b = mV_pixel*height/2
#     signal = [a*np.median(np.argwhere(ecg_segment[:,i]))+b if len(np.argwhere(ecg_segment[:,i]))>1 else np.nan  for i in range(ecg_segment.shape[1])]
#     signal = np.array(signal)
#     return signal

import pywt
import numpy as np

def wavelet_denoising(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(denoised_coeffs, wavelet)

def readOut(num_samples, masks, nrows, bboxes, mV_pixel):
    
    num_signals = 12
    signals_np = np.full((num_signals, num_samples), np.nan)

    def process_signal(index, bboxes, masks, mV_pixel, signallen):
        signal = crop_from_bbox(bboxes[index], masks[index], mV_pixel)
        signal = interpolate_nan(signal) - np.mean(signal)
        
        # signal = np.clip(signal, -2, 2)
        try :
            signal = apply_savgol_filter(signal)
        except:
            print('Error in savgol filter')
        signal = wavelet_denoising(signal)
        if len(signal) < signallen:
            signal = upsample(signal, signallen)
        else:
            signal = downsample(signal, signallen)
        return signal

    if nrows == 4:
        for i in range(num_signals):
            signallen = num_samples if i == 1 else num_samples // 4
            start_idx = (num_samples // 4) * (i // 3)
            end_idx = start_idx + signallen
            if i < len(bboxes):
                signal = process_signal(i, bboxes, masks, mV_pixel, signallen)
                signals_np[i, start_idx:end_idx] = signal
            else:
                signals_np[i, start_idx:end_idx] = np.zeros(signallen)
    elif nrows == 3:
        for i in range(bboxes.shape[0]):
            signallen = num_samples // 4
            start_idx = (num_samples // 4) * (i // 3)
            end_idx = start_idx + signallen
            signal = process_signal(i, bboxes, masks, mV_pixel, signallen)
            signals_np[i, start_idx:end_idx] = signal
    
    signals_np = np.clip(signals_np, -2, 2)
    
    return signals_np.T if signals_np.shape[1] > signals_np.shape[0] else signals_np
    

    
    
        
        
    
    # # case 1: less than 12 boxes, return empty signals
    # # assert bboxes.shape[0] == masks.shape[0], f"Expected shape {bboxes.shape[0]}, got {masks.shape[0]}"
    # if bboxes.shape[0] < 12:
    #     # failed to detect 12 leads
    #     empty_signals_np = np.full((12, num_samples), np.nan)
    #     lead_length = num_samples // 4
    #     empty_signals_np[0:3,0:lead_length] = 0
    #     empty_signals_np[3:6,lead_length:2*lead_length] = 0
    #     empty_signals_np[6:9,2*lead_length:3*lead_length] = 0
    #     empty_signals_np[9:12,3*lead_length:4*lead_length] = 0
    #     return empty_signals_np.T if empty_signals_np.shape[1] > empty_signals_np.shape[0] else empty_signals_np
        
    # # case 2: 12 bboxes, filter boxes
    # if bboxes.shape[0] == 12:
    #     bboxes, masks = bboxes_sorting_12(bboxes, masks)
    #     signals_np = np.full((12, num_samples), np.nan)
    #     for i in range(bboxes.shape[0]):
    #         signal = crop_from_bbox(bboxes[i], masks[i], mV_pixel)
    #         signal = interpolate_nan(signal) - np.mean(signal)
    #         signal = np.clip(signal, -2, 2)
    #         signallen = num_samples // 4
    #         # signal = apply_savgol_filter(signal)
    #         signal = upsample(signal, signallen) if len(signal) < signallen else downsample(signal, signallen)
    #         start_idx = (num_samples // 4) * (i // 3)
    #         end_idx = start_idx + (num_samples // 4)
    #         signals_np[i, start_idx:end_idx] = signal
    #     signals_np = np.clip(signals_np, -2, 2)
    #     return signals_np.T if signals_np.shape[1] > signals_np.shape[0] else signals_np



    # # case 3: at least 13 bboxes
    # if bboxes.shape[0] >= 13:
    #     bboxes, masks = bboxes_sorting_13(bboxes, masks)
    #     signals_np = np.full((12, num_samples), np.nan)

    #     for i in range(12):
    #         signal = crop_from_bbox(bboxes[i], masks[i], mV_pixel)   
    #         if np.isnan(signal).all():
    #             warnings.warn(f"Signal {i} is all nan", UserWarning)
    #             signal = np.zeros_like(signal)
    #         signal = interpolate_nan(signal) - np.mean(signal)
    #         signal = np.clip(signal, -2, 2)
    #         signallen = num_samples if i == 1 else num_samples // 4
    #         if len(signal) < 5:
    #             signal = np.zeros(signallen)
    #         signal = apply_savgol_filter(signal)
    #         signal = upsample(signal, signallen) if len(signal) < signallen else downsample(signal, signallen)
            
    #         if i == 1:
    #             signals_np[i] = signal
    #         else:
    #             start_idx = (num_samples // 4) * (i // 3)
    #             end_idx = start_idx + (num_samples // 4)
    #             signals_np[i, start_idx:end_idx] = signal

        # return signals_np.T if signals_np.shape[1] > signals_np.shape[0] else signals_np




def process_single_file(full_header_file, full_recording_file, args, original_output_dir):
    print(f"Processing {full_header_file}")
    filename = full_recording_file
    header = full_header_file
    args.input_file = os.path.join(args.input_directory, filename)
    args.header_file = os.path.join(args.input_directory, header)
    args.start_index = -1

    folder_struct_list = full_header_file.split('/')[:-1]
    args.output_directory = os.path.join(original_output_dir, '/'.join(folder_struct_list))
    args.encoding = os.path.split(os.path.splitext(filename)[0])[1]

    return run_single_file(args)

def generate_data(data_folder, model_folder, data_amount, verbose):
    with open(os.path.join(model_folder, 'data_format.json'), 'r') as f:
        args_dict = json.load(f)
    args = Namespace(**args_dict)
    random.seed(args.seed)
    args.input_directory = data_folder
    args.output_directory = os.path.join(data_folder, "training_data")
    os.makedirs(args.output_directory, exist_ok=True)
    args.max_num_images = data_amount
    if not os.path.isabs(args.input_directory):
        args.input_directory = os.path.normpath(os.path.join(os.getcwd(), args.input_directory))
    if not os.path.isabs(args.output_directory):
        original_output_dir = os.path.normpath(os.path.join(os.getcwd(), args.output_directory))
    else:
        original_output_dir = args.output_directory

    if not os.path.exists(args.input_directory) or not os.path.isdir(args.input_directory):
        raise Exception("The input directory does not exist, Please re-check the input arguments!")

    if not os.path.exists(original_output_dir):
        os.makedirs(original_output_dir)

    i = 0
    full_header_files, full_recording_files = find_records(args.input_directory, original_output_dir)

    with ThreadPoolExecutor() as executor:
        futures = []
        for full_header_file, full_recording_file in zip(full_header_files, full_recording_files):
            future = executor.submit(process_single_file, full_header_file, full_recording_file, args, original_output_dir)
            futures.append(future)

        for future in futures:
            try:
                i += future.result()
            except Exception as e:
                if verbose:
                    print(f"Error processing file: {e}")
            if args.max_num_images != -1 and i >= args.max_num_images:
                break

def prepare_data_for_training(data_folder, verbose=False):

    def binary_mask_to_rle_np(binary_mask):
        # Create a copy of the original mask
        thickened_mask = np.copy(binary_mask)

        # Use numpy slicing to mark pixels above and below the current mask
        # thickened_mask[:-1, :] |= binary_mask[1:, :]  # Mark the pixel above
        # thickened_mask[1:, :] |= binary_mask[:-1, :]  # Mark the pixel below
        binary_mask = np.asfortranarray(thickened_mask.astype(np.uint8))
        rle = {"counts": [], "size": list(binary_mask.shape)}
        area = np.sum(binary_mask)
        flattened_mask = binary_mask.ravel(order="F")
        diff_arr = np.diff(flattened_mask)
        nonzero_indices = np.where(diff_arr != 0)[0] + 1
        lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

        # note that the odd counts are always the numbers of zeros
        if flattened_mask[0] == 1:
            lengths = np.concatenate(([0], lengths))

        rle["counts"] = lengths.tolist()

        return rle, area

    def binary_mask_to_rle(binary_lead_mask):
        """
        Converts a mask to COCO's RLE format and calculates the area.

        Parameters:
        - lead_mask: 2D numpy array of the mask.
        - threshold: The threshold to binarize the mask.

        Returns:
        - rle: Dictionary representing the run-length encoding in COCO format.
        - area: Integer representing the area of the mask.
        """


        # Step 2: Encode the binary mask using maskutils
        rle = maskutils.encode(np.asfortranarray(binary_lead_mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')
        # Step 3: Calculate the area
        area = maskutils.area(rle)

        return rle, area


    def convert_ecg_to_coco(data_path, bbox_dir, mask_dir, out_file):
        # bbox_dir = osp.join(data_path, 'lead_bounding_box')
        # mask_dir = osp.join(data_path, 'masks')  # Path to mask directory
        annotations = []
        images = []
        obj_count = 0
        img_crop_dir = osp.join(data_path, 'cropped_img')
        os.makedirs(img_crop_dir, exist_ok=True)
        mask_crop_dir = osp.join(data_path, 'cropped_masks')
        os.makedirs(mask_crop_dir, exist_ok=True)
        
        for entry in tqdm(os.listdir(data_path), desc='Processing file'):
            if entry.endswith(".png"):
                img_name = entry[:-4]
                bbox_path = osp.join(bbox_dir, f'{img_name}.json')
                mask_path = osp.join(mask_dir, f'{img_name}.png')  
                
                
                
                mask = imread(mask_path, flag='grayscale')
                image = imread(osp.join(data_path, entry), flag='unchanged')
                
                if mask is None:
                    print(f"Failed to read mask for {mask_path}")
                    continue

                images.append(dict(
                    id=int(entry[:5]),
                    file_name=entry,
                    height=image.shape[0],
                    width=image.shape[1]))
                
                with open(bbox_path, 'r') as file:
                    settings = json.load(file)
                    
                for lead in settings['leads']:
                    coords = lead['lead_bounding_box']
                    x_coords = [coord[1] for coord in coords.values()]
                    y_coords = [coord[0] for coord in coords.values()]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                
                    # Extract the sub-region from the mask
                    lead_mask = mask[y_min:y_max, x_min:x_max]
                    # prepare the lead crop for segmentation model
                    lead_crop = image[y_min:y_max, x_min:x_max]
                    
                    save_crop = np.random.choice([True, False], p=[0.1, 0.9])
                    
                    if save_crop:
                        imwrite(lead_crop, osp.join(img_crop_dir, f'{obj_count}.png'))
                        imwrite(lead_mask, osp.join(mask_crop_dir, f'{obj_count}.png'))
                    
                    

                    # Apply threshold to create a binary mask
                    binary_mask = lead_mask > 0
                    # plt.imshow(binary_mask)

                    # Prepare a full-size binary mask for visualization
                    full_binary_mask = np.zeros_like(mask, dtype=bool)
                    full_binary_mask[y_min:y_max, x_min:x_max] = binary_mask
                    
                    
                    rle, area = binary_mask_to_rle_np(full_binary_mask)

                    
                    data_anno = dict(
                        image_id=int(entry[:5]),
                        id=obj_count,
                        category_id=0,
                        bbox=[x_min, y_min, x_max - x_min, y_max-y_min],
                        area=area,
                        iscrowd=0,
                        segmentation=rle
                    )
                    
                    annotations.append(data_anno)
                    obj_count += 1

        coco_format_json = dict(
            images=images,
            annotations=annotations,
            categories=[{'id': 0, 'name': 'ecg_lead'}])
        mmengine.dump(coco_format_json, out_file)
    convert_ecg_to_coco(
    data_folder,
    data_folder,
    os.path.join(data_folder,'masks'),
    os.path.join(data_folder, 'annotation_coco.json'))

def remove_image_gradients(image_array):
   # Step 2: Split the image into R, G, B channels
    if image_array.shape[2] == 4:
        b_channel, g_channel, r_channel, a_channel = cv2.split(image_array)
    elif image_array.shape[2] == 3:
        b_channel, g_channel, r_channel = cv2.split(image_array)
        a_channel = np.ones_like(b_channel) * 255
    else: return image_array

    # Step 3: Apply Laplacian filter to each channel
    laplacian_b = cv2.Laplacian(b_channel, cv2.CV_64F)
    laplacian_g = cv2.Laplacian(g_channel, cv2.CV_64F)
    laplacian_r = cv2.Laplacian(r_channel, cv2.CV_64F)

    # Convert the result back to uint8 (8-bit image) because Laplacian can result in negative values
    laplacian_b = cv2.convertScaleAbs(laplacian_b)
    laplacian_g = cv2.convertScaleAbs(laplacian_g)
    laplacian_r = cv2.convertScaleAbs(laplacian_r)

    # Step 4: Merge the channels back together
    laplacian_rgb = cv2.merge((laplacian_b, laplacian_g, laplacian_r))
    return laplacian_rgb

class OurDigitizationModel(AbstractDigitizationModel):
    def __init__(self):
        verify_environment()
        self.config = None#os.path.join(work_dir, "maskrcnn_res101.py")
        self.model = None
        self.unet = None
        self.mmseg = None


    @classmethod
    def from_folder(cls, model_folder, verbose):
         # Create an instance of the class
        instance = cls()
        # instance.work_dir = model_folder
        instance.config = os.path.join(model_folder, "maskrcnn_res101.py")
        # Construct checkpoint path based on the model_folder parameter
        maskrcnn_checkpoint_log = os.path.join(model_folder, 'last_checkpoint')
        with open(maskrcnn_checkpoint_log, 'r') as f:
            maskrcnn_checkpoint_file = os.path.join(model_folder, f.read().strip())

        # Initialize the model using instance-specific variables
        # load model parameters from json file
        with open(os.path.join(model_folder, 'ecg_params.json'), 'r') as f:
            ecg_params = json.load(f)['segmentation']
        instance.model = init_detector(instance.config, maskrcnn_checkpoint_file, device=dev)
        instance.unet = ECGPredictor('resunet10', os.path.join(model_folder,'segmentation/segmentation_model.pth'), size=ecg_params['crop'], cbam=ecg_params['cbam'])
        # instance.mmseg = init_model(config='/scratch/hshang/moody/mmsegmentation_MINS/demo/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive-ecg.py', checkpoint='/scratch/hshang/moody/mmsegmentation_MINS/demo/work_dirs/ECG/iter_400.pth', device=dev)
        if verbose:
            print(f"Model loaded from {maskrcnn_checkpoint_file}")

        return instance

    def train_detection_model(self, data_folder, model_folder, verbose):
        # load config
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Path to the directory containing your_script.py
        # work_dir = os.path.join(base_dir, 'work_dir')
        config_file_path = os.path.join(model_folder, 'maskrcnn_res101.py')
        cfg = Config.fromfile(config_file_path)
        cfg.metainfo = {
            'classes': ('ecg_lead', ),
            'palette': [
                (220, 20, 60),
            ]
        }
        cfg.data_root = data_folder
        cfg.train_dataloader.dataset.ann_file = 'annotation_coco.json'
        cfg.train_dataloader.dataset.data_root = cfg.data_root
        cfg.train_dataloader.dataset.data_prefix.img = ''
        cfg.train_dataloader.dataset.metainfo = cfg.metainfo

        # cfg.val_dataloader.dataset.ann_file = 'val/annotation_coco.json'
        # cfg.val_dataloader.dataset.data_root = cfg.data_root
        # cfg.val_dataloader.dataset.data_prefix.img = 'val/'
        # cfg.val_dataloader.dataset.metainfo = cfg.metainfo
        
        cfg.val_cfg = None
        cfg.val_dataloader = None

        # cfg.test_dataloader = cfg.val_dataloader

        # Modify metric config
        # cfg.val_evaluator.ann_file = cfg.data_root+'/'+'val/annotation_coco.json'
        cfg.val_evaluator = None
        # cfg.test_evaluator = cfg.val_evaluator
        
        cfg.work_dir = os.path.join(model_folder, 'maskrcnn_res101.py')
        cfg.data_root = data_folder
        # assert os.path.exists(os.path.join(base_dir,'checkpoints')), f'ckpt_root is not found'
        cfg.load_from = os.path.join(base_dir,'checkpoints', 'mask_rcnn_r101_caffe_fpn_1x_coco_20200601_095758-805e06c1.pth')

        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

        if verbose:
            print("Training detection model...")
        
        cfg.work_dir = model_folder
        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)
            
        # start training
        runner.train()
        
        if verbose:
            print("Detection model training completed.")

    def train_segmentation_model(self, data_folder, model_folder, verbose):
        if verbose:
            print("Training segmentation model...")
        
        param_file = os.path.join(model_folder, 'ecg_params.json')
        param_set = "segmentation"
        unet_data_dir = os.path.join(data_folder, 'cropped_img')
        ecg = ECGSegment(
            param_file=param_file,
            param_set=param_set
        )
        ecg.run(
            data_dir=unet_data_dir,
            models_dir=model_folder,
            cv=3,
            resume_training=True,
            checkpoint_path=os.path.join(model_folder, 'segmentation_base_model.pth')
        )
        
        if verbose:
            print("Segmentation model training completed.")
    
    # def train_mmseg_model(self, data_folder, model_folder, verbose):
    #     classes = ('bg', 'signal')
    #     palette = [[255,255,255], [0,0,0]]


    #     @DATASETS.register_module()
    #     class ECGDataset(BaseSegDataset):
    #         METAINFO = dict(classes = classes, palette = palette)
    #         def __init__(self, **kwargs):
    #             super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)
            
    #     # from mmengine import Config
    #     cfg = Config.fromfile('/scratch/hshang/moody/mmsegmentation_MINS/demo/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive-ecg.py')
    #     # from mmengine.runner import Runner

    #     runner = Runner.from_cfg(cfg)

    #     runner.train()

    def train_model(self, data_folder, model_folder, verbose):
        
        multiprocessing.set_start_method('spawn')
        generate_data(data_folder, model_folder, 5000, verbose)
        prepare_data_for_training(data_folder, verbose)
        # self.train_segmentation_model(data_folder, model_folder, verbose)
        if verbose:
            print('Training the digitization model...')
            print('Finding the Challenge data...')

        # Reduce the number of repeated compilations and improve
        # training speed.
        setup_cache_size_limit_of_dynamo()
        
        
        # self.train_detection_model(cfg, model_folder, verbose)
        

        # Start the training in separate threads
        detection_thread = multiprocessing.Process(target=self.train_detection_model, args=(data_folder, model_folder, verbose))
        segmentation_thread = multiprocessing.Process(target=self.train_segmentation_model, args=(data_folder, model_folder, verbose))
    
        detection_thread.start()
        segmentation_thread.start()

        # Wait for both threads to complete
        detection_thread.join()
        segmentation_thread.join()

        if verbose:
            print("Both detection and segmentation models have been trained.")
        

    
    def run_digitization_model(self, record, verbose):
        
        # load image paths
        path = os.path.split(record)[0]
        image_files = get_image_files(record)

        images = list()
        for image_file in image_files:
            image_file_path = os.path.join(path, image_file)
            if os.path.isfile(image_file_path):
                images.append(image_file_path)
                
        # assume there is only one image per record
        img_path = images[0]

        img = mmcv.imread(img_path,channel_order='rgb')
        # remove image gradient
        # img_no_grad = remove_image_gradients(img)
        # mmcv.imwrite(img_no_grad, os.path.join(record, 'processed.png'))
        result = inference_detector(self.model, img)
        result_dict = result.to_dict()
        pred = result_dict['pred_instances']
        bboxes = pred['bboxes'].to(torch.int).cpu().detach().numpy()
        # check if pred has masks 
        masks = np.zeros((bboxes.shape[0], img.shape[0], img.shape[1]))
        if 'masks' in pred:
            masks = pred['masks'].cpu().detach().numpy()
        scores = pred['scores'].cpu().detach().numpy()
        labels = pred['labels'].cpu().detach().numpy()
        
        
        
        # patches = crop_from_bbox(bboxes, img)
        
        # print(f"patches shape: {patches[0].shape}")

        bboxes, labels, scores, masks = filter_boxes(bboxes, labels, scores, masks)
        # assert len(bboxes) >= 12, f"Expected at least 12 bboxes, got {len(bboxes)}"
        # assert len(bboxes) == masks.shape[0], f"Expected {len(bboxes)} bboxes, got {masks.shape[0]}"
        image = img/255.0
        # assert bboxes.shape == (13, 4), f"Expected shape (13, 4), got {bboxes.shape}"
        
        
        # cfg = Config.fromfile('/scratch/hshang/moody/mmsegmentation_MINS/demo/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive-ecg.py')
        # # Init the model from the config and the checkpoint
        # checkpoint_path = '/scratch/hshang/moody/mmsegmentation_MINS/demo/work_dirs/ECG/iter_400.pth'

        # Load models into memory
        # inferencer = MMSegInferencer(model=cfg, weights=checkpoint_path)
        # Inference
        # crop the images with the bboxes and put them into an array
        # to_be_readout = np.zeros((image.shape[0], image.shape[1], len(bboxes)))
        # print(to_be_readout.shape)
        # for i, (x1, y1, x2, y2) in enumerate(bboxes):
        #     lead = img[y1:y2, x1:x2, :]
        #     cv2.imwrite(f'lead_{i}.png', lead)
        #     # print(lead.shape)
        #     # result = inferencer(lead)['predictions']

        #     result = inference_model(self.mmseg, lead)
        #     print(result.keys())
        #     cv2.imwrite(f'leadout_{i}.png', result)
        #     # print(result.shape)
        #     to_be_readout[y1:y2, x1:x2, i] = result
        mV_pixel = (25.4 *8.5*0.5)/(masks[0].shape[0]*5) #hardcoded for now
        # # mV_pixel = (1.5*25.4 *8.5*0.5)/(masks[0].shape[0]*5)
        header_path = hc.get_header_file(record)
        with open(header_path, 'r') as f:
            input_header = f.read()

        num_samples = get_num_samples(input_header)
        
        sorted_bboxes, nrows = bboxes_sorting(bboxes, masks.shape[1])
        if sorted_bboxes is None:
            # failed to detect leads
            empty_signals_np = np.full((12, num_samples), np.nan)
            lead_length = num_samples // 4
            empty_signals_np[0:3,0:lead_length] = 0
            empty_signals_np[3:6,lead_length:2*lead_length] = 0
            empty_signals_np[6:9,2*lead_length:3*lead_length] = 0
            empty_signals_np[9:12,3*lead_length:4*lead_length] = 0
            return empty_signals_np.T if empty_signals_np.shape[1] > empty_signals_np.shape[0] else empty_signals_np
        
        to_be_readout = self.unet.run(image, sorted_bboxes.astype(int)) # float

        to_be_readout = np.where(to_be_readout > 0.5, True, False)
        # print(min(to_be_readout[0]), max(to_be_readout[0]))
        # assert len(to_be_readout) == 13, f"Expected 13 signals, got {len(to_be_readout)}"
        # assert len(bboxes) == len(to_be_readout), f"Expected {len(bboxes)} signals, got {len(to_be_readout)}"
        # assert to_be_readout[0].shape == (img.shape[0], img.shape[1]), f"Expected shape {(img.shape[0], img.shape[1])}, got {to_be_readout[0].shape}"
        
        
        # assert to_be_readout.shape == masks.shape, f"Expected shape {masks.shape}, got {to_be_readout.shape}"
        # assert to_be_readout.shape[0] == 13, f"Expected 13 signals, got {to_be_readout.shape[0]}"
        
        

        # load gt masks for debuging:
        # Get directory and image name
        # directory_path = os.path.dirname(img_path)
        # img_name = os.path.splitext(os.path.basename(img_path))[0]

        # # Correctly assign paths
        # bbox_path = os.path.join(directory_path, img_name + '.json')
        # mask_path = os.path.join(directory_path, img_name + '_mask.png')

        # # Load bounding box coordinates from JSON
        # with open(bbox_path, 'r') as file:
        #     settings = json.load(file)

        # gt_bboxes = []
        # for lead in settings['leads']:
        #     coords = lead['lead_bounding_box']
        #     x_coords = [coord[1] for coord in coords.values()]
        #     y_coords = [coord[0] for coord in coords.values()]
        #     x_min, x_max = min(x_coords), max(x_coords)
        #     y_min, y_max = min(y_coords), max(y_coords)
        #     gt_bboxes.append([x_min, y_min, x_max, y_max])

        # # Convert gt_bboxes to a numpy array
        # gt_bboxes = np.array(gt_bboxes)

        # # Load the mask
        # gt_mask_load = mmcv.imread(mask_path, flag='grayscale')
        # gt_masks = []

        # # Generate masks for each bounding box
        # for gt_bbox in gt_bboxes:
        #     x1, y1, x2, y2 = map(int, gt_bbox)  # Ensure coordinates are integers
        #     gt_mask = np.zeros_like(gt_mask_load)
        #     gt_mask[y1:y2, x1:x2] = gt_mask_load[y1:y2, x1:x2]
        #     gt_masks.append(gt_mask)

        # # Convert gt_masks to a numpy array
        # gt_masks = np.array(gt_masks)
        # gt_masks = np.where(gt_masks > 0.1, True, False)
                
        
            
        
        # assert gt_masks.shape == to_be_readout.shape, f"Expected shape {to_be_readout.shape}, got {gt_masks.shape}"
        # signal=readOut(header_path, masks, bboxes, mV_pixel)
        
        print('dumping pred')
        
        signal=readOut(num_samples, to_be_readout, nrows, sorted_bboxes, mV_pixel)
        

        to_dump = {'bboxes': sorted_bboxes, 'masks': to_be_readout, 'scores': scores, 'labels': labels, 'record': record, 'nrows': nrows, 'signal_est':signal}
        with open('to_dump.pkl', 'wb') as f:
            pickle.dump(to_dump, f)

        # print('dumping gt')
        # signal=readOut(header_path, gt_masks, gt_bboxes, mV_pixel)
        # to_dump = {'bboxes': gt_bboxes, 'masks': gt_masks, 'scores': scores, 'labels': labels, 'record': record}
        # with open('to_dump.pkl', 'wb') as f:
        #     pickle.dump(to_dump, f)
        return signal
    


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