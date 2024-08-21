

from TeamCode.src import our_paths
from TeamCode.src.interface import AbstractDigitizationModel, AbstractClassificationModel
from TeamCode.src.verify_environment import verify_environment
import helper_code as hc
import numpy as np
import os
import pickle
from PIL import Image

# from mmseg.registry import DATASETS

from mmengine import Registry
from mmseg.datasets import BaseSegDataset
from mmseg.apis import init_model, inference_model
from mmseg.apis import show_result_pyplot



from concurrent.futures import ThreadPoolExecutor

# import multiprocessing

import torch
import warnings

# from TeamCode.src.ecg_predict import ECGPredictor
# from TeamCode.src.ecg_main import ECGSegment

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo
import mmcv

from mmdet.apis import init_detector, inference_detector
from helper_code import get_num_samples, get_signal_names, get_image_files

import random
from TeamCode.src.ecg_image_generator.helper_functions import find_records
from TeamCode.src.ecg_image_generator.gen_ecg_image_from_data import run_single_file
import warnings
import json
from argparse import Namespace

import os.path as osp
from mmcv import imread, imwrite
import mmengine
from tqdm.auto import tqdm
import shutil


# from pycocotools import mask as maskutils
import random

import pywt

from scipy.signal import butter, lfilter
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

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


def filter_boxes(pred_bboxes, pred_labels, pred_scores):
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
        pred_scores = pred_scores[pred_scores > 0]
    
    if len(pred_scores) > 13:
        # sort the scores and get the top 13
        indices = np.argsort(pred_scores)[::-1][:13]
        pred_bboxes = pred_bboxes[indices]
        pred_labels = pred_labels[indices]
        pred_scores = pred_scores[indices]
    
    return pred_bboxes, pred_labels, pred_scores

def apply_savgol_filter(signal):
    return savgol_filter(signal, window_length=5, polyorder=3)  # Adjust parameters as needed

def bboxes_sorting(bboxes, img_height):
    # Validate input
    if not isinstance(bboxes, np.ndarray) or bboxes.ndim != 2 or bboxes.shape[1] < 4:
        print('Invalid input')
        return None, None
    if len(bboxes) == 0:
        print('Empty input')
        return None, None
    # Calculate average height and width
    bboxes_avg_H = (bboxes[:,1] + bboxes[:,3]) / 2
    bboxes_avg_W = (bboxes[:,0] + bboxes[:,2]) / 2
    # Append these to bboxes
    bboxes = np.append(bboxes, bboxes_avg_H.reshape((-1, 1)), axis=1)
    bboxes = np.append(bboxes, bboxes_avg_W.reshape((-1, 1)), axis=1)
    # Sort by height (H)
    sortH_idx = bboxes[:, 4].argsort()
    bboxes = bboxes[sortH_idx]
    # Grouping function (assuming it's defined elsewhere)
    row_dict, row_dict_numcol = group_bboxes_row(bboxes, vertical_distance_threshold=img_height/10)
    if len(row_dict) == 4:
        ncol = max(list(row_dict_numcol.values()))
        if ncol >= 4:
            print('row=4, col>=4')
            rowswith4cols = [k for k, v in row_dict_numcol.items() if int(v) == 4]
            # Initialize left and right with default values
            left = float('inf')
            right = float('-inf')
            # Only proceed if rowswith4cols is not empty
            if rowswith4cols:
                left_values = [np.min(row_dict[row][:, 0]) for row in rowswith4cols if len(row_dict[row]) > 0]
                right_values = [np.max(row_dict[row][:, 2]) for row in rowswith4cols if len(row_dict[row]) > 0]
                if left_values:
                    left = min(left_values)
                if right_values:
                    right = max(right_values)
            # Ensure that left and right have valid values
            if left == float('inf') or right == float('-inf'):
                print('Invalid left or right')
                return None, None
            leadwidth = (right - left) / 4 if right != left else 1  # Prevent division by zero
            leftmid = left + leadwidth
            mid = left + 2 * leadwidth
            rightmid = left + 3 * leadwidth
            for key in [0, 1, 2]:
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
                    new_bboxes = np.nan * np.ones((4, 4))
                    for i in range(row_dict[key].shape[0]):
                        h, w = row_dict[key][i, 4:6]
                        if w < leftmid:
                            new_bboxes[0] = [left, row_dict[key][i, 1], leftmid, row_dict[key][i, 3]]
                        elif w < mid:
                            new_bboxes[1] = [leftmid, row_dict[key][i, 1], mid, row_dict[key][i, 3]]
                        elif w < rightmid:
                            new_bboxes[2] = [mid, row_dict[key][i, 1], rightmid, row_dict[key][i, 3]]
                        elif w < right:
                            new_bboxes[3] = [rightmid, row_dict[key][i, 1], right, row_dict[key][i, 3]]
                    row_dict[key] = new_bboxes
            row_dict[3][0, 0] = min(row_dict[3][0, 0], left)
            row_dict[3][0, 2] = max(row_dict[3][0, 2], right)
            bboxes = np.concatenate([row_dict[0][:, 0:4], row_dict[1][:, 0:4], row_dict[2][:, 0:4], row_dict[3][:, 0:4]], axis=0)
            standard_format = [0, 12, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
            if bboxes.shape[0] != 13:
                print('none!')
                return None, None
            bboxes = bboxes[standard_format]
            if len(bboxes) != 12:
                print(f'Unexpected number of bounding boxes: {len(bboxes)}')
                return None, None
            return bboxes, 4
    elif len(row_dict) == 3:
        ncol = max(list(row_dict_numcol.values()))
        if ncol >= 4:
            print('row=3, col>=4')
            rowswith4cols = [k for k, v in row_dict_numcol.items() if int(v) == 4]
            # Initialize left and right with default values
            left = float('inf')
            right = float('-inf')
            # Only proceed if rowswith4cols is not empty
            if rowswith4cols:
                left_values = [np.min(row_dict[row][:, 0]) for row in rowswith4cols if len(row_dict[row]) > 0]
                right_values = [np.max(row_dict[row][:, 2]) for row in rowswith4cols if len(row_dict[row]) > 0]
                if left_values:
                    left = min(left_values)
                if right_values:
                    right = max(right_values)
            # Ensure that left and right have valid values
            if left == float('inf') or right == float('-inf'):
                print('Invalid left or right')
                return None, None
            leadwidth = (right - left) / 4 if right != left else 1  # Prevent division by zero
            leftmid = left + leadwidth
            mid = left + 2 * leadwidth
            rightmid = left + 3 * leadwidth
            for key in [0, 1, 2]:
                if row_dict_numcol[key] == 4:
                    row_dict[key] = row_dict[key][row_dict[key][:, 5].argsort()]
                    row_dict[key][0, 0] = left
                    row_dict[key][0, 2] = leftmid
                    row_dict[key][1, 0] = leftmid
                    row_dict[key][1, 2] = mid
                    row_dict[key][2, 0] = mid
                    row_dict[key][2, 2] = rightmid
                    row_dict[key][3, 0] = rightmid
                    row_dict[key][3, 2] = right
                else:
                    new_bboxes = np.nan * np.ones((4, 4))
                    for i in range(row_dict[key].shape[0]):
                        h, w = row_dict[key][i, 4:6]
                        if w < leftmid:
                            new_bboxes[0] = [left, row_dict[key][i, 1], leftmid, row_dict[key][i, 3]]
                        elif w < mid:
                            new_bboxes[1] = [leftmid, row_dict[key][i, 1], mid, row_dict[key][i, 3]]
                        elif w < rightmid:
                            new_bboxes[2] = [mid, row_dict[key][i, 1], rightmid, row_dict[key][i, 3]]
                        elif w < right:
                            new_bboxes[3] = [rightmid, row_dict[key][i, 1], right, row_dict[key][i, 3]]
                    row_dict[key] = new_bboxes
            # extrapolate the position of row 4 based on row 1,2,3
            row3_median_H = np.nanmedian(row_dict[2][:, 4])
            row1_median_H = np.nanmedian(row_dict[0][:, 4])
            row4_median_H = row3_median_H + 0.5*(row3_median_H - row1_median_H)
            if (not np.isnan(row_dict[1][0,3])) and (not np.isnan(row_dict[1][0,1])):
                row4_height = row_dict[1][0,3] - row_dict[1][0,1]
                row4_top = row4_median_H - 0.5*row4_height
                row4_bottom = row4_median_H + 0.5*row4_height
            else:
                row4_height = (np.nanmedian(row_dict[0][:,3] - row_dict[0][:,1]) + np.nanmedian(row_dict[1][:,3] - row_dict[1][:,1]) + np.nanmedian(row_dict[2][:,3] - row_dict[2][:,1]))/3
                row4_top = row4_median_H - 0.5*row4_height
                row4_bottom = row4_median_H + 0.5*row4_height
            if row4_top < img_height and row4_bottom < img_height:
                row_dict[3] = np.array([[left, row4_top, right, row4_bottom]])
                bboxes = np.concatenate([row_dict[0][:, 0:4], row_dict[1][:, 0:4], row_dict[2][:, 0:4], row_dict[3][:, 0:4]], axis=0)
                standard_format = [0, 12, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
                if bboxes.shape[0] != 13:
                    print('none!')
                    return None, None
                bboxes = bboxes[standard_format]
                return bboxes, 4
            else:
                bboxes = np.concatenate([row_dict[0][:, 0:4], row_dict[1][:, 0:4], row_dict[2][:, 0:4]], axis=0)
                standard_format = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
                if bboxes.shape[0] != 12:
                    print('none!')
                    return None, None
                bboxes = bboxes[standard_format]
                return bboxes, 3
    print('none!')
    return None, None



## helper functions end 

#format = [['I', 'aVR', 'V1', 'V4'], ['II', 'aVL', 'V2', 'V5'], ['III', 'aVF', 'V3', 'V6'], ['II']] # format is hardcoded for now
#format = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6']
#fullmode = 'II'
# import matplotlib.pyplot as plt





def crop_from_bbox(bbox, mask, mV_pixel):
    bbox = bbox.astype(int)
    ecg_segment = mask[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]

    # Early return if the segment is entirely empty
    if np.sum(ecg_segment) == 0:
        warnings.warn("All empty in ECG segment, returning NaN array", UserWarning)
        return np.full(ecg_segment.shape[1], np.nan)

    # Compute the weighting matrix
    weighting_matrix = np.linspace(
        (bbox[3] - bbox[1]) * mV_pixel / 2,
        -1 * (bbox[3] - bbox[1]) * mV_pixel / 2,
        num=ecg_segment.shape[0]
    ).reshape(-1, 1)
    weighted_ecg_segment = ecg_segment * weighting_matrix

    # Calculate the numerator and denominator
    denominator = np.sum(ecg_segment, axis=0)
    numerator = np.sum(weighted_ecg_segment, axis=0)

    # Initialize signal with NaNs
    signal = np.full(denominator.shape, np.nan)

    # Use a small epsilon to avoid division by very small values or zero
    epsilon = 1e-6
    valid_idx = denominator > epsilon
    signal[valid_idx] = numerator[valid_idx] / denominator[valid_idx]

    # Warn if the signal is all NaN
    if np.isnan(signal).all():
        warnings.warn("Signal is all NaN", UserWarning)

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
#     if np.isnan(signal).all():
#         warnings.warn("Signal is all nan", UserWarning)
    
    # return signal



def wavelet_denoising(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(denoised_coeffs, wavelet)


def readOut(num_samples, masks, nrows, bboxes, mV_pixel, sampling_frequency): # one more input: sampling_frequency
    num_signals = 12
    signals_np = np.full((num_signals, num_samples), np.nan)
    def process_signal(index, bboxes, masks, mV_pixel, signallen,sampling_frequency, filter=True):
        if np.isnan(bboxes[index]).any():
            print(f'Warning: Bounding box is nan, returning zeros')
            return np.zeros(signallen)
        signal = crop_from_bbox(bboxes[index], masks[index], mV_pixel)
        # check the number of nan in signal
        if np.isnan(signal).sum() > 0.5*signallen:
            print(f'Warning: More than 50% of signal is nan, returning zeros')
            return np.zeros(signallen)
        signal = interpolate_nan(signal) - np.mean(signal)
        try :
            signal = apply_savgol_filter(signal)
        except Exception as e:
            print(f'Error in savgol filter: {e}')
        try:
            signal = wavelet_denoising(signal)
        except Exception as e:
            print(f'Error in wavelet denoising: {e}')
        try:
            if len(signal) < signallen:
                signal = upsample(signal, signallen)
            else:
                signal = downsample(signal, signallen)
        except Exception as e:
            print(f'Error in upsampling/downsampling: {e}')
        # low pass filter added by Yani
        try:
            if filter:
                if not np.isnan(signal).any():
                    cutoff = sampling_frequency * 0.45  # normal ecg is 0.05Hz to around 150Hz
                    order =6
                    b, a = butter(order, cutoff, fs=sampling_frequency, btype='low', analog=False)
                    signal = lfilter(b, a, signal)
        except Exception as e:
            print(f'Error in butter filter: {e}')
        return signal
    if nrows == 4:
        for i in range(num_signals):
            signallen = num_samples if i == 1 else num_samples // 4
            start_idx = (num_samples // 4) * (i // 3)
            end_idx = start_idx + signallen
            if i < len(bboxes):
                signal = process_signal(i, bboxes, masks, mV_pixel, signallen, sampling_frequency, filter=True )
                signals_np[i, start_idx:end_idx] = signal
            else:
                signals_np[i, start_idx:end_idx] = np.zeros(signallen)
    elif nrows == 3:
        for i in range(bboxes.shape[0]):
            signallen = num_samples // 4
            start_idx = (num_samples // 4) * (i // 3)
            end_idx = start_idx + signallen
            signal = process_signal(i, bboxes, masks, mV_pixel, signallen, sampling_frequency, filter=True)
            signals_np[i, start_idx:end_idx] = signal
    signals_np = np.clip(signals_np, -2, 2)
    return signals_np.T if signals_np.shape[1] > signals_np.shape[0] else signals_np
    





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

def generate_data(data_folder,  config_folder, output_folder,  data_amount,  verbose):
    with open(os.path.join(config_folder, 'data_format.json'), 'r') as f:
        args_dict = json.load(f)
    args = Namespace(**args_dict)
    random.seed(args.seed)
    args.input_directory = data_folder
    args.output_directory = os.path.join(output_folder, 'processed_data')
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
    print(f"generating {args.max_num_images} images")
    with ThreadPoolExecutor() as executor:
        futures = []
        for full_header_file, full_recording_file in zip(full_header_files, full_recording_files):
            if i >= args.max_num_images:
                break  # Stop submitting new tasks once the limit is reached

            future = executor.submit(process_single_file, full_header_file, full_recording_file, args, original_output_dir)
            futures.append(future)
            i += 1  # Increment i after each task submission

        # Handle the results of the submitted tasks
        for future in futures:
            try:
                result = future.result()
                if verbose and result is not None:
                    print(f"Processed {result} files")
            except Exception as e:
                if verbose:
                    print(f"Error processing file: {e}")

    print(f"Finished generating {i} images")

def prepare_data_for_training(data_folder, verbose=False):
    # data_folder is the processed_data folder
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


    def convert_ecg_to_coco(data_path, bbox_dir, mask_dir, out_file):
        # bbox_dir = osp.join(data_path, 'lead_bounding_box')
        # mask_dir = osp.join(data_path, 'masks')  # Path to mask directory
        annotations = []
        images = []
        obj_count = 0
        img_crop_dir = osp.join(data_path, 'cropped_img')
        os.makedirs(img_crop_dir, exist_ok=True)
        mask_crop_dir = osp.join(data_path,'cropped_masks')
        os.makedirs(mask_crop_dir, exist_ok=True)
        
        for entry in tqdm(os.listdir(data_path,), desc='Processing file'):
            if entry.endswith(".png"):
                img_name = entry[:-4]
                bbox_path = osp.join(bbox_dir, f'{img_name}.json')
                mask_path = osp.join(mask_dir, f'{img_name}.png') 
                if not osp.exists(bbox_path) or not osp.exists(mask_path):
                    print(f"Skipping {img_name} as bbox or mask is missing")
                    continue 
                
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
                    
                for lead in settings['leads'][:12]: # only 12 leads
                    coords = lead['lead_bounding_box']
                    x_coords = [coord[1] for coord in coords.values()]
                    y_coords = [coord[0] for coord in coords.values()]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                
                    # Extract the sub-region from the mask
                    lead_mask = mask[y_min:y_max, x_min:x_max]
                    # prepare the lead crop for segmentation model
                    lead_crop = image[y_min:y_max, x_min:x_max]
                    
                    save_crop = np.random.choice([True, False], p=[0.2, 0.8])
                    
                    if save_crop:
                        palette = [[0,0,0], [255,255,255]]
                        lead_mask = np.where(lead_mask < 50, 0, 1).astype(np.uint8)
                        lead_mask = Image.fromarray(lead_mask).convert('P')
                        lead_mask.putpalette(np.array(palette, dtype=np.uint8))
                        lead_mask = np.array(lead_mask)
                        # print(f"Saving crop for {img_name} to {img_crop_dir}")
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
    os.path.join(data_folder, 'processed_data'),
    os.path.join(data_folder, 'processed_data' ),
    os.path.join(data_folder, 'processed_data', 'masks'),
    os.path.join(data_folder, 'processed_data', 'annotation_coco.json'))


class OurDigitizationModel(AbstractDigitizationModel):
    def __init__(self):
        verify_environment()
        self.det_config = None
        self.det_model = None
        # self.unet = None
        self.seg_config = None
        self.seg_model = None
        # self.base_dir = os.path.dirname(os.path.abspath(__file__))  # Path to the directory containing your_script.py
        self.src_dir = 'TeamCode/src'
        self.config_dir = os.path.join(self.src_dir, 'configs_ckpts')
        self.processed_data_dir = self.src_dir


    @classmethod
    def from_folder(cls, model_folder, verbose):
         # Create an instance of the class
        instance = cls()
        
        if not os.path.exists(os.path.join(model_folder, "maskrcnn")):
            os.makedirs(os.path.join(model_folder, "maskrcnn"))

        if not os.path.exists(os.path.join(model_folder, "maskrcnn/maskrcnn_res101.py")):
            #copy the config file to the model folder
            shutil.copy(os.path.join(instance.config_dir, 'maskrcnn', 'maskrcnn_res101.py'), os.path.join(model_folder, 'maskrcnn','maskrcnn_res101.py'))
        instance.det_config = os.path.join(model_folder, "maskrcnn/maskrcnn_res101.py")
        
        # Construct checkpoint path based on the model_folder parameter
        
        maskrcnn_checkpoint_log = os.path.join(model_folder, 'maskrcnn/last_checkpoint')
        if not os.path.exists(maskrcnn_checkpoint_log):
            print(f"last_checkpoint not found in {model_folder}")
            maskrcnn_checkpoint_log = os.path.join(instance.config_dir, 'maskrcnn/last_checkpoint')
            
        with open(maskrcnn_checkpoint_log, 'r') as f:
            first_line = f.readline().strip()  # Read the first line and strip any whitespace/newline characters
            model_name = os.path.basename(first_line)
            # Check if the path exists based on the first line
            if not os.path.exists(os.path.join(model_folder, first_line)):
                maskrcnn_checkpoint_file = os.path.join(instance.config_dir, 'maskrcnn/epoch_12.pth')
                if not os.path.exists(os.path.join(model_folder, model_name)):
                    shutil.copy(maskrcnn_checkpoint_file, model_folder)
            else: 
                maskrcnn_checkpoint_file = os.path.join(model_folder, first_line)

        if not os.path.exists(os.path.join(model_folder, "unet")):
            os.makedirs(os.path.join(model_folder, "unet"))
        if not os.path.exists(os.path.join(model_folder, "unet/unet.py")):
            #copy the config file to the model folder
            shutil.copy(os.path.join(instance.config_dir, 'unet/unet.py'), os.path.join(model_folder, 'unet/unet.py'))
        instance.seg_config = os.path.join(model_folder, "unet/unet.py")
        
        # Construct checkpoint path based on the model_folder parameter
        
        unet_checkpoint_log = os.path.join(model_folder, 'unet/last_checkpoint')
        if not os.path.exists(unet_checkpoint_log):
            print(f"last_checkpoint not found in {model_folder}")
            unet_checkpoint_log = os.path.join(instance.config_dir, 'unet/last_checkpoint')
            
        with open(unet_checkpoint_log, 'r') as f:
            first_line = f.readline().strip()  # Read the first line and strip any whitespace/newline characters
            unet_model_name = os.path.basename(first_line)
            # Check if the path exists based on the first line
            if not os.path.exists(os.path.join(model_folder, first_line)):
                unet_checkpoint_file = os.path.join(instance.config_dir, 'unet/iter_400.pth')
                if not os.path.exists(os.path.join(model_folder, unet_model_name)):
                    shutil.copy(unet_checkpoint_file, model_folder)
            else: 
                unet_checkpoint_file = os.path.join(model_folder, first_line)
        

                
        # print(f"this is the det_dir {instance.det_config}, this is the mrcnn ckpt {maskrcnn_checkpoint_file}")

        # Initialize the model using instance-specific variables
        # load model parameters from json file
        # if not os.path.exists(os.path.join(model_folder, 'ecg_params.json')):
        #     shutil.copy(os.path.join(instance.config_dir, 'ecg_params.json'), model_folder)
        # with open(os.path.join(model_folder, 'ecg_params.json'), 'r') as f:
        #     ecg_params = json.load(f)['segmentation']
        

        # if not os.path.exists(os.path.join(model_folder, 'segmentation/segmentation_model.pth')):
        #     os.makedirs(os.path.join(model_folder, 'segmentation'), exist_ok=True)
        #     shutil.copy(os.path.join(instance.config_dir, 'segmentation/segmentation_model.pth'), os.path.join(model_folder, 'segmentation/segmentation_model.pth'))
        # # print(f"this is the det_dir {instance.det_config}, this is the mrcnn ckpt {maskrcnn_checkpoint_file}")
        
        # unet_checkpoint_file = os.path.join(model_folder, 'segmentation/segmentation_model.pth')
        # print(f"this is the unet ckpt {unet_checkpoint_file}")
        instance.det_model = init_detector(instance.det_config, maskrcnn_checkpoint_file, device=dev)
        # instance.unet = ECGPredictor('resunet10', unet_checkpoint_file, size=ecg_params['crop'], cbam=False)
        instance.seg_model = init_model(config=instance.seg_config, checkpoint=unet_checkpoint_file, device=dev)
        if verbose:
            print(f"Model loaded from {maskrcnn_checkpoint_file}, {unet_checkpoint_file}")

        return instance

    def train_detection_model(self, model_folder, verbose):
        # load config
       
        # work_dir = os.path.join(base_dir, 'work_dir')
        config_file_path = os.path.join(self.config_dir, 'maskrcnn', 'maskrcnn_res101.py')
        cfg = Config.fromfile(config_file_path)
        cfg.metainfo = {
            'classes': ('ecg_lead', ),
            'palette': [
                (220, 20, 60),
            ]
        }
        cfg.data_root = os.path.join(self.processed_data_dir, 'processed_data')
        cfg.train_dataloader.dataset.ann_file = 'annotation_coco.json'
        cfg.train_dataloader.dataset.data_root = cfg.data_root
        cfg.train_dataloader.dataset.data_prefix.img = ''
        cfg.train_dataloader.dataset.metainfo = cfg.metainfo
        cfg.model.backbone.init_cfg.checkpoint = os.path.join(self.config_dir, "original_pretrained_weights", "resnet101_msra-6cc46731.pth")
        

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
        
        cfg.work_dir = os.path.join(model_folder, 'maskrcnn')
        # assert os.path.exists(os.path.join(base_dir,'checkpoints')), f'ckpt_root is not found'
        cfg.load_from = os.path.join(self.config_dir,"original_pretrained_weights", 'mask_rcnn_r101_caffe_fpn_1x_coco_20200601_095758-805e06c1.pth')

        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

        if verbose:
            print("Training detection model...")
        
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

    # def train_segmentation_model(self, data_folder, model_folder, verbose):
    #     if verbose:
    #         print("Training segmentation model...")
        
    #     param_file = os.path.join(self.config_dir, 'ecg_params.json')
    #     param_set = "segmentation"
    #     unet_data_dir = os.path.join(data_folder, 'processed_data', 'cropped_img')
    #     ecg = ECGSegment(
    #         param_file=param_file,
    #         param_set=param_set
    #     )
    #     ecg.run(
    #         data_dir=unet_data_dir,
    #         models_dir=model_folder,
    #         cv=3,
    #         resume_training=True,
    #         checkpoint_path=os.path.join(self.config_dir, 'segmentation_base_model.pth')
    #     )
        
    #     if verbose:
    #         print("Segmentation model training completed.")
    
    def train_segmentation_model(self, model_folder, verbose):
    #     # split train/val set randomly

        data_root = os.path.join(self.processed_data_dir, 'processed_data')
        split_dir = 'splits'
        img_dir = 'cropped_img'
        ann_dir = 'cropped_masks'
        
        from PIL import Image
        # convert dataset annotation to semantic segmentation map
       

        mmengine.mkdir_or_exist(osp.join(data_root, split_dir))
        filename_list = [osp.splitext(filename)[0] for filename in mmengine.scandir(
            osp.join(data_root, ann_dir), suffix='.png')]
        with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
        # select first 4/5 as train set
            train_length = int(len(filename_list)*4/5)
            f.writelines(line + '\n' for line in filename_list[:train_length])
        with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
        # select last 1/5 as train set
            f.writelines(line + '\n' for line in filename_list[train_length:])

        cfg = Config.fromfile('/scratch/hshang/moody/final_phase_submission/official-phase-mins-eth/TeamCode/src/configs_ckpts/unet/unet.py')

        crop_size = (256, 256)

        cfg.train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(0.5, 1.5),  # Adjusted to prevent aggressive downscaling
                scale=(256, 256),  # Increased scale size
                type='RandomResize'),
            dict(cat_max_ratio=0.75, crop_size=crop_size, type='RandomCrop'),
            dict(type='ResizeToMultiple', size_divisor=16),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ]

        # cfg.val_pipeline = [
        #     dict(type='LoadImageFromFile'),
        #     dict(keep_ratio=True, scale=(512, 512), type='Resize'),  # Match with training scale
        #     dict(type='ResizeToMultiple', size_divisor=16),
        #     dict(type='LoadAnnotations'),
        #     dict(type='PackSegInputs'),
        # ]

        # # Since we use only one GPU, BN is used instead of SyncBN
        # cfg.norm_cfg = dict(type='BN', requires_grad=True)
        # cfg.crop_size = (256, 256)
        # cfg.model.data_preprocessor.size = cfg.crop_size
        # cfg.model.backbone.norm_cfg = cfg.norm_cfg
        # cfg.model.decode_head.norm_cfg = cfg.norm_cfg
        # cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg




        # Modify dataset type and path
        cfg.dataset_type = 'HRFDataset'
        cfg.data_root = data_root

        cfg.data_root = data_root
        cfg.load_from = os.path.join(self.config_dir, "original_pretrained_weights", 'deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive_20211210_201825-6bf0efd7.pth')
        cfg.work_dir = model_folder
        cfg.model.decode_head.num_classes = 2
        cfg.model.auxiliary_head.num_classes = 2


        cfg.train_dataloader.dataset.data_root = cfg.data_root
        cfg.train_dataloader.dataset.data_prefix = dict(img_path='cropped_img', seg_map_path='cropped_masks')
        cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
        cfg.train_dataloader.dataset.ann_file = 'splits/train.txt'

        cfg.val_dataloader.dataset.type = cfg.dataset_type
        cfg.val_dataloader.dataset.data_root = cfg.data_root
        cfg.val_dataloader.dataset.data_prefix = dict(img_path='cropped_img', seg_map_path='cropped_masks')
        cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
        cfg.val_dataloader.dataset.ann_file = 'splits/val.txt'

        cfg.test_dataloader = cfg.val_dataloader

        cfg.train_dataloader.batch_size = 8  # Try reducing the batch size

        cfg.train_cfg.max_iters = 200
        cfg.train_cfg.val_interval = 200
        cfg.default_hooks.logger.interval = 10
        cfg.default_hooks.checkpoint.interval = 200

        # cfg.dataset_type = 'ECGDataset'
        # Set seed to facilitate reproducing the result
        cfg['randomness'] = dict(seed=0)


        runner = Runner.from_cfg(cfg)

        runner.train()

    def train_model(self, data_folder, model_folder, verbose):
        
        # multiprocessing.set_start_method('spawn')
        generate_data(data_folder, self.config_dir, self.processed_data_dir, 5000, verbose)
        prepare_data_for_training(self.processed_data_dir, verbose)
        
        if verbose:
            print('Training the digitization model...')
            print('Finding the Challenge data...')

        # Reduce the number of repeated compilations and improve
        # training speed.
        setup_cache_size_limit_of_dynamo()
        
        # self.train_segmentation_model(data_folder, model_folder, verbose)
        # # self.train_detection_model(cfg, model_folder, verbose)
        self.train_segmentation_model(model_folder, verbose)
        self.train_detection_model(model_folder, verbose)

        

        # self.train_segmentation_model(data_folder, model_folder, verbose)
        # Start the training in separate threads
        # detection_thread = multiprocessing.Process(target=self.train_detection_model, args=(data_folder, model_folder, verbose))
        # segmentation_thread = multiprocessing.Process(target=self.train_segmentation_model, args=(data_folder, model_folder, verbose))
    
        # detection_thread.start()
        # segmentation_thread.start()

        # # Wait for both threads to complete
        # detection_thread.join()
        # segmentation_thread.join()

        if verbose:
            print("Both detection and segmentation models have been trained.")
        

    
    def run_digitization_model(self, record, verbose):

        image_files = get_image_files(record)
        
        # load image paths
        path = os.path.split(record)[0]

        images = list()
        for image_file in image_files:
            image_file_path = os.path.join(path, image_file)
            if os.path.isfile(image_file_path):
                images.append(image_file_path)
                
        # assume there is only one image per record
        img_path = images[0]

        img = mmcv.imread(img_path,channel_order='rgb')
        mV_pixel = (25.4 *8.5*0.5)/(img[0].shape[0]*5) #hardcoded for now
        # # mV_pixel = (1.5*25.4 *8.5*0.5)/(masks[0].shape[0]*5)
        header_path = hc.get_header_file(record)
        with open(header_path, 'r') as f:
            input_header = f.read()

        num_samples = get_num_samples(input_header)
        
        empty_signals_np = np.full((12, num_samples), np.nan)
        lead_length = num_samples // 4
        empty_signals_np[0:3,0:lead_length] = 0
        empty_signals_np[3:6,lead_length:2*lead_length] = 0
        empty_signals_np[6:9,2*lead_length:3*lead_length] = 0
        empty_signals_np[9:12,3*lead_length:4*lead_length] = 0
        
        
        with Registry('scope').switch_scope_and_registry('mmdet'):
            result = inference_detector(self.det_model, img)
        result_dict = result.to_dict()
        pred = result_dict['pred_instances']
        bboxes = pred['bboxes'].to(torch.int).cpu().detach().numpy()
        scores = pred['scores'].cpu().detach().numpy()
        labels = pred['labels'].cpu().detach().numpy()
        

        bboxes, labels, scores= filter_boxes(bboxes, labels, scores)
        image = img/255.0
        

  
        
        
        try:
            sorted_bboxes, nrows = bboxes_sorting(bboxes, img.shape[1])
        except Exception as e:
            print(f'Error in sorting bboxes: {e}')
            return empty_signals_np.T if empty_signals_np.shape[1] > empty_signals_np.shape[0] else empty_signals_np
            
        if sorted_bboxes is None:
            # failed to detect leads
            
            print("sorting failed")
            return empty_signals_np.T if empty_signals_np.shape[1] > empty_signals_np.shape[0] else empty_signals_np
        
        # mmseg stuff
        to_be_readout = []
        img_to_dump = []

        for i, (x1, y1, x2, y2) in enumerate(sorted_bboxes):
            if np.isnan([x1, y1, x2, y2]).any():
                print(f'Warning: Bounding box at index {i} is NaN, skipping')
                continue
            
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            lead = img[y1:y2, x1:x2]
            
            result = inference_model(self.seg_model, lead)
            pred_res = result.pred_sem_seg.data.cpu().detach().numpy().squeeze(0)
            
            assert pred_res.shape == lead.shape[:2]
            
            mask = np.zeros_like(img[:, :, 0], dtype=float)  # Assuming img is 3D (height, width, channels)
            mask[y1:y2, x1:x2] = pred_res
            
            to_be_readout.append(mask)
            img_to_dump.append(pred_res)

        with open ('/scratch/hshang/moody/final_phase_submission/official-phase-mins-eth/images.pkl', 'wb') as f:
            pickle.dump(img_to_dump, f)


        # try:
        #     to_be_readout = self.unet.run(image, sorted_bboxes.astype(int)) # float
        # except Exception as e:
        #     "Error in unet: {e}"
        #     return empty_signals_np.T if empty_signals_np.shape[1] > empty_signals_np.shape[0] else empty_signals_np
        # to_be_readout = [np.where(arr > 0.5, True, False) for arr in to_be_readout]

        
        

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
        
        
        
        freq = hc.get_sampling_frequency(input_header)
        
        try:
            signal=readOut(num_samples, to_be_readout, nrows, sorted_bboxes, mV_pixel, freq)
        except Exception as e:
            print(f'Error in readout: {e}')
            return empty_signals_np.T if empty_signals_np.shape[1] > empty_signals_np.shape[0] else empty_signals_np

        print('dumping pred')
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