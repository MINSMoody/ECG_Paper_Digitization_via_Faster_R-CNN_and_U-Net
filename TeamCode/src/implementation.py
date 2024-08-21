
from TeamCode.src.interface import AbstractDigitizationModel, AbstractClassificationModel
from TeamCode.src.verify_environment import verify_environment
import helper_code as hc
import numpy as np
import os
import pickle
from PIL import Image
from pathlib import Path


from mmengine import Registry
from mmseg.apis import init_model, inference_model



from concurrent.futures import ThreadPoolExecutor

# import multiprocessing

import torch
import warnings


from mmengine.config import Config
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
        self.seg_config = None
        self.seg_model = None
        self.src_dir = Path('TeamCode/src')  # Convert to Path object
        self.config_dir = self.src_dir / 'configs_ckpts' 
        self.processed_data_dir = self.src_dir
        self.device = dev


    @classmethod
    def from_folder(cls, model_folder, verbose):
        """
        Initializes the detection and segmentation models from the specified folder.

        Args:
            model_folder (str): Path to the model directory.
            device (str): Device to load the models on ('cuda:0' or 'cpu').
            verbose (bool): If True, prints detailed information during initialization.

        Returns:
            YourClassName: An instance of the class with loaded models.
        """
        instance = cls()
        model_folder = Path(model_folder)
        
        # Ensure model_folder exists
        model_folder.mkdir(parents=True, exist_ok=True)

        # Initialize Mask R-CNN (Detection Model)
        instance.det_config, det_checkpoint = instance._prepare_model_files(
            model_folder=model_folder,
            model_name='maskrcnn',
            default_config_path=instance.config_dir / 'maskrcnn' / 'maskrcnn_res101.py',
            default_checkpoint_path=instance.config_dir / 'maskrcnn' / 'epoch_24.pth',
            verbose=verbose
        )
        instance.det_model = init_detector(
            config=str(instance.det_config),
            checkpoint=str(det_checkpoint),
            device=instance.device
        )

        # Initialize UNet (Segmentation Model)
        instance.seg_config, seg_checkpoint = instance._prepare_model_files(
            model_folder=model_folder,
            model_name='unet',
            default_config_path=instance.config_dir / 'unet' / 'unet.py',
            default_checkpoint_path=instance.config_dir / 'unet' / 'iter_400.pth',
            verbose=verbose
        )
        instance.seg_model = init_model(
            config=str(instance.seg_config),
            checkpoint=str(seg_checkpoint),
            device=instance.device
        )

        if verbose:
            print(f"Models successfully loaded:")
            print(f" - Detection Model Config: {instance.det_config}")
            print(f" - Detection Model Checkpoint: {det_checkpoint}")
            print(f" - Segmentation Model Config: {instance.seg_config}")
            print(f" - Segmentation Model Checkpoint: {seg_checkpoint}")

        return instance

    def _prepare_model_files(
        self,
        model_folder: Path,
        model_name: str,
        default_config_path: Path,
        default_checkpoint_path: Path,
        verbose: bool = False
    ) -> (Path, Path):
        """
        Prepares the configuration and checkpoint files for a given model.

        Args:
            model_folder (Path): Base directory for models.
            model_name (str): Name of the model ('maskrcnn' or 'unet').
            default_config_path (Path): Default path to the model's config file.
            default_checkpoint_path (Path): Default path to the model's checkpoint file.
            verbose (bool): If True, prints detailed information during preparation.

        Returns:
            Tuple[Path, Path]: Paths to the model's config and checkpoint files.
        """
        model_dir = model_folder / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Prepare config file
        config_path = model_dir / default_config_path.name
        if not config_path.exists():
            shutil.copy(default_config_path, config_path)
            if verbose:
                print(f"Copied default config to {config_path}")

        # Prepare checkpoint file
        checkpoint_log = model_dir / 'last_checkpoint'
        if checkpoint_log.exists():
            with checkpoint_log.open('r') as f:
                checkpoint_relative_path = f.readline().strip()
            checkpoint_path = model_dir / checkpoint_relative_path
            if not checkpoint_path.exists():
                if verbose:
                    print(f"Checkpoint {checkpoint_path} not found. Using default checkpoint.")
                checkpoint_path = default_checkpoint_path
                shutil.copy(default_checkpoint_path, model_dir / default_checkpoint_path.name)
        else:
            if verbose:
                print(f"'last_checkpoint' not found in {model_dir}. Using default checkpoint.")
            checkpoint_path = default_checkpoint_path
            shutil.copy(default_checkpoint_path, model_dir / default_checkpoint_path.name)

        return config_path, checkpoint_path

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
        
        
        cfg.val_cfg = None
        cfg.val_dataloader = None


        # Modify metric config
        cfg.val_evaluator = None
        
        cfg.work_dir = os.path.join(model_folder, 'maskrcnn')
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

        print(f"config dir {self.config_dir}")
        cfg = Config.fromfile(os.path.join(self.config_dir, 'unet/unet.py'))
        

        crop_size = (256, 256)

        cfg.train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(0.5, 1.5),  # Adjusted to prevent aggressive downscaling
                scale=(256, 256),  # Increased scale size
                type='RandomResize'),
            dict(cat_max_ratio=0.75, crop_size=crop_size, type='RandomCrop'),in 
            dict(type='ResizeToMultiple', size_divisor=16),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ]


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

        cfg.train_dataloader.batch_size = 8  

        cfg.train_cfg.max_iters = 200
        cfg.train_cfg.val_interval = 200
        cfg.default_hooks.logger.interval = 10
        cfg.default_hooks.checkpoint.interval = 200

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
            
            mask = np.zeros_like(img[:, :, 0], dtype=float)  
            mask[y1:y2, x1:x2] = pred_res
            
            to_be_readout.append(mask)
            img_to_dump.append(pred_res)


        
        freq = hc.get_sampling_frequency(input_header)
        
        try:
            signal=readOut(num_samples, to_be_readout, nrows, sorted_bboxes, mV_pixel, freq)
        except Exception as e:
            print(f'Error in readout: {e}')
            return empty_signals_np.T if empty_signals_np.shape[1] > empty_signals_np.shape[0] else empty_signals_np


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