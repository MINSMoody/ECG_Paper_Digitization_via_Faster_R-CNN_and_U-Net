

from TeamCode.src import our_paths
from TeamCode.src.interface import AbstractDigitizationModel, AbstractClassificationModel
from TeamCode.src.verify_environment import verify_environment
import helper_code as hc
import numpy as np
import os
import cv2


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

def interpolate_nan(signal):
    if np.isnan(signal[0]):
        signal[0] = 0
    if np.isnan(signal[-1]):
        signal[-1] = 0
    nans, x = nan_helper(signal)
    if len(nans) == 0:
        return signal
    signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
    return signal

def upsample(signal, num_samples):
    x = np.linspace(0, len(signal)-1, num_samples)
    return interpolate.interp1d(np.arange(len(signal)), signal, kind='linear')(x)

def downsample(signal, num_samples):
    return upsample(signal, num_samples)

def filter_boxes(pred_bboxes, pred_labels, pred_scores, pred_masks):
    def bbox_iou(box1, box2):
        inter_rect_x1 = max(box1[0], box2[0])
        inter_rect_y1 = max(box1[1], box2[1])
        inter_rect_x2 = min(box1[2], box2[2])
        inter_rect_y2 = min(box1[3], box2[3])

        inter_area = max(0, inter_rect_x2 - inter_rect_x1 + 1) * max(0, inter_rect_y2 - inter_rect_y1 + 1)

        b1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        b2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        return inter_area / (b1_area + b2_area - inter_area)
    
    if len(pred_bboxes) > 13:
        keep = np.ones(len(pred_bboxes), dtype=bool)
        for i in range(len(pred_bboxes)):
            for j in range(i + 1, len(pred_bboxes)):
                if keep[j] and bbox_iou(pred_bboxes[i], pred_bboxes[j]) > 0.3:
                    keep[j] = pred_scores[i] > pred_scores[j]

        pred_bboxes = pred_bboxes[keep]
        pred_labels = pred_labels[keep]
        pred_masks = pred_masks[keep]
        pred_scores = pred_scores[keep]

    # Ensure there are exactly 13 bboxes
    if len(pred_bboxes) > 13:
        # Sort by scores in descending order
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_bboxes = pred_bboxes[sorted_indices][:13]
        pred_labels = pred_labels[sorted_indices][:13]
        pred_masks = pred_masks[sorted_indices][:13]
        pred_scores = pred_scores[sorted_indices][:13]
    # elif len(pred_bboxes) < 12:
    #     # Pad the remaining slots with the highest scoring bboxes
    #     top_indices = np.argsort(pred_scores)[::-1]
    #     while len(pred_bboxes) < 13:
    #         for idx in top_indices:
    #             if len(pred_bboxes) < 13:
    #                 pred_bboxes = np.append(pred_bboxes, [pred_bboxes[idx]], axis=0)
    #                 pred_labels = np.append(pred_labels, [pred_labels[idx]], axis=0)
    #                 pred_masks = np.append(pred_masks, [pred_masks[idx]], axis=0)
    #                 pred_scores = np.append(pred_scores, [pred_scores[idx]], axis=0)
    #             else:
    #                 break
    
    
    if len(pred_bboxes) != 13:
        warnings.warn(f"Expected 13 boxes, got {len(pred_bboxes)}", UserWarning)
    
    return pred_bboxes, pred_labels, pred_scores, pred_masks

def bboxes_sorting_12(bboxes, masks):
    bboxes_avg_H = list((bboxes[:,1]+bboxes[:,3])/2)
    bboxes_avg_W = list((bboxes[:,0]+bboxes[:,2])/2)
    bboxes_avg = np.array([bboxes_avg_H, bboxes_avg_W]).transpose()
    # sort by W
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
    ### now assign values row by row
    rowl = np.median(bboxes[0:3,0])
    rowr = np.median(bboxes[9:12,2])
    leadwidth = (rowr - rowl)/4
    bboxes[0,0] =  rowl
    bboxes[1,0] =  rowl
    bboxes[2,0] =  rowl
    bboxes[0,2] = rowl+leadwidth
    bboxes[1,2] = rowl+leadwidth
    bboxes[2,2] = rowl+leadwidth
    bboxes[3,0] = rowl+leadwidth
    bboxes[4,0] = rowl+leadwidth
    bboxes[5,0] = rowl+leadwidth
    bboxes[3,2] = rowl+leadwidth*2
    bboxes[4,2] = rowl+leadwidth*2
    bboxes[5,2] = rowl+leadwidth*2
    bboxes[6,0] = rowl+leadwidth*2
    bboxes[7,0] = rowl+leadwidth*2
    bboxes[8,0] = rowl+leadwidth*2
    bboxes[6,2] = rowl+leadwidth*3
    bboxes[7,2] = rowl+leadwidth*3
    bboxes[8,2] = rowl+leadwidth*3
    bboxes[9,0] = rowl+leadwidth*3
    bboxes[10,0] = rowl+leadwidth*3
    bboxes[11,0] = rowl+leadwidth*3
    bboxes[9,2] = rowr
    bboxes[10,2] = rowr
    bboxes[11,2] = rowr
    return bboxes, masks



def bboxes_sorting_13(bboxes, masks):
    bboxes_avg_H = list((bboxes[:,1]+bboxes[:,3])/2)
    bboxes_avg_W = list((bboxes[:,0]+bboxes[:,2])/2)
    bboxes_avg = np.array([bboxes_avg_H, bboxes_avg_W]).transpose()
    sortH_idx = bboxes_avg[:, 0].argsort()
    #first take the long leads and 3x4
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
    bboxes[1] = bbox_last
    masks[1] = mask_last
    # bboxes = np.append(bboxes, bbox_last.reshape((1,-1)), axis=0)
    # masks = np.append(masks, mask_last.reshape((1,mask_last.shape[0],mask_last.shape[1])), axis=0)
    # now manually assign the values
    rowl = np.median(bboxes[0:3,0])
    rowr = np.median(bboxes[9:12,2])
    leadwidth = (rowr - rowl)/4
    bboxes[0,0] =  rowl
    bboxes[1,0] =  rowl
    bboxes[2,0] =  rowl
    bboxes[0,2] = rowl+leadwidth
    #bboxes[1,2] = rowl+leadwidth
    bboxes[2,2] = rowl+leadwidth
    bboxes[3,0] = rowl+leadwidth
    bboxes[4,0] = rowl+leadwidth
    bboxes[5,0] = rowl+leadwidth
    bboxes[3,2] = rowl+leadwidth*2
    bboxes[4,2] = rowl+leadwidth*2
    bboxes[5,2] = rowl+leadwidth*2
    bboxes[6,0] = rowl+leadwidth*2
    bboxes[7,0] = rowl+leadwidth*2
    bboxes[8,0] = rowl+leadwidth*2
    bboxes[6,2] = rowl+leadwidth*3
    bboxes[7,2] = rowl+leadwidth*3
    bboxes[8,2] = rowl+leadwidth*3
    bboxes[9,0] = rowl+leadwidth*3
    bboxes[10,0] = rowl+leadwidth*3
    bboxes[11,0] = rowl+leadwidth*3
    bboxes[9,2] = rowr
    bboxes[10,2] = rowr
    bboxes[11,2] = rowr
    bboxes[1,2] = rowr if bboxes[1,2] < rowr else bboxes[1,2]
    return bboxes, masks


## helper functions end 

#format = [['I', 'aVR', 'V1', 'V4'], ['II', 'aVL', 'V2', 'V5'], ['III', 'aVF', 'V3', 'V6'], ['II']] # format is hardcoded for now
#format = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6']
#fullmode = 'II'

def crop_from_bbox(bbox, mask, mV_pixel):
    bbox = bbox.astype(int)
    ecg_segment = mask[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]

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


def readOut(header_path, masks, bboxes, mV_pixel):
    # bboxes = bboxes.astype(int)
    # print(bboxes.shape[0])
    

    
    with open(header_path, 'r') as f:
        input_header = f.read()

    num_samples = get_num_samples(input_header)
    # set the number of masks same as bboxes
    # masks = masks[:bboxes.shape[0], :, :]
    # case 1: less than 12 boxes, return empty signals
    # assert bboxes.shape[0] == masks.shape[0], f"Expected shape {bboxes.shape[0]}, got {masks.shape[0]}"
    if bboxes.shape[0] < 12:
        # failed to detect 12 leads
        empty_signals_np = np.full((12, num_samples), np.nan)
        lead_length = num_samples // 4
        empty_signals_np[0:3,0:lead_length] = 0
        empty_signals_np[3:6,lead_length:2*lead_length] = 0
        empty_signals_np[6:9,2*lead_length:3*lead_length] = 0
        empty_signals_np[9:12,3*lead_length:4*lead_length] = 0
        return empty_signals_np.T if empty_signals_np.shape[1] > empty_signals_np.shape[0] else empty_signals_np
        
    # case 2: 12 bboxes, filter boxes
    if bboxes.shape[0] == 12:
        bboxes, masks = bboxes_sorting_12(bboxes, masks)
        signals_np = np.full((12, num_samples), np.nan)
        for i in range(bboxes.shape[0]):
            signal = crop_from_bbox(bboxes[i], masks[i], mV_pixel)
            signal = interpolate_nan(signal) - np.mean(signal)
            signal = np.clip(signal, -2, 2)
            signallen = num_samples // 4
            signal = upsample(signal, signallen) if len(signal) < signallen else downsample(signal, signallen)
            start_idx = (num_samples // 4) * (i // 3)
            end_idx = start_idx + (num_samples // 4)
            signals_np[i, start_idx:end_idx] = signal
        signals_np = np.clip(signals_np, -2, 2)
        return signals_np.T if signals_np.shape[1] > signals_np.shape[0] else signals_np



    # case 3: at least 13 bboxes
    if bboxes.shape[0] >= 13:
        bboxes, masks = bboxes_sorting_13(bboxes, masks)
        signals_np = np.full((12, num_samples), np.nan)

        for i in range(12):
            signal = crop_from_bbox(bboxes[i], masks[i], mV_pixel)   
            signal = interpolate_nan(signal) - np.mean(signal)
            signal = np.clip(signal, -2, 2)
            signallen = num_samples if i == 1 else num_samples // 4
            signal = upsample(signal, signallen) if len(signal) < signallen else downsample(signal, signallen)
            if i == 1:
                signals_np[i] = signal
            else:
                start_idx = (num_samples // 4) * (i // 3)
                end_idx = start_idx + (num_samples // 4)
                signals_np[i, start_idx:end_idx] = signal

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

def generate_data(data_folder, model_folder, verbose):
    with open(os.path.join(model_folder, 'data_format.json'), 'r') as f:
        args_dict = json.load(f)
    args = Namespace(**args_dict)
    random.seed(args.seed)
    args.input_directory = data_folder
    args.output_directory = data_folder
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
            i += future.result()
            if args.max_num_images != -1 and i >= args.max_num_images:
                break

def prepare_data_for_training(data_folder, model_folder=None, verbose=False):

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
        # work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work_dir')
        # self.work_dir = None#work_dir
        self.config = None#os.path.join(work_dir, "maskrcnn_res101.py")
        self.model = None
        self.unet = None


    @classmethod
    def from_folder(cls, model_folder, verbose):
         # Create an instance of the class
        instance = cls()
        # instance.work_dir = model_folder
        instance.config = os.path.join(model_folder, "maskrcnn_res101.py")
        # Construct checkpoint path based on the model_folder parameter
        maskrcnn_checkpoint_file = os.path.join(model_folder, 'epoch_12.pth')

        # Initialize the model using instance-specific variables
        instance.model = init_detector(instance.config, maskrcnn_checkpoint_file, device=dev)
        instance.unet = ECGPredictor('resunet10', os.path.join(model_folder,'segmentation/segmentation_model.pth'), size=208, cbam=False)

        if verbose:
            print(f"Model loaded from {maskrcnn_checkpoint_file}")

        return instance

    def train_detection_model(self, cfg, model_folder, verbose):
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
            cv=5,
            resume_training=True,
            checkpoint_path=os.path.join(model_folder, 'segmentation_base_model.pth')
        )
        
        if verbose:
            print("Segmentation model training completed.")

    def train_model(self, data_folder, model_folder, verbose):
        multiprocessing.set_start_method('spawn')
        generate_data(data_folder, model_folder, verbose)
        prepare_data_for_training(data_folder, model_folder, verbose)
        if verbose:
            print('Training the digitization model...')
            print('Finding the Challenge data...')

        # Reduce the number of repeated compilations and improve
        # training speed.
        setup_cache_size_limit_of_dynamo()
        
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
        
        

        # Start the training in separate threads
        detection_thread = multiprocessing.Process(target=self.train_detection_model, args=(cfg, model_folder, verbose))
        segmentation_thread = multiprocessing.Process(target=self.train_segmentation_model, args=(data_folder, model_folder, verbose))
    
        detection_thread.start()
        segmentation_thread.start()

        # Wait for both threads to complete
        detection_thread.join()
        segmentation_thread.join()

        if verbose:
            print("Both detection and segmentation models have been trained.")
        
        

    
    def run_digitization_model(self, record, verbose):
        
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
        img_path = images[0]
        # print(f"Processing image: {img_path}")

        img = mmcv.imread(img_path,channel_order='rgb')
        # remove image gradient
        img = remove_image_gradients(img)
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
        
        to_be_readout = self.unet.run(image, bboxes) # float
        to_be_readout = np.where(to_be_readout > 0.3, True, False)
        # print(min(to_be_readout[0]), max(to_be_readout[0]))
        # assert len(to_be_readout) == 13, f"Expected 13 signals, got {len(to_be_readout)}"
        # assert len(bboxes) == len(to_be_readout), f"Expected {len(bboxes)} signals, got {len(to_be_readout)}"
        # assert to_be_readout[0].shape == (img.shape[0], img.shape[1]), f"Expected shape {(img.shape[0], img.shape[1])}, got {to_be_readout[0].shape}"
        
        import pickle
        to_dump = {'bboxes': bboxes, 'masks': to_be_readout, 'scores': scores, 'labels': labels, 'record': record}
        with open('to_dump.pkl', 'wb') as f:
            pickle.dump(to_dump, f)
        # assert to_be_readout.shape == masks.shape, f"Expected shape {masks.shape}, got {to_be_readout.shape}"
        # assert to_be_readout.shape[0] == 13, f"Expected 13 signals, got {to_be_readout.shape[0]}"
        # if masks.shape[0] < 13:
        #     # add empty masks
        #     empty_masks = np.zeros((13 - masks.shape[0], masks.shape[1], masks.shape[2]))
        #     masks = np.append(masks, empty_masks, axis=0)
        # to_be_readout = to_be_readout + masks
        
        mV_pixel = (25.4 *8.5*0.5)/(masks[0].shape[0]*5) #hardcoded for now
        # # mV_pixel = (1.5*25.4 *8.5*0.5)/(masks[0].shape[0]*5)
        header_path = hc.get_header_file(record)
        # # load gt masks for debuging:
        # directory_path = os.path.dirname(img_path)
        # img_name = os.path.splitext(os.path.basename(img_path))[0]
        # mask_path = os.path.join(directory_path, img_name + '_mask.png')
        # gt_mask_load = mmcv.imread(mask_path, flag='grayscale')
        # gt_masks = []
        # for bbox in bboxes:
        #     x1, y1, x2, y2 = bbox
        #     gt_mask = np.zeros_like(gt_mask_load)
        #     gt_mask[y1:y2, x1:x2] = gt_mask_load[y1:y2, x1:x2]
        #     # mmcv.imwrite(gt_mask, 'gt_mask.png')
        #     gt_masks.append(gt_mask)
        # gt_masks = np.array(gt_masks)
        
        # bbox_path = mask_path = os.path.join(directory_path, img_name + '.json')
        # with open(bbox_path, 'r') as file:
        #     settings = json.load(file)
        # for lead in settings['leads']:
        #     coords = lead['lead_bounding_box']
        #     x_coords = [coord[1] for coord in coords.values()]
        #     y_coords = [coord[0] for coord in coords.values()]
        #     x_min, x_max = min(x_coords), max(x_coords)
        #     y_min, y_max = min(y_coords), max(y_coords)
            
        
        # assert gt_masks.shape == to_be_readout.shape, f"Expected shape {to_be_readout.shape}, got {gt_masks.shape}"
        # signal=readOut(header_path, masks, bboxes, mV_pixel)
        signal=readOut(header_path, to_be_readout, bboxes, mV_pixel)
        # signal=readOut(header_path, gt_masks, bboxes, mV_pixel)
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