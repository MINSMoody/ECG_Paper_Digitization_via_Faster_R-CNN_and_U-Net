import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tqdm import *
from ecg_models import build_model
from imageio import imread, imwrite
from torch.autograd import Variable


class ECGPredictor(object):

    def __init__(self, model_name, model_path, bboxes=None, size=128, cbam=False):

        self.model = build_model(model_name, cbam)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model.cuda()

        self.size = size
        self.bboxes = bboxes
        self.res_maps = []
        return

    def __pred(self, image):
        cimage = image.copy()
        cimage = np.expand_dims(cimage, 0)
        cimage = np.expand_dims(cimage, 0)
        image_tensor = torch.tensor(cimage).float()
        if self.cuda:
            image_tensor = image_tensor.cuda()

        pred = self.model(image_tensor)
        pred = torch.sigmoid(pred)
        pred = pred.squeeze(0).squeeze(0)
        prednp = pred.cpu().detach().numpy()
        return prednp
    
    def run(self, image):
        for _, bbox in enumerate(self.bboxes):
            res_map = self.__run(image, bbox)
            self.res_maps.append(res_map)
        return self.res_maps, self.bboxes


    def __run(self, image, bbox=None):
        if bbox is not None:
            x1, y1, x2, y2 = self.bbox
            image = image[y1:y2, x1:x2]
        h, w = image.shape
        p = self.size // 2
        image_pad = np.pad(image, ((p, p), (p, p)), mode='constant')

        rs = np.arange(0, h, p)
        cs = np.arange(0, w, p)
        grid = np.array(np.meshgrid(rs, cs)).T.reshape(-1, 2)

        count_map = np.zeros(image_pad.shape)
        res_map = np.zeros(image_pad.shape)


        for r, c in tqdm(grid, ncols=75):
            cimage = image_pad[r:r + self.size, c:c + self.size]

            prednp = self.__pred(cimage.copy())
            lr = self.__pred(np.fliplr(cimage))
            ud = self.__pred(np.flipud(cimage))
            rot90 = self.__pred(np.rot90(cimage, 2))
            prednp += np.fliplr(lr)
            prednp += np.flipud(ud)
            prednp += np.rot90(rot90, 2)
            prednp /= 4.0

            res_map[r:r + self.size, c:c + self.size] += prednp
            count_map[r:r + self.size, c:c + self.size] += 1
            
        # print(f"Prediction Stats - min: {res_map.min()}, max: {res_map.max()}, mean: {res_map.mean()}")
        # print(f"Prediction Stats - min: {count_map.min()}, max: {count_map.max()}, mean: {count_map.mean()}")
        res_map = res_map / count_map
        res_map = res_map[p:-p, p:-p]
        return res_map


def main(args):
    model_name = 'resunet10-combo'
    model_name = model_name + '-cbam' if args.cbam else model_name
    model_path = os.path.join(args.model_dir, 'model.pth')
    predictor = ECGPredictor('resunet10', model_path, 128, args.cbam)

    subjects = os.listdir(args.data_dir)
    for subject in tqdm(subjects, ncols=75):
        subj_dir = os.path.join(args.data_dir, subject)
        samples = os.listdir(subj_dir)
        subj_output_dir = os.path.join(args.output_dir, subject)
        if not os.path.isdir(subj_output_dir):
            os.makedirs(subj_output_dir)
        for sample in samples:
            image_path = os.path.join(subj_dir, sample)
            image = imread(image_path) / 255
            if image.ndim == 3 and image.shape[2] == 4:  # Image has 3 dimensions and the third dimension has 3 channels
                # Convert RGBA to grayscale
                image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
                
                # imwrite(os.path.join(subj_dir, "tif_L.png"), (image * 255).astype(np.uint8))

            res_map, boxes = predictor.run(image)
            res_map = (res_map * 255).astype(np.uint8)
            # Drawing bounding boxes on the image
            # for box in boxes:
            #     # Convert coordinates to integers
            #     start_point = (int(box[0]), int(box[1]))  # Top left corner
            #     end_point = (int(box[2]), int(box[3]))  # Bottom right corner
            #     color = (0, 255, 0)  # Green color in BGR
            #     thickness = 2  # Line thickness in px
                
            #     # Draw the rectangle on the original image
            #     image = cv2.rectangle(image, start_point, end_point, color, thickness)
            # # Normalize the image to the range 0-255 and convert to uint8
            # image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

            # # Saving the image with bounding boxes
            # boxed_image_file = os.path.join(subj_output_dir, f"boxed_{sample}")
            # imwrite(boxed_image_file, image)
            res_map_file = os.path.join(subj_output_dir, sample)

            imwrite(res_map_file, res_map)

    return


if __name__ == '__main__':
    import argparse
    import warnings

    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description='Segment ECG Signals in Report'
    )

    parser.add_argument('--program', '-p', type=str)
    parser.add_argument('--data-dir', '-d', type=str,
                        action='store', dest='data_dir',
                        help='Directory of training data')
    parser.add_argument('--output-dir', '-o', type=str,
                        action='store', dest='output_dir',
                        help='Directory of training data')
    parser.add_argument('--model-dir', '-m', type=str,
                        action='store', dest='model_dir',
                        help='Directory of model')
    parser.add_argument('--cbam', '-c', default=False,
                        action='store_true', dest='cbam',
                        help='Apply cbam')
    parser.add_argument('--gpu', '-g', type=str,
                        action='store', dest='gpu',
                        help='Devoce NO. of GPU')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
