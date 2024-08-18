import os
import json
import pickle
import random
import numpy as np

from imageio import imread
from TeamCode.src.ecg_train import ECGTrainer
from tqdm.auto import tqdm


class ECGSegment(object):

    def __init__(self, param_file, param_set):
        with open(param_file, 'r') as f:
            param_json = json.load(f)
        self.params = param_json[param_set]
        self.param_set = param_set
        self.random_state = self.params['random_state']
        return

    # def __load(self, data_dir):
    #     self.dataset = []
    #     for subject in os.listdir(data_dir):
    #         subject_dir = os.path.join(data_dir, subject)
    #         mask_path = os.path.join(subject_dir, 'mask.png')
    #         mask = imread(mask_path) / 255.0
    #         if np.sum(mask) == 0:
    #             continue
    #         tif = imread(os.path.join(subject_dir, 'tif.png')) / 255.0
    #         self.dataset.append([tif, mask])

    #     random.seed(self.random_state)
    #     random.shuffle(self.dataset)
    #     return
    def __load(self, image_path):
        self.dataset = []
        # counter = 0
        for filename in tqdm(os.listdir(image_path)):
            # print(filename)
            tif = imread(os.path.join(image_path, filename)) / 255.0
            # to gray scale
            # tif = np.dot(tif[..., :3], [0.2989, 0.5870, 0.1140])
            mask_path = os.path.join(os.path.dirname(image_path),'cropped_masks', filename)
            # print(mask_path)
            mask = imread(mask_path) / 255.0
            # print(tif.shape, mask.shape)
            
            self.dataset.append([tif, mask])
            # counter += 1
            # if counter == 1000:
            #     break

        random.seed(self.random_state)
        random.shuffle(self.dataset)
        return

    def __create_dir(self, dir_path):
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        return

    def __split(self, data, i):
        n = len(data)
        cv_subjects = int(n / self.cv) + 1

        idx1 = cv_subjects * (i - 1)
        idx2 = cv_subjects * i if i != self.cv else n

        trainset = data[0:idx1] + data[idx2:]
        validset = data[idx1:idx2]
        return trainset, validset

    def __train(self, trainset, validset, cv_model_dir, resume_training=False, checkpoint_path=None):
        self.__create_dir(cv_model_dir)
        trainer = ECGTrainer(**self.params)
        if resume_training:
            # Assuming ECGTrainer has a method to load the checkpoint and resume training
            trainer.resume_training(trainset, validset, cv_model_dir, checkpoint_path)
        else:
            # Normal training process
            trainer.run(trainset, validset, cv_model_dir)
        return

    def run(self, data_dir, models_dir, cv, resume_training, checkpoint_path):
        self.cv = cv
        self.__load(data_dir)

        for i in range(1, self.cv + 1):
            cv_model_dir = os.path.join(models_dir, self.param_set)
            trainset, validset = self.__split(self.dataset, i)

            with open('trainset.pkl', 'wb') as f:
                pickle.dump(trainset, f)
            with open('validset.pkl', 'wb') as f:
                pickle.dump(validset, f)

            self.__train(trainset, validset, cv_model_dir, resume_training, checkpoint_path)
            break
        return


def main(args):
    ecg = ECGSegment(
        param_file=args.param_file,
        param_set=args.param_set
    )
    ecg.run(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        cv=args.cv,
        resume_training=args.resume_training,
        checkpoint_path=args.checkpoint_path
    )
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
    parser.add_argument('--models-dir', '-m', type=str,
                        action='store', dest='models_dir',
                        help='Directory of models')
    parser.add_argument('--param-file', '-pf', type=str,
                        action='store', dest='param_file',
                        help='Json file of parameters')
    parser.add_argument('--param-set', '-ps', type=str,
                        action='store', dest='param_set',
                        help='Set of parameters')
    parser.add_argument('--cv', '-cv', type=int,
                        action='store', dest='cv',
                        help='Number of Splits for CV')
    parser.add_argument('--gpu', '-gpu', type=str,
                        action='store', dest='gpu',
                        help='Devoce NO. of GPU')
    parser.add_argument('--resume-training', '-rt', default=False,
                        action='store_true', dest='resume_training',
                        help='Resume Training')
    parser.add_argument('--checkpoint-path', '-cp', default=None, type=str,
                        action='store', dest='checkpoint_path',
                        help='Checkpoint Path')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
