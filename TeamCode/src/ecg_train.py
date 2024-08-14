import os
import torch
import numpy as np

from TeamCode.src.adabound import AdaBound
from TeamCode.src.ecg_losses import ComboLoss
from TeamCode.src.ecg_models import build_model
from TeamCode.src.ecg_loader import ECGDataLoader
from sklearn.metrics import f1_score


class ECGTrainer(object):

    def __init__(self, **kwargs):
        torch.set_num_threads(3)
        np.random.seed(kwargs['random_state'])
        torch.manual_seed(kwargs['random_state'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.crop = kwargs['crop']
        self.cbam = kwargs['cbam']
        self.model_name = kwargs['model']
        self.n_epochs = kwargs['n_epochs']
        self.batch_size = kwargs['batch_size']

        self.cuda = torch.cuda.is_available()
        self.model = self.__build_model()
        self.criterion = self.__get_criterion()
        self.opt, self.sche = self.__get_optimizer(**kwargs)
        return

    def __build_model(self):
        model = build_model(self.model_name, self.cbam)
        if self.cuda:
            model.cuda()
        return model

    def __get_criterion(self):
        criterion = ComboLoss(
            weights={'dice': 1, 'focal': 1},
            channel_weights=[1],
            channel_losses=[['dice', 'focal']],
            per_image=False
        )
        return criterion

    def resume_training(self, trainset, validset, model_dir, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint, strict=False)
        print("Checkpoint loaded successfully.")
        self.run(trainset, validset, model_dir)
        return
    
    def __get_optimizer(self, **kwargs):
        optimizer = AdaBound(
            amsbound=True, lr=kwargs['lr'],
            params=self.model.parameters(),
            weight_decay=kwargs['weight_decay']
        )

        scheduler = None
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 'min', factor=0.5,
        #     patience=10, verbose=True, min_lr=1e-5
        # )
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        return optimizer, scheduler

    def run(self, trainset, validset, model_dir):
        print('=' * 100)
        print('TRAINING MODEL - {}'.format(self.model_name))
        print('-' * 100 + '\n')

        model_path = os.path.join(model_dir, 'segmentation_model.pth')
        loader_params = {'batch_size': self.batch_size, 'crop': self.crop}
        dataloader = {
            'train': ECGDataLoader(trainset, **loader_params, augment=True).build(),
            'valid': ECGDataLoader(validset, **loader_params, augment=False).build()
        }

        best_loss = None
        for epoch in range(self.n_epochs):
            e_message = '[EPOCH {:0=3d}/{:0=3d}]'.format(epoch + 1, self.n_epochs)

            for phase in ['train', 'valid']:
                ep_message = e_message + '[' + phase.upper() + ']'
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                f1s, losses = [], []
                batch_num = len(dataloader[phase])
                for ith_batch, data in enumerate(dataloader[phase]):
                    patches, masks = [d.cuda() for d in data] if self.cuda else data
                    patches = patches.squeeze(0)
                    masks = masks.squeeze(0)

                    pred = self.model(patches)
                    loss = self.criterion(pred, masks)

                    pred = torch.sigmoid(pred)
                    pred[pred > 0.5] = 1
                    pred[pred <= 0.5] = 0
                    pred = pred.cpu().detach().numpy().flatten()
                    label = masks.cpu().detach().numpy().flatten()
                    f1 = f1_score(label, pred, average='macro')

                    f1s.append(f1)
                    losses.append(loss.item())

                    if phase == 'train':
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()

                    sr_message = '[STEP {:0=3d}/{:0=3d}]-[F1: {:.6f} LOSS: {:.6f}]'
                    sr_message = ep_message + sr_message
                    print(sr_message.format(ith_batch + 1, batch_num, f1, loss), end='\r')

                avg_f1 = np.mean(f1s)
                avg_loss = np.mean(losses)

                er_message = '[AVERAGE][F1: {:.6f} LOSS: {:.6f}]'
                er_message = '\n\033[94m' + ep_message + er_message + '\033[0m'
                print(er_message.format(avg_f1, avg_loss))

                if phase == 'valid':
                    if self.sche is not None:
                        # self.sche.step(avg_loss)
                        self.sche.step()

                    if best_loss is None or best_loss > avg_loss:
                        best_loss = avg_loss
                        best_loss_mtc = [epoch + 1, avg_f1, avg_loss]
                        torch.save(self.model.state_dict(), model_path)
                        print('[Best validation loss, model: {}]'.format(model_path))

                    print()

        res_message = 'VALIDATION PERFORMANCE: BEST LOSS' + '\n' \
            + '[EPOCH:{} F1: {:.6f} LOSS:{:.6f}]\n'.format(
                best_loss_mtc[0], best_loss_mtc[1], best_loss_mtc[2]) \
            + '=' * 100 + '\n'

        print(res_message)
        return
