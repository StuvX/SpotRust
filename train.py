'''
Training script for the Bayesian models.
This file is used for training models. Please see the README for details about training.
'''
import sys
import os
import torch
from torchvision import transforms
import json
import argparse
import logging
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from sklearn.model_selection import KFold

from HRNet import HRNet, HRNet_dropout, HRNet_var

from utils import tsv_DataLoader, find_normals, threshold_tensor, varDiceLoss, bce_loss_var, gtImages, is_image_file, \
    pil_loader, normalize_tensor, mse_loss_var, comboLossVar, pltImage, pltDetected, enumerateWithEstimate

import datetime

import shutil
import hashlib

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Used for computeClassificationLoss and logMetrics to index into metrics_t/metrics_a
# METRICS_LABEL_NDX = 0
METRICS_LOSS_NDX = 1
# METRICS_FN_LOSS_NDX = 2
# METRICS_ALL_LOSS_NDX = 3

# METRICS_PTP_NDX = 4
# METRICS_PFN_NDX = 5
# METRICS_MFP_NDX = 6
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9

METRICS_SIZE = 10

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
#
def cleanup():
    dist.destroy_process_group()

class SegmentationTraining:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Segmentation training file")
        parser.add_argument('hypes', type=str, help='Path to hyperparameter json file [required]')
        parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--pretrained', type=str)
        self.args = parser.parse_args()

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.totalTrainingSamples_count = 0
        self.trn_writer = None
        self.val_writer = None

        # load hypes dictionary from json file
        with open(self.args.hypes, 'r') as f:
            logging.info("f: %s", f)
            self.hypes = json.load(f)

        self.normalize = self.initNormalise()

        self.optDict = {
            'Adam': torch.optim.Adam,
            'AdaDelta': torch.optim.Adadelta,
            'SGD': torch.optim.SGD,
            'RMSProp': torch.optim.RMSprop
        }

        # self.use_cuda = torch.cuda.is_available()
        # if self.use_cuda:
        #     log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
        #     self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        #     torch.cuda.set_device(self.device)
        #     torch.distributed.init_process_group(
        #         backend="nccl", init_method="env://",)
        # else:
        #     self.device = torch.device("cpu")
        #
        # self.segmentation_model = self.initModel()
        # self.segmentation_model = self.segmentation_model.to(self.device)
        # if self.use_cuda:
        #     self.segmentation_model = torch.nn.parallel.DistributedDataParallel(
        #         self.segmentation_model,
        #         find_unused_parameters=False,
        #         device_ids=[self.args.local_rank],
        #         output_device=self.args.local_rank
        #     )
        # self.optimizer, self.scheduler = self.initOptimizer()
        # self.normalize = self.initNormalise()
        # self.mask_transforms, self.train_transforms, self.val_transforms = self.initTransforms()
        # weight = torch.tensor(self.hypes['data']['class_weights'][1])
        # if self.use_cuda:
        #     weight = weight.to(self.device)
        #
        # if self.hypes['solver']['loss'] == 'xentropy':
        #     self.criterion = bce_loss_var(weight=weight)
        # elif self.hypes['solver']['loss'] == 'dice':
        #     self.criterion = varDiceLoss()
        # elif self.hypes['solver']['loss'] == 'combo':
        #     self.criterion = comboLossVar(lossWeight=0.15, classWeight=weight)
        # else:
        #     self.criterion = mse_loss_var()
        #
        # self.save_dir = os.path.join("saved", self.hypes['arch']['config'], time.strftime('%y-%m-%d[%H.%M]', time.localtime(time.time())))

    def initModel(self):
        # arch vars from hypes
        image_shape = self.hypes['arch']['image_shape']
        batch_size = self.hypes['solver']['batch_size']
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        if self.hypes['arch']['config'] == 'HRNet':
            segmentation_model = HRNet(config=self.hypes)
            print(self.args.pretrained)
            segmentation_model.init_weights(pretrained=self.args.pretrained)
        elif self.hypes['arch']['config'] == 'HRNet_do':
            segmentation_model = HRNet_dropout(config=self.hypes)
            print(self.args.pretrained)
            segmentation_model.init_weights(pretrained=self.args.pretrained)
        elif self.hypes['arch']['config'] == 'HRNet_var':
            segmentation_model = HRNet_var(config=self.hypes)
            print(self.args.pretrained)
            segmentation_model.init_weights(pretrained=self.args.pretrained)

        return segmentation_model

    def initOptimizer(self):
        model_optimizer = self.optDict[self.hypes['solver']['opt']](self.segmentation_model.parameters(),
                                                                    lr=self.hypes['solver']['learning_rate'])
        model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer,
                                                             patience=self.hypes['solver']['sched_patience'],
                                                             threshold=self.hypes['solver']['sched_thresh'],
                                                             verbose=True, factor=self.hypes['solver']['sched_factor'],
                                                             min_lr=1e-5)
        return model_optimizer, model_scheduler

    def initNormalise(self):
        if self.hypes['data']['pop_mean'] == [0, 0, 0]:

            print('...no pop mean or std found for normalizing...')
            print('finding mean and std now')
            pop_mean, pop_std0 = find_normals(self.hypes, self.hypes['data']['train_file'],
                                              self.hypes['arch']['image_shape'][1:3])
            self.hypes['data']['pop_mean'] = pop_mean.tolist()
            self.hypes['data']['pop_std0'] = pop_std0.tolist()
        else:
            pop_mean = self.hypes['data']['pop_mean']
            pop_std0 = self.hypes['data']['pop_std0']

        return transforms.Normalize(pop_mean, pop_std0)

    def initTransforms(self):
        #NOTE that the transform shouldn't use ToTensor, it is embedded within the tsv_DataLoader
        image_shape = self.hypes['arch']['image_shape']
        input_res = image_shape[1:3]

        mask_transforms = transforms.Compose([
            transforms.Resize(input_res, interpolation=TF.InterpolationMode.NEAREST)]
        )

        train_transforms = transforms.Compose([
            transforms.Resize(input_res, interpolation=TF.InterpolationMode.NEAREST)]
        )

        val_transforms = transforms.Compose(
             [transforms.Resize(input_res, interpolation=TF.InterpolationMode.NEAREST)])

        return mask_transforms, train_transforms, val_transforms

    def initKFoldDL(self):
        kFoldDS = tsv_DataLoader(self.hypes,
                                  self.hypes['data']['train_file'], normalize=self.normalize,
                                  img_transform=self.train_transforms,
                                  mask_transform=self.mask_transforms
                                  )

        return kFoldDS

    def initTrainDl(self):
        train_ds = tsv_DataLoader(self.hypes,
            self.hypes['data']['train_file'], normalize=self.normalize, img_transform=self.train_transforms,
            mask_transform=self.mask_transforms, random_crop=True
        )

        batch_size = self.hypes['solver']['batch_size']
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.hypes['solver']['num_workers'],
            pin_memory=self.use_cuda,
            drop_last=True,
            shuffle=True,
        )
        return train_dl

    def initValDl(self):
        val_ds = tsv_DataLoader(self.hypes,
            self.hypes['data']['val_file'], normalize=self.normalize, img_transform=self.val_transforms,
                                mask_transform=self.mask_transforms)

        batch_size = self.hypes['solver']['batch_size']
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.hypes['solver']['num_workers'],
            pin_memory=self.use_cuda,
            drop_last=True,
            shuffle=True,
        )

        return val_dl

    def initTestDl(self):
        test_ds = tsv_DataLoader(self.hypes, self.hypes['data']['test_file'],
                                normalize=self.normalize,
                                img_transform=self.val_transforms,
                                mask_transform=self.mask_transforms,
                                return_path=True)

        test_dl = DataLoader(
            test_ds,
            batch_size=1,
            num_workers=self.hypes['solver']['num_workers'],
            pin_memory=self.use_cuda,
            drop_last=True,
            shuffle=False,
        )

        return test_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            savedir = os.path.join('saved', self.hypes['arch']['config'], self.time_str)
            logging.info('Training start at {}, files will be saved to {}'.format(self.time_str, savedir))
            log_dir = os.path.join('runs', self.hypes['arch']['config'], self.time_str)
            logging.info('Tensorboard logs can be found at {}'.format(log_dir))

            self.trn_writer = SummaryWriter(log_dir=log_dir + '_trn_seg_' + self.hypes['arch']['config'] + str(self.args.local_rank))
            self.val_writer = SummaryWriter(log_dir=log_dir + '_val_seg_' + self.hypes['arch']['config'] + str(self.args.local_rank))

    def main(self, rank, world_size):
        log.info("Starting {}, {}".format(type(self).__name__, self.hypes))

        self.rank = rank
        self.device = torch.device(rank)
        self.use_cuda = True

        self.normalize = self.initNormalise()
        self.mask_transforms, self.train_transforms, self.val_transforms = self.initTransforms()

        # train_dl = self.initTrainDl() #currently set up to use k-fold cross validation
        # val_dl = self.initValDl()
        self.dataset = self.initKFoldDL()

        weight = torch.tensor(self.hypes['data']['class_weights'][1])
        if self.use_cuda:
            weight = weight.to(self.device)

        if self.hypes['solver']['loss'] == 'xentropy':
            self.criterion = bce_loss_var(weight=weight)
        elif self.hypes['solver']['loss'] == 'dice':
            self.criterion = varDiceLoss()
        elif self.hypes['solver']['loss'] == 'combo':
            self.criterion = comboLossVar(lossWeight=0.15, classWeight=weight)
        else:
            self.criterion = mse_loss_var()

        self.save_dir = os.path.join("saved", self.hypes['arch']['config'],
                                     time.strftime('%y-%m-%d[%H.%M]', time.localtime(time.time())))

        batch_size = self.hypes['solver']['batch_size']
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=self.k_folds, shuffle=True)

        for fold, (train_ids, test_ids) in enumerate(kfold.split(self.dataset)):
            print(f'FOLD {fold}')
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size, sampler=test_subsampler)

            model = self.initModel().to(self.device)
            self.segmentation_model = DDP(model, device_ids=[self.rank])

            self.optimizer, self.scheduler = self.initOptimizer()

            best_score = 0.0

            self.validation_cadence = self.hypes['logging']['eval_iter']
            for epoch_ndx in range(1, self.hypes['solver']['max_steps'] + 1):
                log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                    epoch_ndx,
                    self.hypes['solver']['max_steps'],
                    len(trainloader),
                    len(testloader),
                    self.hypes['solver']['batch_size'],
                    (torch.cuda.device_count() if self.use_cuda else 1),
                ))

                trnMetrics_t = self.doTraining(epoch_ndx, trainloader)
                self.logMetrics(epoch_ndx, 'train', trnMetrics_t)

                if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                    # if validation is wanted
                    valMetrics_t = self.doValidation(epoch_ndx, testloader)
                    score = self.logMetrics(epoch_ndx, 'test', valMetrics_t)
                    best_score = max(score, best_score)
                    log.info("Best score is: {}".format(best_score))
                    self.saveModel('seg', epoch_ndx, score, fold, score == best_score)

                    self.logImages(epoch_ndx, 'train', trainloader, fold)
                    self.logImages(epoch_ndx, 'test', testloader, fold)

            valMetrics_t = self.doValidation(epoch_ndx, testloader)
            score = self.logMetrics(epoch_ndx, 'test', valMetrics_t)
            best_score = max(score, best_score)
            log.info("Best score is: ".format(best_score))

            self.results[fold] = score

            self.saveModel('seg', fold, score, fold, score == best_score)

            self.logImages(epoch_ndx, 'train', trainloader, fold)
            self.logImages(epoch_ndx, 'test', testloader, fold)

            if self.hypes['data']['test_file'] is not None:
                # if self.args.local_rank == 0:
                test_dl = self.initTestDl()
                self.finalTest(test_dl)

            self.train_writer.close()
            self.train_writer.close()


    def doTraining(self, epoch_ndx, train_dl):
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        self.segmentation_model.train(True)
        train_dl.dataset.shuffleSamples()

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
            backoff=5
        )

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad(set_to_none=True)

            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g, epoch_ndx,
                                             self.hypes['solver']['threshold'])

            loss_var.backward()

            self.optimizer.step()

        self.totalTrainingSamples_count += trnMetrics_g.size(1)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            self.segmentation_model.eval()
            self.segmentation_model.train(False)

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
                backoff=4
            )
            val_loss = 0
            blanks = 0
            for batch_ndx, batch_tup in batch_iter:
                if batch_tup[1].max() == 0:
                    blanks += 1
                    continue
                val_loss += self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g, epoch_ndx)
            val_loss = val_loss / val_dl.batch_size
            self.scheduler.step(val_loss)
        return valMetrics_g.to('cpu')

    def finalTest(self, test_dl):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.args.local_rank}
        model_dict = torch.load(self.best_path, map_location)['state_dict']

        log.info('the best model is {}'.format(self.best_path))
        self.segmentation_model.load_state_dict(model_dict)

        self.segmentation_model.train(False)

        with torch.no_grad():
            for input, gt, index, filename in test_dl:
                print(filename)
                input = input.to(self.device)
                gt = gt.to(self.device)
                out = []
                var = []
                for j in range(self.hypes['solver']['n_MC']):
                    outDict = self.segmentation_model(input)
                    out.append(outDict['out'].squeeze().detach())
                    var.append(outDict['logVar'].squeeze().detach())

                outs = torch.stack(out)
                vars = torch.stack(var)

                filenm = os.path.splitext(os.path.basename(filename[0]))[0]
                recordName = os.path.join(self.save_dir, 'test_imgs', str(self.hypes['arch']['config'] + '_' + filenm))
                os.makedirs(os.path.dirname(recordName), mode=0o755, exist_ok=True)

                outMean = normalize_tensor(outs).mean(dim=0)
                output_tensor = torch.zeros_like(outMean, dtype=torch.uint8)
                output_tensor[outMean > self.hypes['solver']['threshold']] = int(1)

                f_score = gtImages(output_tensor, recordName, index, gt)
                log.info('image {} fscore = {:.3f}'.format(filenm, f_score))

                eps = torch.var(outs, dim=0).cpu().numpy()
                eps = normalize_tensor(eps)
                pltImage(eps, recordName, 'epistemic_uncertainty', label='uncertainty')
                log.info('eps image saved to {}'.format(str(recordName + '_epistemic_uncertainty.png')))

                ali_corr = (torch.mean(vars, dim=0)).cpu().numpy()
                ali_corr = normalize_tensor(ali_corr)
                pltImage(ali_corr, recordName, 'aleatoric_uncertainty', label='uncertainty')
                log.info('var image saved to {}'.format(str(recordName+'_aleatoric_uncertainty.png')))

                image = pil_loader(filename[0])
                pltDetected(image, outMean.cpu().numpy(), output_tensor.cpu().numpy(), recordName)

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g, epoch_ndx, classificationThreshold):
        input_t, label_t, index = batch_tup

        input_g = input_t.to(self.device, non_blocking=True).requires_grad_(True)
        label_g = label_t.to(input_g.device, non_blocking=True)

        if epoch_ndx < self.hypes['solver']['var_loss_epoch']:
            varOn = False
        else:
            varOn = True

        outDict = self.segmentation_model(input_g)
        if self.hypes['arch']['bayes'] is True:
            loss = self.criterion(outDict['out'], label_g, outDict['logVar'], varOn) + 0.1 * outDict['kl']
        else:
            loss = self.criterion(outDict['out'], label_g, outDict['logVar'], varOn)

        if torch.isnan(loss):
            print('loss has nan from input ', index)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)
        with torch.no_grad():
            if not self.segmentation_model.training:
                out = [outDict['out']]
                outVar = [outDict['logVar']]
                for j in range(self.hypes['solver']['n_MC']-1):
                    outDict = self.segmentation_model(input_g)
                    out.append(outDict['out'].detach())
                    outVar.append(outDict['logVar'].detach())
                prediction_g = torch.stack(out).mean(dim=0)
                logVar = torch.stack(outVar).mean(dim=0)
            else:
                prediction_g = outDict['out']
                logVar = outDict['logVar']

            prediction = normalize_tensor(prediction_g)
            predictionBool_g = threshold_tensor(prediction, classificationThreshold).to(torch.int)

            tp = (predictionBool_g * label_g).sum(dim=[1, 2, 3])
            fn = ((1 - predictionBool_g) * label_g).sum(dim=[1, 2, 3])
            fp = (predictionBool_g * (1 - label_g)).sum(dim=[1, 2, 3])

            if fp.min() < 0:
                print('*** fp < 0, index is {:.3f} ***'.format(index))

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        return loss

    def logImages(self, epoch_ndx, mode_str, dl):
        self.segmentation_model.eval()
        with torch.no_grad():
            tsv_path = os.path.abspath(os.path.dirname(self.hypes['data']['train_file']))
            loglist = [-1, -4, -8, -12, -16, -20]
            for i in range(6):
                file, gt = dl.dataset.imgs[loglist[i]]
                filename = os.path.join(tsv_path, file)
                maskname = os.path.join(tsv_path, gt)
                if is_image_file(filename):
                    image = pil_loader(filename)
                else:
                    pass
                if is_image_file(maskname):
                    mask = pil_loader(maskname)
                else:
                    pass
                if self.val_transforms is not None:
                    image = self.val_transforms(image)
                if self.mask_transforms is not None:
                    mask = self.val_transforms(mask)

                mask = np.asarray(mask)
                label_mask = np.zeros((mask.shape[0], mask.shape[1]))
                for ii, label in enumerate(self.hypes['data']['class_colours']):
                    label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = int(ii)

                mask = torch.Tensor(label_mask).to(torch.long)
                image = TF.to_tensor(image)
                image = image.to(self.device, non_blocking=True).unsqueeze(0)
                mask = mask.to(image.device, non_blocking=True).unsqueeze(0)

                if self.hypes['arch']['bayes'] is True:
                    if self.hypes['arch']['recon'] is True:
                        prediction_g, logVar, kl, recon = self.segmentation_model(image)
                    else:
                        prediction_g, logVar, kl = self.segmentation_model(image)
                else:
                    if self.hypes['arch']['recon'] is True:
                        prediction_g, logVar, recon = self.segmentation_model(image)
                    else:
                        prediction_g, logVar = self.segmentation_model(image)

                prediction_g = normalize_tensor(prediction_g.detach()).squeeze()
                output_tensor = torch.zeros_like(prediction_g)
                output_tensor[prediction_g > 0.7] = int(1)

                writer = getattr(self, mode_str + '_writer')

                gtImages(output_tensor, i, index=epoch_ndx, gt=mask, writer=getattr(self, mode_str + '_writer'))

                outName = str('{}_var'.format(i))
                logVar = logVar.squeeze()
                normal_logvar = normalize_tensor(logVar)
                writer.add_image(outName, normal_logvar, epoch_ndx, dataformats='HW')

                outName = str('{}_prediction'.format(i))
                writer.add_image(outName, prediction_g, epoch_ndx, dataformats='HW')

                writer.flush()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        if not np.isfinite(metrics_a).all():
            print('nan in loss')

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()

        metrics_dict['percent_all/tp'] = \
            sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = \
            sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = \
            sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100

        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
                                                   / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall = metrics_dict['pr/recall'] = sum_a[METRICS_TP_NDX] \
                                             / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
                                      / ((precision + recall) or 1)

        log.info(("E{} {:8} LR: {:.5f}, "
                  + "loss: {loss/all:.4f}, "
                  + "precision: {pr/precision:.4f}, "
                  + "recall: {pr/recall:.4f}, "
                  + "f1 score: {pr/f1_score:.4f}, "
                  + ""
                  ).format(
            epoch_ndx,
            mode_str,
            self.scheduler.optimizer.param_groups[0]['lr'],
            **metrics_dict,

        ))
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
                  ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'seg_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)

        writer.flush()

        score = metrics_dict['pr/f1_score']

        return score

    def saveModel(self, type_str, epoch_ndx, score, isBest=False):
        print(f"Saving model on rank {self.args.local_rank}")
        file_path = os.path.join(
            self.save_dir,
            '{}_{}.{}.pt'.format(
                epoch_ndx,
                self.time_str,
                epoch_ndx,
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        if isinstance(self.segmentation_model, torch.nn.DataParallel) or isinstance(self.segmentation_model, torch.nn.parallel.DistributedDataParallel):

            if self.args.local_rank == 0:
                log.info('=> saving checkpoint to {}'.format(
                    file_path))
                torch.save({
                    'epoch': epoch_ndx + 1,
                    'score': score,
                    'state_dict': self.segmentation_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, file_path)

            dist.barrier()

            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.args.local_rank}
            self.segmentation_model.load_state_dict(torch.load(file_path, map_location=map_location)['state_dict'])

        else:
            log.info('=> saving checkpoint to {}'.format(
                file_path + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch_ndx + 1,
                'score': score,
                'state_dict': self.segmentation_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, file_path)
        hypesout = os.path.join(self.save_dir, 'hypes.json')

        with open(hypesout, 'w+') as outfile:
            json.dump(self.hypes, outfile, sort_keys = True, indent = 4, ensure_ascii = False)

        log.info("Saved model params to {}".format(file_path))

        if isBest:
            self.best_path = os.path.join(
                self.save_dir,
                f'{epoch_ndx}_{self.time_str}.best.state')
            shutil.copyfile(file_path, self.best_path)

            log.info("Saved model params to {}".format(self.best_path))
            self.hypes.update({'model': self.best_path})

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

def segCall(rank, world_size):
    print(f"Running DDP training on rank {rank}.")
    setup(rank, world_size)
    SegmentationTraining().main(rank, world_size)

    cleanup()

if __name__ == '__main__':
    print('torch version ', torch.__version__)
    world_size = torch.cuda.device_count()
    assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"
    mp.spawn(segCall, args=(world_size,),
             nprocs=world_size,
             join=True)