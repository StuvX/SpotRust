import numpy as np
from PIL import Image
import torch
import os
import matplotlib.patches as mpatches
import logging
import time
import datetime

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import pandas as pd
import torchvision
import torchvision.transforms.functional as TF

from torchvision import transforms
from skimage import filters

from torch.nn.modules.loss import _Loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.JPG', '.PNG']

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)


    Returns:
        bool: True if the filename ends with one of given extensions

    """

    filename_lower = filename.lower()

    return any(filename_lower.endswith(ext) for ext in extensions)

def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:

        filename (string): path to a file

    Returns:

        bool: True if the filename ends with a known image extension

    """

    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img.convert('RGB')

class tsv_DataLoader(torch.utils.data.Dataset):
    """
    Load dataset from tab separated value
    This is useful for tensorboard visualization later
    This is set up for semantic segmentation
    """

    def __init__(self, hypes, tsv_file, img_transform=None, mask_transform=None, normalize=None, return_path=False,
                 random_crop=False):
        """
        Args:
            tsv_file (string): Path to csv file with relative image paths and labels.
            img_transform (callable, optional): Optional transforms to be applied to the image.
            mask_transform (callable, optional): Optional transforms to be applied to the mask.
        """
        super(tsv_DataLoader, self).__init__()
        self.tsv_path = os.path.abspath(os.path.dirname(tsv_file))
        self.series_list = pd.read_csv(tsv_file, sep='\t')
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.normalize = normalize

        self.imgs = self._make_dataset()
        self.colours = hypes['data']['class_colours']
        self.img_size = hypes['arch']['image_shape'][1:3]
        self.return_path = return_path
        self.random_crop = random_crop
        if random_crop:
            self.random_crop = torchvision.transforms.RandomResizedCrop(size=self.img_size, scale=(0.5, 1.2), ratio=(3. / 4., 4. / 3.))

    def __len__(self):
        return len(self.series_list)

    def __getitem__(self, idx):
        filename = os.path.join(self.tsv_path, self.series_list.iloc[idx, 0])
        maskname = os.path.join(self.tsv_path, self.series_list.iloc[idx, 1])
        if is_image_file(filename):
            image = pil_loader(filename)
            if is_image_file(maskname):
                mask = pil_loader(maskname)
                if self.random_crop:
                    try:
                        i, j, h, w = self.random_crop.get_params(image, [*self.random_crop.scale], [*self.random_crop.ratio])
                        image = TF.resized_crop(image, i, j, h, w, self.img_size,
                                                interpolation=TF.InterpolationMode.BILINEAR)
                        mask = TF.resized_crop(mask, i, j, h, w, self.img_size, interpolation=TF.InterpolationMode.NEAREST)
                    except:
                        image = TF.resize(image, self.img_size, Image.BILINEAR)
                        mask = TF.resize(mask, self.img_size, Image.NEAREST)
                        if idx == 1:
                            print('random_crop failed, resized')
                if self.mask_transform is not None:
                    mask = self.mask_transform(mask)
                if self.img_transform is not None:
                    image = self.img_transform(image)
                mask = np.asarray(mask)
                label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.long)
                for ii, label in enumerate(self.colours):
                    label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = int(ii)
                mask = torch.Tensor(label_mask).to(torch.float32).unsqueeze(dim=0)
                image = TF.to_tensor(image)
                if self.normalize is not None:
                    image = self.normalize(image)
                if self.return_path is True:
                    return image, mask, idx, filename
                else:
                    return image, mask, idx
            else:
                pass
        else:
            pass

    def _make_dataset(self):
        images = []
        for i in range(len(self.series_list)):
            if is_image_file(self.series_list['Image'].iloc[i]):
                path = self.series_list['Image'].iloc[i]
                mask = self.series_list['Mask'].iloc[i]
                item = (path, mask)
                images.append(item)
        return images

    def shuffleSamples(self):
        self.series_list = self.series_list.sample(frac = 1).reset_index(drop=True)

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)

def enumerateWithEstimate(
        iter,
        desc_str,
        start_ndx=0,
        print_ndx=4,
        backoff=None,
        iter_len=None,):
    """
    In terms of behavior, `enumerateWithEstimate` is almost identical
    to the standard `enumerate` (the differences are things like how
    our function returns a generator, while `enumerate` returns a
    specialized `<enumerate object at 0x...>`).
    However, the side effects (logging, specifically) are what make the
    function interesting.
    :param iter: `iter` is the iterable that will be passed into
        `enumerate`. Required.
    :param desc_str: This is a human-readable string that describes
        what the loop is doing. The value is arbitrary, but should be
        kept reasonably short. Things like `"epoch 4 training"` or
        `"deleting temp files"` or similar would all make sense.
    :param start_ndx: This parameter defines how many iterations of the
        loop should be skipped before timing actually starts. Skipping
        a few iterations can be useful if there are startup costs like
        caching that are only paid early on, resulting in a skewed
        average when those early iterations dominate the average time
        per iteration.
        NOTE: Using `start_ndx` to skip some iterations makes the time
        spent performing those iterations not be included in the
        displayed duration. Please account for this if you use the
        displayed duration for anything formal.
        This parameter defaults to `0`.
    :param print_ndx: determines which loop interation that the timing
        logging will start on. The intent is that we don't start
        logging until we've given the loop a few iterations to let the
        average time-per-iteration a chance to stablize a bit. We
        require that `print_ndx` not be less than `start_ndx` times
        `backoff`, since `start_ndx` greater than `0` implies that the
        early N iterations are unstable from a timing perspective.
        `print_ndx` defaults to `4`.
    :param backoff: This is used to how many iterations to skip before
        logging again. Frequent logging is less interesting later on,
        so by default we double the gap between logging messages each
        time after the first.
        `backoff` defaults to `2` unless iter_len is > 1000, in which
        case it defaults to `4`.
    :param iter_len: Since we need to know the number of items to
        estimate when the loop will finish, that can be provided by
        passing in a value for `iter_len`. If a value isn't provided,
        then it will be set by using the value of `len(iter)`.
    :return:
    """
    if iter_len is None:
        iter_len = len(iter)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    log.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))
    start_ts = time.time()
    for (current_ndx, item) in enumerate(iter):
        yield (current_ndx, item)
        if current_ndx == print_ndx:
            # ... <1>
            duration_sec = ((time.time() - start_ts)
                            / (current_ndx - start_ndx + 1)
                            * (iter_len-start_ndx)
                            )

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_ndx,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            print_ndx *= backoff

        if current_ndx + 1 == start_ndx:
            start_ts = time.time()

    log.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))

def find_normals(hypes, dl_file, input_res):
    mask_transforms = init_transforms = transforms.Compose([
        transforms.Resize(input_res, 0),
        transforms.CenterCrop(input_res)
    ])

    init_dataset = tsv_DataLoader(hypes, dl_file, init_transforms, mask_transforms)
    init_loader = torch.utils.data.DataLoader(init_dataset, shuffle=False, num_workers=10)

    pop_mean = []
    pop_std0 = []
    pop_std1 = []

    for image, mask, index in init_loader:
        numpy_image = image.numpy()

        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
        batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)

        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    pop_mean = torch.Tensor(pop_mean).mean(dim=0)
    pop_std0 = torch.Tensor(pop_std0).mean(dim=0)
    pop_std1 = torch.Tensor(pop_std1).mean(dim=0)

    print('pop mean is ', pop_mean)
    print('pop std0 is ', pop_std0)
    print('pop std1 is ', pop_std1)

    return pop_mean, pop_std0

def threshold_tensor(input_tensor, threshold=None):
    if threshold is None:
        if type(input_tensor) is torch.Tensor:
            threshold = filters.threshold_otsu(input_tensor.cpu().numpy())
        else:
            threshold = filters.threshold_otsu(input_tensor)
    output_tensor = torch.where(input_tensor > threshold, 1, 0)

    return output_tensor

def normalize_tensor(input_tensor):
    if input_tensor.max() == input_tensor.min(): #if the tensor is all the same value return ones.
        return input_tensor ** 0
    else:
        return (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min())

def process_images(hypes, savename, image_orig, detected, var, gt=None, out_res=None, threshold=None, printout=False):
    if out_res is not None:
        if len(out_res) == 1:
            input_res = [out_res[0], out_res[0]]
        elif len(out_res) > 2:
            print('out res must be length 2')
            exit()
        else:
            input_res = out_res
    else:
        input_res = hypes['arch']['image_shape'][1:3]
    confidence_mean = torch.mean(detected, dim=0).cpu().detach()
    thresholded = threshold_tensor(confidence_mean, threshold)
    extent = 0, image_orig.size[0], 0, image_orig.size[1]
    F_score = None

    if gt is not None:
        gt_transforms = transforms.Compose(
            [transforms.Resize(input_res, Image.NEAREST), ]
        )
        class_colors = hypes['data']['class_colours']
        if is_image_file(gt):
            with open(gt, 'rb') as f:
                gt = Image.open(f)
                if gt.mode == 'L':
                    gt = gt.convert('I')
                    color = False
                else:
                    gt = gt.convert('RGB')
                    color = True
        gt = gt_transforms(gt)

        if color is True:
            temp = np.asarray(gt)
            gt = np.zeros((temp.shape[0], temp.shape[1]), dtype=np.long)

            for ii, label in enumerate(class_colors):
                gt = np.where(np.all(temp==label, axis=-1), ii, gt)

        gt = np.asarray(gt)

        thresholded = thresholded.numpy().astype(int)
        gt = gt.astype(int)
        TP = (thresholded * gt).sum()
        FN = ((1-thresholded) * gt).sum()
        FP = (thresholded * (1 - gt)).sum()
        mIOU = TP / (TP + FN + FP)
        precision = np.divide(np.float(TP), (np.float(TP) + np.float(FP)))
        recall = np.divide(np.float(TP), (np.float(TP) + np.float(FN)))
        F_score = 2 * (precision * recall) / (precision + recall)
        if printout:
            print('threshold {:.2f}, TP is {:.0f}, FN is {:.0f}, FP is {:.0f}, mIoU: {:.3f}, F-score: {:.3f}'.format(threshold, TP, FN, FP, mIOU, F_score))

        if savename is not None:
            compared = 2 * gt + thresholded
            compared = compared.astype(int)
            colormapper = ListedColormap(['black', 'cyan', 'red', 'white'])

            tp = mpatches.Patch(color='white', label='True Positives', ec='black')
            fp = mpatches.Patch(color='cyan', label='False Positives', ec='black')
            fn = mpatches.Patch(color='red', label='False Negatives', ec='black')
            tn = mpatches.Patch(color='black', label='True Negatives')

            fig, ax = plt.subplots()

            ax.imshow(compared, cmap=colormapper, extent=extent)
            ax.set_title('accuracy map')
            ax.set_xticks([-1])
            ax.set_yticks([-1])
            ax.legend(handles=[tp, fp, fn, tn], loc='lower center', bbox_to_anchor=(0.5, -0.13),
                      fontsize=8, ncol=4, handlelength=1, columnspacing=1.)
            ax.set_xlabel('mIOU {:.2f} F-score {:.2f} '.format(mIOU, F_score))
            fig.savefig(str(savename + '_F-score_{:.2f}.png'.format(F_score)))
            plt.close(fig)

    if savename is not None:
        epi_corr = torch.std(detected,
                             dim=0).detach().cpu().numpy()  # take the epistemic uncertainty of corrosion as standard dev of stack

        ali_corr = (torch.mean(var, dim=0)).cpu().numpy()
        image_orig.putalpha(255)

        cmap = plt.get_cmap('cool_r')
        colors = cmap(confidence_mean)
        colors[:, :, -1] = thresholded

        shrinkfactor = 0.8
        fig, ax = plt.subplots()
        ax.imshow(image_orig, extent=extent)
        ax.imshow(colors, extent=extent)
        ax.set_title('detected')
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False, left=False)
        fig.savefig(str(savename + '_detected.png'))
        plt.close(fig)

        fig, ax = plt.subplots()
        corrTicks = [epi_corr.min(), epi_corr.max()]
        eps_corr = ax.imshow(epi_corr, extent=extent, cmap='plasma')
        ax.set_title('corrosion epistemic uncertainty')
        cb2 = plt.colorbar(eps_corr, ax=ax, orientation='vertical', shrink=shrinkfactor, pad=0.01,
                           label='uncertainty' , ticks=corrTicks)
        cb2.set_label('epistemic uncertainty', labelpad=-2)
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False, left=False)
        fig.savefig(str(savename + '_corr_epistemic.png'))
        plt.close(fig)

        ali_corrTicks = [ali_corr.min(), ali_corr.max()]
        fig, ax = plt.subplots()
        als_corr = ax.imshow(ali_corr, extent=extent, cmap='plasma')
        ax.set_title('aleatoric uncertainty')
        cb3 = plt.colorbar(als_corr, ax=ax, orientation='vertical', shrink=shrinkfactor, pad=0.01,
            ticks=ali_corrTicks)
        cb3.set_label('uncertainty', labelpad=-2)
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False, left=False)
        fig.savefig(str(savename + '_aleatoric.png'))
        plt.close(fig)

        #save detected figure
        colormapper = ListedColormap(['black', 'cyan'])

        corrosion = mpatches.Patch(color='cyan', label='corrosion', ec='black')
        background = mpatches.Patch(color='black', label='background')
        fig, ax = plt.subplots()
        ax.imshow(thresholded, cmap=colormapper, extent=extent)
        ax.set_title('prediction map')
        ax.set_xticks([-1])
        ax.set_yticks([-1])
        ax.legend(handles=[background, corrosion], loc='lower center', bbox_to_anchor=(0.5, -0.1),
                  fontsize=8, ncol=2, handlelength=1, columnspacing=1.)
        fig.savefig(str(savename + '_detection.png'))
        plt.close(fig)

    return F_score

def gtImages(input, recordName, index, gt=None, writer=None):
    if gt is not None:
        if writer is not None:
            outName = str('{}_accuracy'.format(recordName))
            outImg = torch.zeros([3, *input.shape])
            outImg[0] = gt
            outImg[1] = outImg[2] = input
            outImg = outImg.numpy()
            writer.add_image(outName, outImg, index, dataformats='CHW')
        else:
            compared = 2 * gt.to('cpu').numpy().squeeze() + input.to('cpu').numpy().squeeze()
            compared = compared / 3
            colormapper = ListedColormap(['black', 'cyan', 'red', 'white'])
            TP = (input * gt).sum()
            FN = ((1-input) * gt).sum()
            FP = (input * (1 - gt)).sum()
            mIOU = TP / (TP + FN + FP)
            precision = np.divide(np.float(TP), (np.float(TP) + np.float(FP)))
            recall = np.divide(np.float(TP), (np.float(TP) + np.float(FN)))
            F_score = (2 * (precision * recall)) / (precision + recall)
            print(
                'TP is {:.3f}, FN is {:.3f}, FP is {:.3f}, mIoU: {:.3f}, F-score: {:.3f}'.format(TP, FN, FP, mIOU,
                                                                                                 F_score))
            tp = mpatches.Patch(color='white', label='True Positives', ec='black')
            fp = mpatches.Patch(color='cyan', label='False Positives', ec='black')
            fn = mpatches.Patch(color='red', label='False Negatives', ec='black')
            tn = mpatches.Patch(color='black', label='True Negatives')  # , ec='black')

            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.imshow(compared, cmap=colormapper)
            ax.set_title('accuracy map')
            ax.set_xticks([-1])
            ax.set_yticks([-1])
            ax.legend(handles=[tp, fp, fn, tn], loc='lower center', bbox_to_anchor=(0.5, -0.14),
                      fontsize=8, ncol=4, handlelength=1, columnspacing=1.)
            ax.set_xlabel('mIOU {:.2f} F-score {:.2f} '.format(mIOU, F_score))

            ax.grid(False)
            plt.tight_layout()

            fig.savefig(str(recordName + '_F-score.png'))
            plt.close(fig)
            return F_score

class varDiceLoss(_Loss):
    def __init__(self, epsilon=1e-6, sigmoid=False):
        super(varDiceLoss, self).__init__()
        self.epsilon = epsilon
        if sigmoid:
            self.sig = torch.nn.Sigmoid()
        self.sigmoid = sigmoid

    def forward(self, prediction_g, label_g, logVar, varOn=True):
        if self.sigmoid:
            prediction_g = self.sig(prediction_g)

        numerator = 2 * (prediction_g * label_g) + self.epsilon
        denominator = prediction_g + label_g + self.epsilon

        fscore = numerator / denominator

        if varOn is False:
            return (1 - fscore.mean())+0*logVar.mean()
        else:
            varLoss = 0.5*(torch.exp(-logVar)*(1-fscore) + logVar)
            return torch.abs(varLoss.mean())

class diceLoss(_Loss):
    def __init__(self, epsilon=1E-6, sigmoid=False, reduction='mean'):
        super(diceLoss, self).__init__()
        self.epsilon = epsilon
        if sigmoid:
            self.sig = torch.nn.Sigmoid()
        self.sigmoid = sigmoid
        self.reduction = reduction

    def forward(self, prediction_g, label_g):
        if self.sigmoid:
            prediction_g = self.sig(prediction_g)
        if self.reduction == 'mean':
            diceLabel_g = (label_g).sum(dim=[1, 2, 3])
            dicePrediction_g = (prediction_g).sum(dim=[1, 2, 3])
            diceCorrect_g = (prediction_g * label_g).sum(dim=[1, 2, 3])
        elif self.reduction == 'none':
            diceCorrect_g = (prediction_g * label_g)
            dicePrediction_g = prediction_g
            diceLabel_g = label_g

        diceRatio_g = (2 * diceCorrect_g + self.epsilon) \
                      / (dicePrediction_g + diceLabel_g + self.epsilon)

        return 1 - diceRatio_g

class bce_loss_var(_Loss):
    def __init__(self, weight=torch.tensor(1)):
        super(bce_loss_var, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=weight)

    def forward(self, prediction_g, label_g, logVar, varOn=True):
        bce_loss = self.bce(prediction_g, label_g)
        if varOn is False:
            return bce_loss.mean()+0*logVar.mean()
        else:
            bce_var_loss = 0.5*(torch.exp(-logVar)*bce_loss+logVar)
            return torch.abs(bce_var_loss.mean())

class mse_loss_var(_Loss):
    def __init__(self, sigmoid=False):
        super(mse_loss_var, self).__init__()
        if sigmoid:
            self.sig = torch.nn.Sigmoid()
        self.sigmoid = sigmoid
        self.mse = torch.nn.MSELoss(reduction='none')

    def forward(self, prediction_g, label_g, logVar, varOn=True):
        if self.sigmoid:
            prediction_g = self.sig(prediction_g)
        mse_loss = self.mse(prediction_g, label_g)
        if varOn is False:
            return mse_loss.mean()+0*logVar.mean()
        else:
            mse_var_loss = 0.5*(torch.exp(-logVar)*mse_loss+logVar)
            return torch.abs(mse_var_loss.mean())

class comboLossVar(_Loss):
    '''Loss function that combines bce loss with diceloss'''
    def __init__(self, lossWeight, classWeight):
        super(comboLossVar, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=classWeight)
        self.dice = diceLoss(reduction='none', sigmoid=True)
        self.lossWeight = lossWeight

    def forward(self, prediction_g, label_g, logVar, varOn=True):
        bceLoss = self.bce(prediction_g, label_g)
        diceLoss = self.dice(prediction_g, label_g)
        comboLoss = self.lossWeight*bceLoss + (1-self.lossWeight)*diceLoss
        if varOn is False:
            return comboLoss.mean()+0*logVar.mean()
        else:
            comboLossVar = 0.5*(torch.exp(-1*logVar)*comboLoss+logVar)
            return torch.abs(comboLossVar.mean())

def plot_to_tensorboard(writer, fig, name, step):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = img / 255.0
    writer.add_image(name, img, step)
    plt.close(fig)

def pltImage(input_array, recordName, title, label):
    '''helper function to save plots to disc'''
    shrinkfactor = 0.8
    fig, ax = plt.subplots()
    img = ax.imshow(input_array, cmap='viridis')
    ax.set_title(title)
    cb3 = plt.colorbar(img, ax=ax, orientation='vertical', shrink=shrinkfactor, pad=0.01)  # ,
    cb3.set_label(label, labelpad=-2)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False,
                   left=False)
    fig.savefig(str(recordName + '_' + title + '.png'))
    plt.close('all')

def pltDetected(image, detected, thresholded, recordName):
    image.putalpha(255)
    extent = 0, image.size[0], 0, image.size[1]
    cmap = plt.get_cmap('viridis')
    colors = cmap(detected)
    colors[:, :, -1] = thresholded
    fig, ax = plt.subplots()
    ax.imshow(image, extent=extent)
    ax.imshow(colors, extent=extent)
    ax.set_title('detected')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False, left=False)
    fig.savefig(str(recordName + '_detected.png'))
    plt.close('all')
