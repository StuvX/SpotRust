from __future__ import division
import os
import torch
from torchvision import transforms
import argparse
import json
import torch.distributed as dist

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("MacOSX")

from utils import is_image_file, pil_loader, process_images, normalize_tensor
from HRNet import HRNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference on images for corrosion detection")
    parser.add_argument('--models', nargs='+', help='List of paths to directory with model file and hypes.json [required]')
    parser.add_argument('--image', type=str, help='Path to an image to run inference on')
    parser.add_argument('--gt', type=str, help='Optional path to ground truth file, will return confusion matrix.')
    parser.add_argument('--target', type=int, default=1, help='Optional target class, default to 1')
    parser.add_argument('--out_res', nargs='+', type=int, default=None, help='Optional output resolution')
    parser.add_argument('--thresh', type=float, default=None, help='Optional threshold')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dist.is_available():
        setup(0, 1)

    models = args.models

    #TODO pass the model savefiles as args
    modelos = ['fold0_epoch110.pt',
               'fold1_epoch90.pt',
               'fold2_epoch90.pt',
               'fold3_epoch110.pt',
               'fold4_epoch90.pt',
               'fold5_epoch90.pt',
               'fold6_epoch90.pt',
               'fold7_epoch90.pt',
               'fold8_epoch90.pt', ]

    print('image file is ', args.image)
    if is_image_file(args.image):
        image_orig = pil_loader(args.image)
    else:
        RuntimeError('image provided is not a supported image format')

    detector = []
    out = []
    seg = []
    var = []

    for model in models:
        hypesfile = os.path.join(model, 'hypes.json')
        with open(hypesfile, 'r') as f:
            hypes = json.load(f)

        image_shape = hypes['arch']['image_shape']
        num_classes = hypes['arch']['num_classes']
        class_colors = hypes['data']['class_colours']
        class_labels = hypes['data']['class_labels']
        overlay_colours = hypes['data']['overlay_colours']
        label_colours = hypes['data']['overlay_colours']
        batch_size = 1
        batch_shape = [batch_size] + image_shape
        weights_factor = hypes['data']['class_weights']

        if args.out_res is not None:
            if len(args.out_res) == 1:
                input_res = [args.out_res[0], args.out_res[0]]
            elif len(args.out_res) > 2:
                print('out res must be length 2')
                exit()
            else:
                input_res = args.out_res
        else:
            input_res = hypes['arch']['image_shape'][1:3]

        input_transforms = transforms.Compose(
            [transforms.Resize(input_res, 0),
             transforms.ToTensor(),
             transforms.Normalize(hypes['data']['pop_mean'], hypes['data']['pop_std0'])
             ]
        )

        channels = hypes['solver']['channels']

        image = input_transforms(image_orig)
        image = image.unsqueeze(dim=0).to(device)

        for modelo in modelos:
            modelfile = os.path.join(model, modelo)
            seg_model = HRNet(config=hypes)

            pretrained_dict = torch.load(modelfile, map_location=device)
            if 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']

            prefix = "module."
            keys = sorted(pretrained_dict.keys())
            for key in keys:
                if key.startswith(prefix):
                    newkey = key[len(prefix):]
                    pretrained_dict[newkey] = pretrained_dict.pop(key)
            # also strip the prefix in metadata if any.
            if "_metadata" in pretrained_dict:
                metadata = pretrained_dict["_metadata"]
                for key in list(metadata.keys()):
                    if len(key) == 0:
                        continue
                    newkey = key[len(prefix):]
                    metadata[newkey] = metadata.pop(key)
            seg_model.load_state_dict(pretrained_dict)
            seg_model.to(device)

            with torch.no_grad():
                outDict = seg_model(image)
                out.append(outDict['out'].squeeze().detach())
                var.append(outDict['logVar'].squeeze().detach())

    out = torch.stack(out)
    var = torch.stack(var)
    varmax = var.max()
    varmin = var.min()
    out = normalize_tensor(out)
    var = normalize_tensor(var) * (varmax - varmin)

    savename = os.path.join(os.getcwd(), 'figures',
                            str(hypes['arch']['config']), str(args.thresh),
                            str('multimodel_' + os.path.splitext(os.path.basename(args.image))[0]))

    os.makedirs(os.path.dirname(savename), mode=0o755, exist_ok=True)
    fscore = process_images(hypes, savename, image_orig, out, var, args.gt, input_res, threshold=args.thresh,
                            printout=True)
