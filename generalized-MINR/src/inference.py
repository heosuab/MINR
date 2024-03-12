import argparse
import datetime
import json
import numpy as np
import os
import cv2
import time
import yaml
import einops
import random
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from datetime import datetime
from models import create_model
from utils.setup import setup



def get_args_parser():
    parser = argparse.ArgumentParser('Inference', add_help=False)
    parser.add_argument('--ckpt_path', default=None, type=str, metavar='PATH')
    parser.add_argument('--save_dir', default=None, type=str, metavar='PATH')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Dataset parameters
    # parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
    #                     help='dataset path')
    parser.add_argument('--load_root', default='data', type=str, help='dataset root dir')
    parser.add_argument('--source_saved', default=None, type=str, help='saved root dir from mae')
    parser.add_argument('--cfg')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--masking', default=True)
    parser.add_argument('--arch', default='low_rank_modulated_transinr', type=str,)
    parser.add_argument('--ema', default=None, type=str, help='ema model path')
    parser.add_argument('--result_path', default='./results', type=str)
    parser.add_argument('--model_config', default='./configs/image/imagenette178_low_rank_modulated_transinr.yaml', type=str)
    parser.add_argument('--eval', default=True, action='store_true')

    args, extra_args = parser.parse_known_args()
    return parser, extra_args



def main(args, cfg, save_dir, extra_args):
    # base inform
    ckpt_file = args.ckpt_path.split('/')[-1]
    device = torch.device(args.device)
    
    # create test dataset, loader
    test_dataset = CustomDataset(args.source_saved, transform=None)
    print(f'Test dataset: len={len(test_dataset)}')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    # create and load model
    config, logger, writer = setup(args, extra_args)
    model, model_ema = create_model(config.arch, ema=config.arch.ema is not None)
    model = model.to(device)
    model_ema = model_ema.to(device) if model_ema is not None else None
    # model.load_state_dict(torch.load(args.ckpt_path)['model']['sd'])
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device).eval()
    print(f'Loaded model from {args.ckpt_path.split("/")[-1]}')

    # saving directories
    save_dirs = ['masked', 'pred', 'gt', 'pred+gt', 'mask']
    for d in save_dirs:
        Path(os.path.join(save_dir, d)).mkdir(parents=True, exist_ok=True)

    # inference
    for i, (img_name, masked_img, gt_img, mask_img) in enumerate(tqdm(test_loader)):
        masked_img, gt_img, mask_img = masked_img.to(device), gt_img.to(device), mask_img.to(device)
        B = masked_img.shape[0]

        # forward
        xs = masked_img
        gt = gt_img

        coord_inputs = model.sample_coord_input(xs, device=xs.device)

        pred = model(xs, coord_inputs)
        gt = gt.detach()
        
        masked_img = torch.einsum('bchw->bhwc', masked_img).detach().cpu()
        gt_img = torch.einsum('bchw->bhwc', gt_img).detach().cpu()
        mask_img = torch.einsum('bchw->bhwc', mask_img).detach().cpu()
        pred = torch.einsum('bchw->bhwc', pred).detach().cpu()
        gt = torch.einsum('bchw->bhwc', gt).detach().cpu()

        pasted_img = gt_img * (1-mask_img) + pred * mask_img

        # # forward
        # input = {'inp': masked_img}
        # hyponet = model(input)
        # coord = utils.make_coord_grid(gt_img.shape[-2:], (-1,1), device=gt_img.device)
        # coord = einops.repeat(coord, 'h w c -> b h w c', b=B)
        # pred = hyponet(coord).detach().cpu()

        # masked_img = torch.einsum('bchw->bhwc', masked_img).detach().cpu()
        # gt_img = torch.einsum('bchw->bhwc', gt_img).detach().cpu()
        # mask_img = torch.einsum('bchw->bhwc', mask_img).detach().cpu()

        # pasted_img = gt_img * (1-mask_img) + pred * mask_img

        for i in range(B):
            # save masked image
            img = (masked_img[i].numpy().squeeze()*255).astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_dir, save_dirs[0], f'{img_name[i]}.png'), img)

            # save reconstructed image
            img = (pred[i].numpy().squeeze()*255).astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_dir, save_dirs[1], f'{img_name[i]}.png'), img)

            # save gt image
            img = (gt_img[i].numpy().squeeze()*255).astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_dir, save_dirs[2], f'{img_name[i]}.png'), img)

            # save pasted image
            img = (pasted_img[i].numpy().squeeze()*255).astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_dir, save_dirs[3], f'{img_name[i]}.png'), img)

            # save mask image
            img = (mask_img[i].numpy().squeeze()*255).astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_dir, save_dirs[4], f'{img_name[i]}.png'), img)


def make_model(self, model_spec=None, load_sd=False):
    if model_spec is None:
        model_spec = self.cfg['model']
    model = models.make(model_spec, load_sd=load_sd)
    self.log(f'Model: #params={utils.compute_num_params(model)}')

    if self.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()
        model_ddp = DistributedDataParallel(model, device_ids=[self.rank])
    else:
        model.cuda()
        model_ddp = model
    self.model = model
    self.model_ddp = model_ddp


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def make_cfg(args):
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    def translate_cfg_(d):
        for k, v in d.items():
            if isinstance(v, dict):
                translate_cfg_(v)
            elif isinstance(v, str):
                if '$load_root$' in v:
                    d[k] = v.replace('$load_root$', args.load_root)
                if '$masking$' in v:
                    d[k] = v.replace('$masking$', args.masking)
    translate_cfg_(cfg)
    return cfg


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transforms.ToTensor()
        self.save_dirs = ['masked', 'gt', 'mask']
        self.masked_imgs = self.get_images(self.save_dirs[0])
        self.gt_imgs = self.get_images(self.save_dirs[1])
        self.mask_imgs = self.get_images(self.save_dirs[2])

    def __len__(self):
        return len(self.masked_imgs)

    def __getitem__(self, idx):
        img_name = self.masked_imgs[idx]

        masked_img = self.load_image(self.save_dirs[0], img_name)
        gt_img = self.load_image(self.save_dirs[1], img_name)
        mask_img = self.load_image(self.save_dirs[2], img_name)

        return (img_name, masked_img, gt_img, mask_img)

    def get_images(self, save_dir):
        return list(sorted(os.listdir(os.path.join(self.root, save_dir))))
    
    def load_image(self, save_dir, img_name):
        image = Image.open(os.path.join(self.root, save_dir, img_name))
        return self.transform(image)


if __name__ == '__main__':
    args, extra_args = get_args_parser()
    args = args.parse_args()
    cfg = make_cfg(args)

    seed_everything(args.seed)

    exp_name = args.ckpt_path.split('/')[-2]

    if args.save_dir:
        save_dir = os.path.join(args.save_dir, exp_name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
    main(args, cfg, save_dir, extra_args)
