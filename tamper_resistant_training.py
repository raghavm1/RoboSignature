# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
from copy import deepcopy
from omegaconf import OmegaConf
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image

import utils
import utils_img
import utils_model
sys.path.append('stablesignature/')
sys.path.append('stablesignature/src/')
sys.path.append('stablesignature/src/ldm/')
from src.ldm.models.autoencoder import AutoencoderKL
from src.ldm.models.diffusion.ddpm import LatentDiffusion
from src.loss.loss_provider import LossProvider

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--train_dir", type=str, help="Path to the training data directory", required=True)
    aa("--val_dir", type=str, help="Path to the validation data directory", required=True)
    aa("--atrain_dir", type=str, help="Path to the atrain set", required=False, default="train2014_10000")
    group = parser.add_argument_group('Model parameters')
    aa("--ldm_config", type=str, default="sd/stable-diffusion-v-1-4-original/v1-inference.yaml", help="Path to the configuration file for the LDM model") 
    aa("--ldm_ckpt", type=str, default="sd/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt", help="Path to the checkpoint file for the LDM model") 
    aa("--msg_decoder_path", type=str, default= "models/hidden/dec_48b_whit.torchscript.pt", help="Path to the hidden decoder for the watermarking model")
    aa("--num_bits", type=int, default=48, help="Number of bits in the watermark")
    aa("--redundancy", type=int, default=1, help="Number of times the watermark is repeated to increase robustness")
    aa("--decoder_depth", type=int, default=8, help="Depth of the decoder in the watermarking model")
    aa("--decoder_channels", type=int, default=64, help="Number of channels in the decoder of the watermarking model")

    group = parser.add_argument_group('Training parameters')
    aa("--batch_size", type=int, default=4, help="Batch size for training")
    aa("--img_size", type=int, default=256, help="Resize images to this size")
    aa("--loss_i", type=str, default="watson-vgg", help="Type of loss for the image loss. Can be watson-vgg, mse, watson-dft, etc.")
    aa("--loss_w", type=str, default="bce", help="Type of loss for the watermark loss. Can be mse or bce")
    aa("--lambda_i", type=float, default=0.2, help="Weight of the image loss in the total loss")
    aa("--lambda_w", type=float, default=1.0, help="Weight of the watermark loss in the total loss")
    aa("--optimizer", type=str, default="AdamW,lr=5e-4", help="Optimizer and learning rate for training")
    aa("--steps", type=int, default=100, help="Number of steps to train the model for")
    aa("--warmup_steps", type=int, default=20, help="Number of warmup steps for the optimizer")
    aa("--finetuned_ckpt", type=str, help="the first non-adversarially finetuned LDM ckpt")

    group = parser.add_argument_group('Logging and saving freq. parameters')
    aa("--log_freq", type=int, default=10, help="Logging frequency (in steps)")
    aa("--save_img_freq", type=int, default=1000, help="Frequency of saving generated images (in steps)")

    group = parser.add_argument_group('Experiments parameters')
    aa("--num_keys", type=int, default=1, help="Number of fine-tuned checkpoints to generate")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--seed", type=int, default=0)
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")
    aa("--strategy", type=int, default=0, help="Watermarking strategy (0: original, 1: randomfrompaper, 2: gradual randomness)")
    aa("--outer_steps", type=int, default=100,help="Outer loop steps")
    aa("--inner_steps", type=int, default=10,help="Inner loop steps")

    return parser

@torch.no_grad()
def val(data_loader: Iterable, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, msg_decoder: nn.Module, vqgan_to_imnet:nn.Module, key: torch.Tensor, params: argparse.Namespace):
    header = 'Eval'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.eval()
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        
        imgs = imgs.to(device)

        imgs_z = ldm_ae.encode(imgs) # b c h w -> b z h/f w/f
        imgs_z = imgs_z.mode()

        imgs_d0 = ldm_ae.decode(imgs_z) # b z h/f w/f -> b c h w
        imgs_w = ldm_decoder.decode(imgs_z) # b z h/f w/f -> b c h w
        
        keys = key.repeat(imgs.shape[0], 1)

        log_stats = {
            "iteration": ii,
            "psnr": utils_img.psnr(imgs_w, imgs_d0).mean().item(),
            # "psnr_ori": utils_img.psnr(imgs_w, imgs).mean().item(),
        }
        attacks = {
            'none': lambda x: x,
            'crop_01': lambda x: utils_img.center_crop(x, 0.1),
            'crop_05': lambda x: utils_img.center_crop(x, 0.5),
            'rot_25': lambda x: utils_img.rotate(x, 25),
            'rot_90': lambda x: utils_img.rotate(x, 90),
            'resize_03': lambda x: utils_img.resize(x, 0.3),
            'resize_07': lambda x: utils_img.resize(x, 0.7),
            'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
            'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            'jpeg_80': lambda x: utils_img.jpeg_compress(x, 80),
            'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
        }
        for name, attack in attacks.items():
            imgs_aug = attack(vqgan_to_imnet(imgs_w))
            decoded = msg_decoder(imgs_aug) # b c h w -> b k
            diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            word_accs = (bit_accs == 1) # b
            log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
            log_stats[f'word_acc_{name}'] = torch.mean(word_accs.type(torch.float)).item()
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})

        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs),0,1), os.path.join(params.imgs_dir, f'{ii:03}_val_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0),0,1), os.path.join(params.imgs_dir, f'{ii:03}_val_d0.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w),0,1), os.path.join(params.imgs_dir, f'{ii:03}_val_w.png'), nrow=8)
    
    print("Averaged {} stats:".format('eval'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(params):

    # Set seeds for reproductibility 
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    
    # Print the arguments
    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(json.dumps(vars(params))))

    # Create the directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    imgs_dir = os.path.join(params.output_dir, 'imgs')
    params.imgs_dir = imgs_dir
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir, exist_ok=True)

    # Loads LDM auto-encoder models
    print(f'>>> Building LDM model with config {params.ldm_config} and weights from {params.ldm_ckpt}...')
    config = OmegaConf.load(f"{params.ldm_config}")
    ldm_ae: LatentDiffusion = utils_model.load_model_from_config(config, params.ldm_ckpt)
    ldm_ae: AutoencoderKL = ldm_ae.first_stage_model
    if(params.strategy!=0):
        print("Finetuned ckpt is at - ", params.finetuned_ckpt)
        state_dict = torch.load("/scratch/rm6418/output/checkpoint_000.pth")['ldm_decoder']
        msg = ldm_ae.load_state_dict(state_dict, strict=False)
        print(f"loaded LDM decoder state_dict with message\n{msg}")
        print("you should check that the decoder keys are correctly matched")
    ldm_ae.eval()
    ldm_ae.to(device)
    
    # Loads hidden decoder
    print(f'>>> Building hidden decoder with weights from {params.msg_decoder_path}...')
    if 'torchscript' in params.msg_decoder_path: 
        msg_decoder = torch.jit.load(params.msg_decoder_path).to(device)
        # already whitened
        
    else:
        msg_decoder = utils_model.get_hidden_decoder(num_bits=params.num_bits, redundancy=params.redundancy, num_blocks=params.decoder_depth, channels=params.decoder_channels).to(device)
        ckpt = utils_model.get_hidden_decoder_ckpt(params.msg_decoder_path)
        print(msg_decoder.load_state_dict(ckpt, strict=False))
        msg_decoder.eval()

        # whitening
        print(f'>>> Whitening...')
        with torch.no_grad():
            # features from the dataset
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            loader = utils.get_dataloader(params.train_dir, transform, batch_size=16, collate_fn=None)
            ys = []
            for i, x in enumerate(loader):
                x = x.to(device)
                y = msg_decoder(x)
                ys.append(y.to('cpu'))
            ys = torch.cat(ys, dim=0)
            nbit = ys.shape[1]
            
            # whitening
            mean = ys.mean(dim=0, keepdim=True) # NxD -> 1xD
            ys_centered = ys - mean # NxD
            cov = ys_centered.T @ ys_centered
            e, v = torch.linalg.eigh(cov)
            L = torch.diag(1.0 / torch.pow(e, exponent=0.5))
            weight = torch.mm(L, v.T)
            bias = -torch.mm(mean, weight.T).squeeze(0)
            linear = nn.Linear(nbit, nbit, bias=True)
            linear.weight.data = np.sqrt(nbit) * weight
            linear.bias.data = np.sqrt(nbit) * bias
            msg_decoder = nn.Sequential(msg_decoder, linear.to(device))
            torchscript_m = torch.jit.script(msg_decoder)
            params.msg_decoder_path = params.msg_decoder_path.replace(".pth", "_whit.pth")
            print(f'>>> Creating torchscript at {params.msg_decoder_path}...')
            torch.jit.save(torchscript_m, params.msg_decoder_path)
    
    msg_decoder.eval()
    nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(device)).shape[-1]

    # Freeze LDM and hidden decoder
    for param in [*msg_decoder.parameters(), *ldm_ae.parameters()]:
        param.requires_grad = False

    # Loads the data
    print(f'>>> Loading data from {params.train_dir} and {params.val_dir}...')
    vqgan_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])
    atrain_loader = utils.get_dataloader(params.atrain_dir, vqgan_transform, params.batch_size, num_imgs=params.batch_size*params.steps, shuffle=True, num_workers=4, collate_fn=None)
    train_loader = utils.get_dataloader(params.train_dir, vqgan_transform, params.batch_size, num_imgs=params.batch_size*params.steps, shuffle=True, num_workers=4, collate_fn=None)
    val_loader = utils.get_dataloader(params.val_dir, vqgan_transform, params.batch_size*4, num_imgs=1000, shuffle=False, num_workers=4, collate_fn=None)
    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])
    
    # Create losses
    print(f'>>> Creating losses...')
    print(f'Losses: {params.loss_w} and {params.loss_i}...')
    if params.loss_w == 'mse':        
        loss_w = lambda decoded, keys, temp=10.0: torch.mean((decoded*temp - (2*keys-1))**2) # b k - b k
    elif params.loss_w == 'bce':
        loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded*temp, keys, reduction='mean')
    else:
        raise NotImplementedError
    
    if params.loss_i == 'mse':
        loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs)**2)
    elif params.loss_i == 'watson-dft':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif params.loss_i == 'watson-vgg':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif params.loss_i == 'ssim':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    else:
        raise NotImplementedError

    for ii_key in range(params.num_keys):

        # Creating key
        print(f'\n>>> Creating key with {nbit} bits...')
        if(params.strategy==0):
            key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=device)
            key_str = "".join([ str(int(ii)) for ii in key.tolist()[0]])
        else:
            key=torch.tensor([[1,1,1,0,1,0,1,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,1,1,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,1,1,1]], dtype=torch.float32, device=device)
            key_str="111010110101000001010111010011010100010000100111"
        print(f'Key: {key_str}')

        # Copy the LDM decoder and finetune the copy
        ldm_decoder = deepcopy(ldm_ae)
        ldm_decoder.encoder = nn.Identity()
        ldm_decoder.quant_conv = nn.Identity()
        ldm_decoder.to(device)
        for param in ldm_decoder.parameters():
            param.requires_grad = True
        optim_params = utils.parse_params(params.optimizer)
        optimizer = utils.build_optimizer(model_params=ldm_decoder.parameters(), **optim_params)

        atrain_dataset = atrain_loader.dataset

        # get 80% split
        atrain_size = int(0.8 * len(atrain_dataset))
        dtr_size = len(atrain_dataset) - atrain_size
        train_dataset, val_dataset = random_split(atrain_dataset, [atrain_size, dtr_size])
        atrain_loader = DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=4,
            collate_fn=None,
        )

        dtr_loader = DataLoader(
            val_dataset,
            batch_size=params.batch_size,
            shuffle=False,  # No shuffle for validation
            num_workers=4,
            collate_fn=None,
        )


        # Training loop
        print(f'>>> Training...')
        
        tr_decoder, train_stats_arr = tamper_train(atrain_loader, dtr_loader, optimizer, loss_w, loss_i, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, params)
        val_stats = val(val_loader, ldm_ae, tr_decoder, msg_decoder, vqgan_to_imnet, key, params)
        log_stats = {
            'key': key_str,
            # Aggregate train stats from train_stats_arr
            **{f'train_{k}_{i}': v for i, train_stats in enumerate(train_stats_arr) for k, v in train_stats.items()},
            # Log validation stats as-is
            **{f'val_{k}': v for k, v in val_stats.items()},
        }
        save_dict = {
            'ldm_decoder': tr_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        }

        # Save checkpoint
        if(params.strategy==0):
            torch.save(save_dict, os.path.join(params.output_dir, f"checkpoint_{ii_key:03d}.pth"))
            with (Path(params.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            with (Path(params.output_dir) / "keys.txt").open("a") as f:
                f.write(os.path.join(params.output_dir, f"checkpoint_{ii_key:03d}.pth") + "\t" + key_str + "\n")
            print('\n')
        else:
            torch.save(save_dict, os.path.join(params.output_dir, f"checkpointtar_{ii_key:03d}_{params.strategy}.pth"))
            with (Path(params.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            with (Path(params.output_dir) / "keys.txt").open("a") as f:
                f.write(os.path.join(params.output_dir, f"checkpointtar_{ii_key:03d}_{params.strategy}.pth") + "\t" + key_str + "\n")
            print('\n')

def train(data_loader: Iterable, optimizer: torch.optim.Optimizer, loss_w: Callable, loss_i: Callable, ldm_ae: AutoencoderKL, ldm_decoder:AutoencoderKL, msg_decoder: nn.Module, vqgan_to_imnet:nn.Module, key: torch.Tensor, params: argparse.Namespace):
    header = 'Train'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.train()
    base_lr = optimizer.param_groups[0]["lr"]
    current_key = key.clone()
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device)
        if(params.strategy==1):
            key = torch.randint(0, 2, (1,params.num_bits), dtype=torch.float32, device=device)
        if(params.strategy==2):
            num_bits_to_flip = min(ii + 1, params.num_bits)
            indices_to_flip = torch.randperm(params.num_bits)[:num_bits_to_flip]
            current_key[:,indices_to_flip] = 1 - current_key[:,indices_to_flip]
            key = current_key
        keys = key.repeat(imgs.shape[0], 1)
        
        utils.adjust_learning_rate(optimizer, ii, params.steps, params.warmup_steps, base_lr)
        # encode images
        imgs_z = ldm_ae.encode(imgs) # b c h w -> b z h/f w/f
        imgs_z = imgs_z.mode()

        # decode latents with original and finetuned decoder
        imgs_d0 = ldm_ae.decode(imgs_z) # b z h/f w/f -> b c h w
        imgs_w = ldm_decoder.decode(imgs_z) # b z h/f w/f -> b c h w

        # extract watermark
        decoded = msg_decoder(vqgan_to_imnet(imgs_w)) # b c h w -> b k

        # compute loss
        lossw = loss_w(decoded, keys)
        lossi = loss_i(imgs_w, imgs_d0)
        loss = params.lambda_w * lossw + params.lambda_i * lossi

        # optim step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # log stats
        diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        word_accs = (bit_accs == 1) # b
        log_stats = {
            "iteration": ii,
            "loss": loss.item(),
            "loss_w": lossw.item(),
            "loss_i": lossi.item(),
            "psnr": utils_img.psnr(imgs_w, imgs_d0).mean().item(),
            # "psnr_ori": utils_img.psnr(imgs_w, imgs).mean().item(),
            "bit_acc_avg": torch.mean(bit_accs).item(),
            "word_acc_avg": torch.mean(word_accs.type(torch.float)).item(),
            "lr": optimizer.param_groups[0]["lr"],
        }
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})
        if ii % params.log_freq == 0:
            print(json.dumps(log_stats))
        
        # save images during training
        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs),0,1), os.path.join(params.imgs_dir, f'{ii:03}_train_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0),0,1), os.path.join(params.imgs_dir, f'{ii:03}_train_d0.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w),0,1), os.path.join(params.imgs_dir, f'{ii:03}_train_w.png'), nrow=8)
    
    print("Averaged {} stats:".format('train'), metric_logger)
    return ldm_decoder, {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def get_weights_mean(base_model_outputs, model_outputs):
    """
        modified from the original paper to support dicts
    """
    return torch.mean(
        torch.stack(
            [
                (torch.norm(base_model_outputs[key] - model_outputs[key], dim=-1)).mean()
                for key in base_model_outputs.keys() & model_outputs.keys()
            ]
        )
    )

# these two functions to register and save hidden states in the LDMs
def save_hidden_states(target_dict, name):
    def hook(module, input, output):
        target_dict[name] = output.detach()
    return hook

def register_hooks(model, model_name, target_dict):
    for name, layer in model.named_modules():
        layer.register_forward_hook(save_hidden_states(target_dict, f"{name}"))

def tamper_train(atrain_loader: Iterable, dtr_loader: Iterable, optimizer: torch.optim.Optimizer, loss_w: Callable, loss_i: Callable, ldm_ae: AutoencoderKL, ldm_decoder:AutoencoderKL, msg_decoder: nn.Module, vqgan_to_imnet:nn.Module, key: torch.Tensor, params: argparse.Namespace):
    vqgan_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load all datasets needed
    Retain_loader = utils.get_dataloader(params.train_dir, vqgan_transform, params.batch_size, num_imgs=params.batch_size*params.steps, shuffle=True, num_workers=4, collate_fn=None)    
    
    attacked_decoder = deepcopy(ldm_decoder) # original decoder, to be used for adversarial fine-tuning
    og_decoder = deepcopy(ldm_decoder) # create copy for comparison
    hidden_states_ldm = {}
    hidden_states_og = {}
    register_hooks(ldm_decoder, "ldm_decoder", hidden_states_ldm)
    register_hooks(og_decoder, "og_decoder", hidden_states_og)
    og_key = key.clone()
    l_tr_grad_scale=4.0 # lambda_tr
    l_retain_grad_scale=1.0 # lambda term for retaining
    train_stats_arr = []
    for i in range(params.outer_steps):
        g_tr=0 # for accumulating gradients
        
        x_tr = next(iter(dtr_loader))[:params.batch_size].to(device) # sample x_tr from d_tr; TODO: need to improve
        
        attacked_decoder = deepcopy(ldm_decoder)
        for k in range(params.inner_steps):

            # entire adversarial fine-tuning step
            attacked_decoder, train_stats = train(data_loader=atrain_loader, optimizer=optimizer, loss_w = loss_w, loss_i = loss_i, ldm_ae = ldm_ae, ldm_decoder = ldm_decoder, msg_decoder = msg_decoder, vqgan_to_imnet=vqgan_to_imnet, key=key, params=params)
            train_stats_arr.append(train_stats)
            # x_tr step | single backprop to obtain gradients
            imgs_z = ldm_ae.encode(x_tr) # encode image first, b c h w -> b z h/f w/f
            imgs_z = imgs_z.mode()
            decoded = attacked_decoder.decode(imgs_z) # b z h/f w/f -> b c h w

            # extract watermark to compute loss with
            attacked_msg = msg_decoder(decoded) # b c h w -> b k
            l_tr = loss_w(attacked_msg, og_key.repeat(attacked_msg.shape[0], 1))
            l_tr*=l_tr_grad_scale/(params.inner_steps)
            l_tr.backward() # backward pass to compute gradients
            #g_tr += attacked_msg.grad + ((l_tr)/params.inner_steps)# gradient accumulation
        

        x_retain = next(iter(Retain_loader))[:params.batch_size].to(device) # TODO: need to improve
        
        imgs_z = ldm_ae.encode(x_retain) # encode image first, b c h w -> b z h/f w/f
        imgs_z = imgs_z.mode()
        attacked_img = ldm_decoder.decode(imgs_z) # b z h/f w/f -> b c h w
        og_img = og_decoder.decode(imgs_z)
        attacked_msg = msg_decoder(attacked_img) # original, b c h w -> b k
        loss_watermark = loss_w(attacked_msg, og_key.repeat(attacked_msg.shape[0], 1))
        loss_image = loss_i(attacked_img, og_img)
        loss_combined = loss_watermark + loss_image
        #loss_combined.backward()
        
        # image loss
        l_retain = loss_combined + get_weights_mean(hidden_states_ldm, hidden_states_og)
        l_retain*=l_retain_grad_scale
        l_retain.backward()
        optimizer.step()
        
    return ldm_decoder, train_stats_arr


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
