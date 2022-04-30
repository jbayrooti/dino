# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import getpass
import wandb

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from utils import l2_normalize
from img_2_img import Img2Img

import utils
import vision_transformer as vits
from vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=256, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--encoder_optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of encoder optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--global_crops_size', type=float, default=224,
        help="""Size of the global crops.""")
    parser.add_argument('--local_crops_size', type=float, default=96,
        help="""Size of the local crops.""")

    # Viewmaker parameters
    parser.add_argument('--filter_size', default=32, type=int, 
        help='Please specify the filter size.')
    parser.add_argument('--noise_dim', default=100, type=int,
        help='Please specify the dimension of noise.')
    parser.add_argument('--view_bound_magnitude', default=0.1, type=float,
        help='Please specify the budget size.')
    parser.add_argument('--budget_type', default="all", type=str,
        help='Please specify the budget type ("all", "partial", or "none").')
    parser.add_argument('--viewmaker_network', default="basic", type=str,
        help='Please specify the type of viewmaker network.')
    parser.add_argument('--viewmaker_optimizer', default='adam', type=str, help="""Type of view optimizer.""")
    parser.add_argument('--t', default=0.07, type=float,
        help='Please specify the temperature.')

    # Misc
    parser.add_argument('--dataset', default="imagenet", type=str,
        help='Please specify the training dataset.')
    parser.add_argument('--data_path', default='/scr/jasmine7/imagenet_raw/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--exp_name', default="test", type=str, help='Name for the experiment.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=1, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=20, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    if args.rank == 0:
        wandb.init(project='dino', entity='vm', name=args.exp_name, sync_tensorboard=True) #, settings=wandb.Settings(start_method="fork"))
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    if args.dataset == "imagenet":
        transforms = DataAugmentationImageNetDINO(
            args.global_crops_scale,
            args.local_crops_scale,
            args.local_crops_number,
            args.global_crops_size,
            args.local_crops_size
        )
        dataset = datasets.ImageFolder(args.data_path, transform=transforms)
        print(f"Set up imagenet augmentations")
    else:
        transforms = DataAugmentationCIFAR10DINO(
            args.global_crops_scale,
            args.local_crops_scale,
            args.local_crops_number,
        )
        dataset = datasets.CIFAR10(root = f"/scr/jasmine7/cifar10", transform=transforms, download=True)
        print(f"Set up cifar10 augmentations")
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknown architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ building viewmaker network ... ============
    viewmaker = Img2Img(
        args.filter_size,
        args.noise_dim,
        num_channels=3,
        bound_magnitude=args.view_bound_magnitude,
        budget_type = args.budget_type,
        neutralad=True,
        network=args.viewmaker_network,
        activation='relu',
        clamp=True,
    )
    # move viewmaker to gpu
    viewmaker = viewmaker.cuda()

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizers ... ============
    torch.autograd.set_detect_anomaly(True)
    encoder_params_groups = utils.get_params_groups(student)
    if args.encoder_optimizer != "adamw":
        print("Encoder optimizer type must be adamw")
    encoder_optimizer = torch.optim.AdamW(encoder_params_groups)  # to use with ViTs
    viewmaker_optimizer = torch.optim.Adam(viewmaker.parameters(), lr=0.001)
    
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # for nccl backend
        # args.lr * (args.batch_size_per_gpu) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        viewmaker=viewmaker,
        encoder_optimizer=encoder_optimizer,
        viewmaker_optimizer=viewmaker_optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, viewmaker, teacher_without_ddp, dino_loss,
            data_loader, encoder_optimizer, viewmaker_optimizer, lr_schedule, wd_schedule, 
            momentum_schedule, epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'viewmaker': viewmaker.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'viewmaker_optimizer': viewmaker_optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, viewmaker, teacher_without_ddp, dino_loss, data_loader,
                    encoder_optimizer, viewmaker_optimizer, lr_schedule, wd_schedule, 
                    momentum_schedule, epoch, fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(encoder_optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images] # 2 x batch_size x 3 x 96 x 96
        
        # compute encoder loss
        norm_views1 = normalize(viewmaker(images[0]))
        norm_views2 = normalize(viewmaker(images[1]))
        norm_views = [norm_views1, norm_views2]
        for i in range(args.local_crops_number):
            norm_views.append(normalize(viewmaker(images[i + 2])))
        teacher_embs = teacher(norm_views[:2])  # only the 2 global views pass through the teacher
        student_embs = student(norm_views) # pass in images here instead for a check
        encoder_loss = dino_loss(student_embs, teacher_embs, epoch)

        # student update
        param_norms = None
        if fp16_scaler is None:
            encoder_loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            encoder_optimizer.step()
        else:
            fp16_scaler.scale(encoder_loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(encoder_optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(encoder_optimizer)
            fp16_scaler.update()
        encoder_optimizer.zero_grad()
        viewmaker_optimizer.zero_grad()

        # compute viewmaker loss
        anchor_images = images[0]
        views1 = viewmaker(anchor_images)
        views2 = viewmaker(anchor_images)
        if args.rank == 0 and 2 * it % 1000 == 0:
            images_to_log = anchor_images.permute(0,2,3,1).detach()[0].cpu().numpy()
            view1_to_log = views1.permute(0,2,3,1).detach()[0].cpu().numpy()
            view2_to_log = views2.permute(0,2,3,1).detach()[0].cpu().numpy()
            views_to_log = np.concatenate((images_to_log, view1_to_log, view1_to_log - images_to_log, view2_to_log, view2_to_log - images_to_log), axis=1)
            wandb.log({"view examples": wandb.Image(views_to_log, caption=f"Epoch: {epoch}, Step {it}")})

            small_views = []
            for i in range(args.local_crops_number):
                small_view_to_log = norm_views[i + 2].permute(0,2,3,1).detach()[0].cpu().numpy()
                small_views.append(small_view_to_log)
            small_views_conc = np.concatenate(tuple(small_views), axis=1)
            wandb.log({"small views": wandb.Image(small_views_conc, caption=f"Epoch: {epoch}, Step {it}")})

        embs = student(normalize(anchor_images)) # I think this should not be normalized
        embs1 = student(normalize(views1))
        embs2 = student(normalize(views2))
        loss_function = NeuTraLADLoss(embs, embs1, embs2, t=args.t)
        viewmaker_loss = loss_function.get_loss()

        # compute loss for the vm
        viewmaker_loss.backward()
        viewmaker_optimizer.step()
        viewmaker_optimizer.zero_grad()
        encoder_optimizer.zero_grad()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(encoder_loss=encoder_loss.item())
        metric_logger.update(viewmaker_loss=viewmaker_loss.item())
        metric_logger.update(lr=encoder_optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=encoder_optimizer.param_groups[0]["weight_decay"])
        if args.rank == 0:
            wandb.log({"encoder_loss": encoder_loss.item(), "viewmaker_loss": viewmaker_loss.item(),"lr": encoder_optimizer.param_groups[0]["lr"], "wd": encoder_optimizer.param_groups[0]["weight_decay"], "epoch": epoch, "view_bound_magnitude": args.view_bound_magnitude})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def normalize(imgs):
        # CIFAR 10 normalization
        # mean = torch.tensor([0.491, 0.482, 0.446], device=imgs.device)
        # std = torch.tensor([0.247, 0.243, 0.261], device=imgs.device)
        # IMAGENET/standard normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device)
        imgs = (imgs - mean[None, :, None, None]) / std[None, :, None, None]
        return imgs


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                # q is torch.Size([128, 65536]) for iq = 0
                # student_out[1] is torch.Size([26, 65536])
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size()) # for nccl
        # batch_center = batch_center / len(teacher_output) # for gloo

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class NeuTraLADLoss(object):
    
    def __init__(self, outputs_orig, outputs1, outputs2, t=0.07):
        super().__init__()
        self.outputs1 = l2_normalize(outputs1, dim=1)
        self.outputs2 = l2_normalize(outputs2, dim=1)
        self.outputs_orig = l2_normalize(outputs_orig, dim=1)
        self.t = t

    def get_loss(self):
        # https://arxiv.org/pdf/2103.16440.pdf

        batch_size = self.outputs_orig.size(0)  # batch_size x out_dim
        
        sim_x_x1 = torch.sum(self.outputs1 * self.outputs_orig, dim=-1) / self.t # [256]
        sim_x_x2 = torch.sum(self.outputs2 * self.outputs_orig, dim=-1) / self.t # [256]
        sim_x1_x2 = torch.sum(self.outputs1 * self.outputs2, dim=-1) / self.t # [256]
        
        sim_x1_12_cat = torch.cat([sim_x_x1.unsqueeze(-1), sim_x1_x2.unsqueeze(-1)], dim=-1) # [256, 2]
        sim_x1_12_norm = torch.logsumexp(sim_x1_12_cat, dim=1) # [256]

        sim_x2_12_cat = torch.cat([sim_x_x2.unsqueeze(-1), sim_x1_x2.unsqueeze(-1)], dim=-1) # [256, 2]
        sim_x2_12_norm = torch.logsumexp(sim_x2_12_cat, dim=1) # [256]
        
        loss = -torch.mean((sim_x_x1 - sim_x1_12_norm) + (sim_x_x2 - sim_x2_12_norm))
        return loss

class DataAugmentationImageNetDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, global_crops_size=224, local_crops_size=96):
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
    
class DataAugmentationCIFAR10DINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=local_crops_scale, interpolation=Image.BICUBIC),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    print("Using NeuTraL AD Loss + Viewmaker")
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
