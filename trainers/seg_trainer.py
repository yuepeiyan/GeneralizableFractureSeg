import math
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import torchvision
from matplotlib.pyplot import grid
from numpy import nanmean, nonzero, percentile
from torchprofile import profile_macs

sys.path.append('..')

from .. import models
from .. import networks
# import models
# import networks
from ..utils import SmoothedValue, LayerDecayValueAssigner

from data.med_transforms import get_scratch_train_transforms, get_val_transforms, get_post_transforms, \
    get_vis_transforms, get_raw_transforms
from data.med_datasets import get_msd_trainset, get_train_loader, get_val_loader, idx2label_all, btcv_8cls_idx, \
    tibial_plateau_idx
from tools.visualization import patches3d_to_grid, images3d_to_grid
from .base_trainer import BaseTrainer

from timm.data import Mixup
from timm.utils import accuracy
from timm.models.layers.helpers import to_3tuple

from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import compute_meandice, compute_hausdorff_distance

from collections import defaultdict, OrderedDict
from torch.utils.tensorboard import SummaryWriter


def compute_avg_metric(metric, meters, metric_name, batch_size, args):
    assert len(metric.shape) == 2
    if args.dataset in ['tibial_plateau', 'pelvic']:
        cls_avg_metric = np.mean(np.ma.masked_invalid(np.nanmean(metric, axis=0)))
        cls4_avg_metric = np.mean(np.ma.masked_invalid(np.nanmean(metric[..., tibial_plateau_idx], axis=0)))
        meters[metric_name].update(value=cls_avg_metric, n=batch_size)
        meters[f'cls4_{metric_name}'].update(value=cls4_avg_metric, n=batch_size)
    else:
        cls_avg_metric = np.nanmean(np.nanmean(metric, axis=0))
        meters[metric_name].update(value=cls_avg_metric, n=batch_size)


class SegTrainer(BaseTrainer):
    r"""
    General Segmentation Trainer
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.proj_name
        self.writer = SummaryWriter(args.output_dir)
        self.scaler = torch.cuda.amp.GradScaler()
        if args.test:
            self.metric_funcs = OrderedDict([
                ('Dice',
                 compute_meandice),
                ('HD',
                 partial(compute_hausdorff_distance, percentile=95))
            ])
        else:
            self.metric_funcs = OrderedDict([
                ('Dice',
                 compute_meandice)
            ])

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name}")
            if args.dataset in ['tibial_plateau', 'pelvic']:
                args.num_classes = 5
                self.loss_fn = DiceLoss(to_onehot_y=True,
                                        softmax=True,
                                        squared_pred=True,
                                        smooth_nr=args.smooth_nr,
                                        smooth_dr=args.smooth_dr)
            else:
                raise ValueError(f"Unsupported dataset {args.dataset}")
            self.post_pred, self.post_label = get_post_transforms(args)

            # setup mixup and loss functions
            if args.mixup > 0:
                raise NotImplemented("Mixup for segmentation has not been implemented.")
            else:
                self.mixup_fn = None

            self.model = getattr(models, self.model_name)(encoder=getattr(networks, args.enc_arch),
                                                          decoder=getattr(networks, args.dec_arch),
                                                          args=args)

            # load pretrained weights
            if hasattr(args, 'test') and args.test and args.pretrain is not None and os.path.exists(args.pretrain):
                print(f"=> Start loading the model weights from {args.pretrain} for test")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                state_dict = checkpoint['state_dict']
                msg = self.model.load_state_dict(state_dict, strict=False)
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")
            elif args.pretrain is not None and os.path.exists(args.pretrain):
                print(f"=> Start loading pretrained weights from {args.pretrain}")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                # import pdb
                # pdb.set_trace()
                if self.model_name == 'UNETR3D':
                    for key in list(state_dict.keys()):
                        if key.startswith('encoder.'):
                            state_dict[key[len('encoder.'):]] = state_dict[key]
                            del state_dict[key]
                        # need to concat and load pos embed. too
                        # TODO: unify the learning of pos embed of pretraining and finetuning
                        if key == 'encoder_pos_embed':
                            pe = torch.zeros([1, 1, state_dict[key].size(-1)])
                            state_dict['pos_embed'] = torch.cat([pe, state_dict[key]], dim=1)
                            del state_dict[key]
                        if key == 'patch_embed.proj.weight' and \
                                state_dict[
                                    'patch_embed.proj.weight'].shape != self.model.encoder.patch_embed.proj.weight.shape:
                            del state_dict['patch_embed.proj.weight']
                            del state_dict['patch_embed.proj.bias']
                        if key == 'pos_embed' and \
                                state_dict['pos_embed'].shape != self.model.encoder.pos_embed.shape:
                            del state_dict[key]
                    msg = self.model.encoder.load_state_dict(state_dict, strict=False)
                # self.model.load(state_dict)
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")

            self.wrap_model()
        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        assert (self.model is not None and self.wrapped_model is not None), \
            "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args
        model = self.model

        num_layers = model.get_num_layers()
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))

        # optim_params = self.group_params(model)
        optim_params = self.get_parameter_groups(get_layer_id=partial(assigner.get_layer_id, prefix='encoder.'),
                                                 get_layer_scale=assigner.get_scale,
                                                 verbose=True)
        # TODO: create optimizer factory
        self.optimizer = torch.optim.AdamW(optim_params,
                                           lr=args.lr,
                                           betas=(args.beta1, args.beta2),
                                           weight_decay=args.weight_decay)

    def build_dataloader(self):
        if self.dataloader is None:
            print("=> creating train dataloader")
            args = self.args

            if args.dataset in ['tibial_plateau', 'pelvic']:
                # build train dataloader
                if not args.test:
                    train_transform = get_scratch_train_transforms(args)
                    self.dataloader = get_train_loader(args,
                                                       batch_size=self.batch_size,
                                                       workers=self.workers,
                                                       train_transform=train_transform)
                    self.iters_per_epoch = len(self.dataloader)
                    print(f"==> Length of train dataloader is {self.iters_per_epoch}")
                else:
                    self.dataloader = None
                # build val dataloader
                val_transform = get_val_transforms(args)
                self.val_dataloader = get_val_loader(args,
                                                     batch_size=args.val_batch_size,  # batch per gpu
                                                     workers=self.workers,
                                                     val_transform=val_transform)
                # build vis dataloader
                vis_transform = get_vis_transforms(args)
                self.vis_dataloader = get_val_loader(args,
                                                     batch_size=args.vis_batch_size,
                                                     workers=self.workers,
                                                     val_transform=vis_transform)
            else:
                raise ValueError("Currently only support brats2020 dataset")
        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")
        print("=> finish creating dataloader")

    def run(self):
        args = self.args
        # Compute iterations when resuming
        niters = args.start_epoch * self.iters_per_epoch

        best_metric = 0
        best_ts_metric = 0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                self.dataloader.sampler.set_epoch(epoch)
                torch.distributed.barrier()

            # if epoch == args.start_epoch:
            #     self.evaluate(epoch=epoch, niters=niters)

            # train for one epoch
            niters = self.epoch_train(epoch, niters)

            # evaluate after each epoch training
            if (epoch + 1) % args.eval_freq == 0:
                metric_list = self.evaluate(epoch=epoch, niters=niters)
                metric = metric_list[0]
                self.writer.add_scalar('Training/Dice', metric, epoch)
                if len(metric_list) == 2:
                    ts_metric = metric_list[1]
                else:
                    ts_metric = None
                if metric > best_metric:
                    print(f"=> New val best metric: {metric} | Old val best metric: {best_metric}!")
                    best_metric = metric
                    self.writer.add_scalar('Training/Best_Dice', best_metric, epoch)
                    if ts_metric is not None:
                        print(f"=> New ts best metric: {ts_metric} | Old ts best metric: {best_ts_metric}!")
                        best_ts_metric = ts_metric
                    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                        self.save_checkpoint(
                            {
                                'epoch': epoch + 1,
                                'arch': args.arch,
                                'state_dict': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'scaler': self.scaler.state_dict(),  # additional line compared with base imple
                                'metric': metric
                            },
                            is_best=False,
                            filename=f'{args.ckpt_dir}/best_model.pth.tar'
                        )
                        print("=> Finish saving best model.")
                else:
                    print(f"=> Still old val best metric: {best_metric}")
                    if ts_metric is not None:
                        print(f"=> Still old ts best metric: {best_ts_metric}")

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if (epoch + 1) % args.save_freq == 0:
                    # TODO: save the best
                    self.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(),  # additional line compared with base imple
                        },
                        is_best=False,
                        filename=f'{args.ckpt_dir}/checkpoint_{epoch:04d}.pth.tar'
                    )

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler
        mixup_fn = self.mixup_fn
        loss_fn = self.loss_fn

        # switch to train mode
        model.train()

        load_start_time = time.time()
        for i, batch_data in enumerate(train_loader):
            load_time = time.time() - load_start_time
            # adjust learning at the beginning of each iteration
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            image = batch_data['image']
            target = batch_data['label']

            # print(image.shape)

            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            if mixup_fn is not None:
                image, target = mixup_fn(image, target)

            # compute output and loss
            forward_start_time = time.time()
            # forward_start_time_1 = time.perf_counter()
            with torch.cuda.amp.autocast(True):
                loss = self.train_class_batch(model, image, target, loss_fn)
            forward_time = time.time() - forward_start_time

            # compute gradient and do SGD step
            bp_start_time = time.time()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bp_time = time.time() - bp_start_time

            # torch.cuda.synchronize()
            # print(f"training iter time is {time.perf_counter() - forward_start_time_1}")

            # Log to the screen
            if i % args.print_freq == 0:
                if 'lr_scale' in optimizer.param_groups[0]:
                    last_layer_lr = optimizer.param_groups[0]['lr'] / optimizer.param_groups[0]['lr_scale']
                else:
                    last_layer_lr = optimizer.param_groups[0]['lr']

                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.iters_per_epoch} | "
                      f"TotalIter: {niters:06d} | "
                      f"Init Lr: {self.lr:.05f} | "
                      f"Lr: {last_layer_lr:.05f} | "
                      f"Load Time: {load_time:.03f}s | "
                      f"Forward Time: {forward_time:.03f}s | "
                      f"Backward Time: {bp_time:.03f}s | "
                      f"Loss: {loss.item():.03f}")

                self.writer.add_scalar('Training/Total_Loss', loss.item(), niters)
                self.writer.add_scalar('Training/Learning_Rate', last_layer_lr, niters)

            niters += 1
            load_start_time = time.time()
        return niters

    @staticmethod
    def train_class_batch(model, samples, target, criterion):
        outputs = model(samples)
        loss = criterion(outputs, target)
        return loss

    @torch.no_grad()
    def evaluate(self, epoch=0, niters=0):
        print("=> Start Evaluating")
        args = self.args
        model = self.wrapped_model
        val_loader = self.val_dataloader

        if args.spatial_dim == 3:
            roi_size = (args.roi_x, args.roi_y, args.roi_z)
        elif args.spatial_dim == 2:
            roi_size = (args.roi_x, args.roi_y)
        else:
            raise ValueError(f"Do not support this spatial dimension (={args.spatial_dim}) for now")

        meters = defaultdict(SmoothedValue)
        if hasattr(args, 'ts_ratio') and args.ts_ratio != 0:
            assert args.batch_size == 1, "Test mode requires batch size 1"
            ts_samples = int(len(val_loader) * args.ts_ratio)
            val_samples = len(val_loader) - ts_samples
            ts_meters = defaultdict(SmoothedValue)
        else:
            ts_samples = 0
            val_samples = len(val_loader)
            ts_meters = None
        print(f"val samples: {val_samples} and test samples: {ts_samples}")

        # switch to evaluation mode
        model.eval()
        for i, batch_data in enumerate(val_loader):
            image, target = batch_data['image'], batch_data['label']
            if args.gpu is not None:
                image = image.to(args.gpu, non_blocking=True)
                target = target.to(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = sliding_window_inference(image,
                                                  roi_size=roi_size,
                                                  sw_batch_size=4,
                                                  predictor=model,
                                                  overlap=args.infer_overlap)
            target_convert = torch.stack([self.post_label(target_tensor) for target_tensor in decollate_batch(target)],
                                         dim=0)
            output_convert = torch.stack([self.post_pred(output_tensor) for output_tensor in decollate_batch(output)],
                                         dim=0)

            batch_size = image.size(0)
            idx2label = idx2label_all[args.dataset]
            for metric_name, metric_func in self.metric_funcs.items():
                if i < val_samples:
                    log_meters = meters
                else:
                    log_meters = ts_meters
                metric = metric_func(y_pred=output_convert, y=target_convert,
                                     include_background=False if args.dataset in ['btcv', 'tibial_plateau',
                                                                                  'pelvic'] else True)
                metric = metric.cpu().numpy()
                compute_avg_metric(metric, log_meters, metric_name, batch_size, args)
                for k in range(metric.shape[-1]):
                    cls_metric = np.nanmean(metric, axis=0)[k]
                    if np.isnan(cls_metric) or np.isinf(cls_metric):
                        continue
                    log_meters[f'{idx2label[k]}.{metric_name}'].update(value=cls_metric, n=batch_size)
            print(f'==> Evaluating on the {i + 1}th batch is finished.')

        # gather the stats from all processes
        if args.distributed:
            for k, v in meters.items():
                print(f'==> start synchronizing meter {k}...')
                v.synchronize_between_processes()
                print(f'==> finish synchronizing meter {k}...')
            if ts_meters is not None:
                for k, v in ts_meters.items():
                    print(f'==> start synchronizing meter {k}...')
                    v.synchronize_between_processes()
                    print(f'==> finish synchronizing meter {k}...')
        # pdb.set_trace()
        log_string = f"==> Epoch {epoch:04d} val results: \n"
        total_dice = []
        for k, v in meters.items():
            global_avg_metric = v.global_avg
            total_dice.append(global_avg_metric)
            new_line = f"===> {k}: {global_avg_metric:.05f} \n"
            log_string += new_line
        avg_dice = np.nanmean(total_dice)
        print(log_string)
        if ts_meters is not None:
            log_string = f"==> Epoch {epoch:04d} test results: \n"
            for k, v in ts_meters.items():
                global_avg_metric = v.global_avg
                new_line = f"===> {k}: {global_avg_metric:.05f} \n"
                log_string += new_line
            print(log_string)

        print("=> Finish Evaluating")

        if args.dataset in ['tibial_plateau', 'pelvic']:
            assert ts_meters is None
            # return [meters['cls4_Dice'].global_avg]
            return [avg_dice]

    def mask2image(self, images, targets, outputs, id, channel_ind=0, directory='seg_vis_policy'):
        args = self.args

        directory = os.path.join(args.output_dir, directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

        image_list = [im for im in decollate_batch(images)]
        target_convert = [self.post_label(target_tensor) for target_tensor in decollate_batch(targets)]
        output_convert = [self.post_pred(output_tensor) for output_tensor in decollate_batch(outputs)]
        from matplotlib import cm
        import matplotlib.pyplot as plt
        for image_t, target_t, output_t in zip(image_list, target_convert, output_convert):
            depth = target_t.size(3)
            for i, ratio in enumerate([0.3, 0.4, 0.5, 0.6, 0.7]):
                image = image_t.permute(1, 2, 3, 0)[:, :, int(depth * ratio), channel_ind]
                target = target_t.permute(1, 2, 3, 0)[:, :, int(depth * ratio), :]
                output = output_t.permute(1, 2, 3, 0)[:, :, int(depth * ratio), :]
                if args.dataset == 'btcv':
                    vmin, vmax = 0, 12
                    target_mask = target.argmax(dim=-1)
                    output_mask = output.argmax(dim=-1)
                    target_alphas = 1 - target[..., 0]
                    output_alphas = 1 - output[..., 0]
                elif args.dataset == 'msd_brats':
                    vmin, vmax = 0, 3
                    target_mask = torch.zeros(image.shape).int()
                    target_mask[target[:, :, 1].bool()] = 1
                    target_mask[target[:, :, 0].bool()] = 2
                    target_mask[target[:, :, 2].bool()] = 3
                    target_alphas = (target_mask > 0).float()

                    output_mask = torch.zeros(image.shape).int()
                    output_mask[output[:, :, 1].bool()] = 1
                    output_mask[output[:, :, 0].bool()] = 2
                    output_mask[output[:, :, 2].bool()] = 3
                    output_alphas = (output_mask > 0).float()

                image = image.cpu().numpy()
                target_mask = target_mask.cpu().numpy()
                output_mask = output_mask.cpu().numpy()
                target_alphas = target_alphas.cpu().numpy()
                output_alphas = output_alphas.cpu().numpy()
                # pdb.set_trace()
                print("start saving")

                # target
                fig = plt.figure(frameon=False)
                # fig.set_size_inches(16, 16)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(image, cmap='gray')
                ax.imshow(target_mask, alpha=target_alphas, vmin=vmin, vmax=vmax, cmap='viridis')
                fig.savefig(os.path.join(directory, f'{args.dataset}_bid{id}_depidx{i}_gt.png'))
                print(f"finish saving gt {i}")

                # output
                fig = plt.figure(frameon=False)
                # fig.set_size_inches(16, 16)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(image, cmap='gray')
                ax.imshow(output_mask, alpha=output_alphas, vmin=vmin, vmax=vmax, cmap='viridis')
                fig.savefig(os.path.join(directory, f'{args.dataset}_bid{id}_depidx{i}.png'))
                print(f"finish saving output {id}")

    def speedometer(self, niters=0):
        args = self.args
        loader = self.vis_dataloader
        model = self.wrapped_model

        model.eval()

        time_meters = defaultdict(list)
        num_trials = 5
        for t in range(num_trials):
            len_ds = len(loader)
            for b_idx, batch_data in enumerate(loader):
                image = batch_data['image']
                # target = batch_data['label']
                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)  # [B, 4, IH, IW, ID]
                start_time = time.time()
                if b_idx > 1 and b_idx < len_ds - 1:
                    model(image, time_meters=time_meters)
                    end_time = time.time()
                    time_meters['total'].append(end_time - start_time)
                else:
                    model(image)
            print(f"finish trial {t}")
        for key in time_meters.keys():
            # print(f'num of records in {key} is {len(time_meters[key])}')
            avg_time = np.mean(time_meters[key])
            print(f"=> averaged inference time for {key} is {avg_time}")
        # print(f"==> avg inference time over all trials is {np.mean(trials)}")

    def speedometerv2(self):
        args = self.args
        model = self.wrapped_model

        model.eval()

        time_meters = defaultdict(list)
        num_trials = 16
        for t in range(num_trials):
            image = torch.rand(args.batch_size, args.in_chans, args.roi_x, args.roi_y, args.roi_z)
            single_image = torch.rand(1, args.in_chans, args.roi_x, args.roi_y, args.roi_z)
            if args.gpu is not None:
                single_image = single_image.cuda(args.gpu, non_blocking=True)  # [B, 4, IH, IW, ID]
                image = image.cuda(args.gpu, non_blocking=True)  # [B, 4, IH, IW, ID]
            print(f"image shape is {image.shape}")
            if t == 0:
                try:
                    macs = profile_macs(model, single_image) * 1e-9
                except:
                    macs = -1
                print(f"MACS is {macs} G")
            # target = batch_data['label']
            if t > 2 and t < 13:
                start_time = time.perf_counter()
                model(image, time_meters=time_meters)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                time_meters['total'].append(end_time - start_time)
            else:
                model(image)
            print(f"finish trial {t}")
        for key in time_meters.keys():
            # print(f'num of records in {key} is {len(time_meters[key])}')
            avg_time = np.mean(time_meters[key])
            print(f"=> averaged inference time for {key} is {avg_time}")
        print(f"MACS is {macs} G")
        print(f"{4 / np.mean(time_meters['total'])} total")
        print(f"{4 / np.mean(time_meters['enc'])} enc")
        # print(f"==> avg inference time over all trials is {np.mean(trials)}")

    def calc_sparsity(self):
        args = self.args
        raw_transform = get_raw_transforms(args)
        raw_dataloader = get_val_loader(args,
                                        batch_size=1,
                                        workers=1,
                                        val_transform=raw_transform)
        if args.dataset == 'msd_brats':
            cls_sparsity_sum = [0] * 3
        elif args.dataset == 'btcv':
            cls_sparsity_sum = [0] * 14
        for b_idx, batch_data in enumerate(raw_dataloader):
            print("=========================================")
            print(f"the {b_idx} sample")
            # image = batch_data['image']
            target = batch_data['label']
            # pdb.set_trace()
            # print(f"{b_idx}th sample")
            volume_shape = target.shape
            num_voxels = volume_shape[-3] * volume_shape[-2] * volume_shape[-1]
            if args.dataset == 'msd_brats':
                for c in range(3):
                    cls_target = target[0, c]
                    num_target = cls_target.sum()
                    sparsity = (num_voxels - num_target) / num_voxels
                    cls_sparsity_sum[c] += sparsity
                    print("-----------------------")
                    print(f"current class {c} sparsity is ({num_voxels} - {num_target})/{num_voxels} = {sparsity}")
                    print(f"current class {c} accum sparsity is {cls_sparsity_sum[c] / (b_idx + 1)}")
            elif args.dataset == 'btcv':
                for c in range(14):
                    if c == 0:
                        num_target = num_voxels - (target[0, 0] == c).sum()
                    else:
                        num_target = (target[0, 0] == c).sum()
                    sparsity = (num_voxels - num_target) / num_voxels
                    cls_sparsity_sum[c] += sparsity
                    print("-----------------------")
                    print(f"current class {c} sparsity is ({num_voxels} - {num_target})/{num_voxels} = {sparsity}")
                    print(f"current class {c} accum sparsity is {cls_sparsity_sum[c] / (b_idx + 1)}")

    def resume(self):
        args = self.args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])  # additional line compared with base imple
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    def adjust_learning_rate(self, epoch, args):
        """Base schedule: CosineDecay with warm-up."""
        init_lr = self.lr
        if epoch < args.warmup_epochs:
            cur_lr = init_lr * epoch / args.warmup_epochs
        else:
            cur_lr = init_lr * 0.5 * (
                        1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if 'lr_scale' in param_group:
                param_group['lr'] = cur_lr * param_group['lr_scale']
            else:
                param_group['lr'] = cur_lr
