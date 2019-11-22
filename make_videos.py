
import numpy as np
import os, sys, math, random, glob, time, itertools
from fire import Fire

#os.environ['CUDA_VISIBLE_DEVICES']='6,7'
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,4,5,6,7'
os.environ['CUDA_VISIBLE_DEVICES']='4,5,6,7'
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils import *
from models import TrainableModel, DataParallelModel, AbstractModel
from task_configs import RealityTask, get_task, get_model, tasks
from transfers import Transfer
from logger import Logger, VisdomLogger
from graph import TaskGraph
from datasets import TaskDataset, ImageDataset, SintelDataset, load_test, load_ood, load_generic
import transforms

torch.manual_seed(229) # cpu  vars
torch.cuda.manual_seed_all(229) # gpu vars

from modules.unet import UNet, UNetOld, UNetOld2, UNetReshade
#UNet = UNetReshade
from transfers import functional_transfers as ft

import IPython


### Use case: when finetuning a rgb2normal network
### Inputs rgb2normal/depth net, image-task dataset 

### Depth/normal

### Almena Depth/normal Check
### Almena corrupted intensity 1 Depth/normal Check
### Almena corrupted intensity 2 Depth/normal Check
### Almena corrupted intensity 3 Depth/normal Check
### Almena corrupted intensity 4 Depth/normal Check

### Almena PGD epsilon 1e-3 Depth/normal
### Almena PGD epsilon 1e-1 Depth/normal

### Sintel Depth/normal
### NYU Depth/normal


class ValidationMetrics(object):

    PLOT_METRICS = ["ang_error_median", "eval_mse"]

    def __init__(self, name, src_task=get_task("rgb"), dest_task=get_task("normal")):
        self.name = name
        self.src_task, self.dest_task = src_task, dest_task
        self.load_dataset()

    def load_dataset(self):
        # self.dataset = TaskDataset(["almena", "albertville"], tasks=[self.src_task, self.dest_task])

        self.dataset = TaskDataset(["almena", "albertville"], tasks=[self.src_task, self.dest_task])


    def build_dataloader(self, sample=None, batch_size=16, seed=229):
        sampler = torch.utils.data.SequentialSampler() if sample is None else \
            torch.utils.data.SubsetRandomSampler(random.Random(seed).sample(range(len(self.dataset)), sample))

        eval_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size,
            num_workers=16, sampler=sampler, pin_memory=True
        )
        return eval_loader

    def get_metrics(self, pred, target):
        """ Gets standard set of metrics for predictions and targets """

        batch_losses, _ = self.dest_task.norm(pred, target, batch_mean=False)

        masks = self.dest_task.build_mask(target, val=self.dest_task.mask_val)
        (pred * masks.float() - target * masks.float()) ** 2
        original_pred, original_target, masks = (x.data.permute((0, 2, 3, 1)).cpu().numpy() for x in [pred, target, masks])
        masks = masks[:, :, :, 0]
        
        norm = lambda a: np.sqrt((a * a).sum(axis=1))
        def cosine_similarity(x1, x2, dim=1, eps=1e-8):
            w12 = np.sum(x1 * x2, dim)
            w1, w2 = norm(x1), norm(x2)
            return (w12 / (w1 * w2).clip(min=eps)).clip(min=-1.0, max=1.0)

        original_pred = original_pred.astype('float64')
        original_target = original_target.astype('float64')
        num_examples, width, height, num_channels = original_pred.shape
        _, _, _, num_channels_targ = original_pred.shape

        pred = original_pred.reshape([-1, num_channels])
        target = original_target.reshape([-1, num_channels_targ])
        num_valid_pixels, num_invalid_pixels = np.sum(masks), np.sum(1 - masks)

        # ang_errors_per_pixel_unraveled = np.arccos(cosine_similarity(pred, target)) * 180 / math.pi
        # ang_errors_per_pixel = ang_errors_per_pixel_unraveled.reshape(num_examples, width, height)
        # ang_errors_per_pixel_masked = ang_errors_per_pixel * masks
        # ang_error_mean = np.sum(ang_errors_per_pixel_masked) / num_valid_pixels
        # ang_error_without_masking = np.mean(ang_errors_per_pixel)
        # ang_error_median = np.mean(np.median(np.ma.masked_equal(ang_errors_per_pixel_masked, 0), axis=1))

        normed_pred = pred / (norm(pred)[:, None] + 2e-1)
        normed_target = target / (norm(target)[:, None] + 2e-1)
        masks_expanded = np.expand_dims(masks, num_channels_targ).reshape([-1])

        mse = (normed_pred - normed_target) * masks_expanded[:, None]
        l1 = np.mean(np.absolute(mse))
        losses = np.mean(mse.reshape(num_examples, -1) ** 2, axis=0)
        mse, rmse = np.mean(mse ** 2), np.sqrt(np.mean(mse ** 2)) * 255.0
        
        # threshold_1125 = (np.sum(ang_errors_per_pixel_masked <= 11.25) - num_invalid_pixels) / num_valid_pixels
        # threshold_225 = (np.sum(ang_errors_per_pixel_masked <= 22.5) - num_invalid_pixels) / num_valid_pixels
        # threshold_30 = (np.sum(ang_errors_per_pixel_masked <= 30) - num_invalid_pixels) / num_valid_pixels
        
        return {
            # "ang_error_without_masking": ang_error_without_masking,
            # "ang_error_mean": f"{ang_error_mean:0.2f}",
            # "ang_error_median": f"{ang_error_median:0.2f}",
            "eval_mse": f"{np.mean(losses) * 100:0.5f}",
            "eval_std": f"{np.std(losses)*100:0.3f}",
            # "eval_rmse": rmse,
            "eval_L1": f"{l1 * 100:0.2f}",
            # 'percentage_within_11.25_degrees': threshold_1125,
            # 'percentage_within_22.5_degrees': threshold_225,
            # 'percentage_within_30_degrees': threshold_30,
        }

    @staticmethod
    def plot(logger):
        for metric in ValidationMetrics.PLOT_METRICS:
            keys = [key for key in logger.data if metric in key]
            data = np.stack((logger.data[key] for key in keys), axis=1)
            logger.plot(data, metric, opts={"legend": keys})

    def evaluate(self, model, logger=None, sample=None, show_images=False, image_limit=64):
        """ Evaluates dataset on model. """

        eval_loader = self.build_dataloader(sample=sample)
        images, preds, targets, _, _ = model.predict_with_data(eval_loader)
        metrics = self.get_metrics(preds, targets)

        if logger is not None:
            # for metric in ValidationMetrics.PLOT_METRICS:
            #     logger.update(f"{self.name}_{metric}", metrics[metric])
            if show_images:
                logger.images_grouped([images[0:image_limit], preds[0:image_limit], targets[0:image_limit]], self.name, resize=256)

        return metrics


    def evaluate_with_percep(self, model, logger=None, sample=None, show_images=False, image_limit=64, percep_model=None):
        """ Evaluates dataset on model. """

        eval_loader = self.build_dataloader(sample=sample)
        with torch.no_grad():
            images, preds, targets, _, _ = model.predict_with_data(eval_loader)

            #old_dataset = self.dataset
            #self.dataset = 
            # Perceptual eval
            percep_model.eval()
            eval_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(preds), batch_size=16,
                num_workers=16, shuffle=False, pin_memory=True
            )
            final_preds = []
            for preds, in eval_loader:
                print('preds shape', preds.shape)
                final_preds += [percep_model.forward(preds[:, -3:])]
            final_preds = torch.cat(final_preds, dim=0)

        metrics = self.get_metrics(final_preds.cpu(), targets.cpu())

        if logger is not None:
            # for metric in ValidationMetrics.PLOT_METRICS:
            #     logger.update(f"{self.name}_{metric}", metrics[metric])
            if show_images:
                logger.images_grouped([images[0:image_limit], preds[0:image_limit], targets[0:image_limit]], self.name, resize=256)

        return metrics




class ImageCorruptionMetrics(ValidationMetrics):

    TRANSFORMS = [
        # transforms.resize, 
        transforms.resize_rect, 
        transforms.color_jitter, 
        transforms.scale,
        transforms.rotate,
        transforms.elastic,
        transforms.translate,
        transforms.gauss,
        transforms.motion_blur,
        transforms.noise,
        transforms.flip,
        transforms.impulse_noise,
        transforms.crop,
        transforms.jpeg_transform,
        transforms.brightness,
        transforms.contrast,
        transforms.blur,
        transforms.pixilate,
    ]

    def __init__(self, *args, **kwargs):
        self.corruption = kwargs.pop('corruption', 1)
        self.transforms = kwargs.pop('transforms', ImageCorruptionMetrics.TRANSFORMS)
        super().__init__(*args, **kwargs)

    def build_dataloader(self, sample=None, seed=229):
        eval_loader = super().build_dataloader(sample=sample, batch_size=8, seed=seed)

        for i, (X, Y) in enumerate(eval_loader):
            transform = self.transforms[i % len(self.transforms)]
            self.orig_image = X
            print ("Transform: ", transform)
            if transform.geometric:
                images = torch.stack([X, Y], dim=1)
                B, N, C, H, W = images.shape
                result = transform.sample_with_intensity(images.view(B*N, C, H, W).to(DEVICE), self.corruption/4.0)
                result = result.view(B, N, C, result.shape[2], result.shape[3])
                yield result[:, 0], result[:, 1]
            else:
                yield transform(X.to(DEVICE)), Y.to(DEVICE)


class AdversarialMetrics(ValidationMetrics):

    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop('model', None)
        self.eps = kwargs.pop('eps', 1e-3)
        self.n = kwargs.pop('n', 10)
        self.lr = kwargs.pop('lr', 0.9)
        super().__init__(*args, **kwargs)

    def build_dataloader(self, sample=None, seed=229):
        eval_loader = super().build_dataloader(sample=sample, batch_size=32, seed=seed)

        for i, (X, Y) in enumerate(eval_loader):

            perturbation = torch.randn_like(X).requires_grad_(True)
            optimizer = torch.optim.Adam([perturbation], lr=self.lr)

            for i in range(0, self.n):
                perturbation_zc = (perturbation - perturbation.mean())/perturbation.std()
                Xn = (X + perturbation_zc).clamp(min=0, max=1)
                Yn_pred = self.model(Xn)
                loss, _ = self.dest_task.norm(Yn_pred, Y.to(Yn_pred.device))
                loss = -1.0*loss
                # print ("Loss: ", loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                perturbation_zc = (perturbation - perturbation.mean())/perturbation.std()
                Xn = (X + self.eps*perturbation_zc.cpu()).clamp(min=0, max=1)

            yield Xn.detach(), Y.detach()

    def evaluate(self, model, logger=None, sample=None, show_images=False):
        self.model = model #set the adversarial model to the current model
        return super().evaluate(model, logger=logger, sample=sample, show_images=show_images)


class SintelMetrics(ValidationMetrics):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_dataset(self):
        self.dataset = SintelDataset(tasks=[self.src_task, self.dest_task])
        print (len(self.dataset))

# datasets = [
#     ValidationMetrics("almena"),
#     # ImageCorruptionMetrics("almena_corrupted1", corruption=1),
#     # ImageCorruptionMetrics("almena_corrupted2", corruption=2),
#     # ImageCorruptionMetrics("almena_corrupted3", corruption=3),
#     # ImageCorruptionMetrics("almena_corrupted4", corruption=4),
#     # AdversarialMetrics("almena_adversarial_eps0.005", eps=5e-3),
#     # AdversarialMetrics("almena_adversarial_eps0.01", eps=1e-2, n=20),
#     # SintelMetrics("sintel"),
# ]

class DummyModel(AbstractModel):
    
    def __init__(self, fn):
        super().__init__()
        self.forward = fn

    def compile(self, *args, **kwargs):
        pass
    
def run_viz_suite(name, data_loader, dest_task=tasks.depth_zbuffer, graph_file=None, model_file=None, old=False, multitask=False, percep_mode=None, downsample=6, out_channels=3, final_task=tasks.normal, oldpercep=False):
    
    extra_task = [final_task] if percep_mode else []

    if graph_file is not None:
        graph = TaskGraph(tasks=[tasks.rgb, dest_task] + extra_task, pretrained=False)
        graph.load_weights(graph_file)
        model = graph.edge(tasks.rgb, dest_task).load_model()
    elif old:
        model = DataParallelModel.load(UNetOld().cuda(), model_file)
    elif multitask:
        model = DataParallelModel.load(UNet(downsample=5, out_channels=6).cuda(), model_file)
    elif model_file is not None:
        # downsample = 5 or 6
        print('loading main model')
        #model = DataParallelModel.load(UNetReshade(downsample=downsample,  out_channels=out_channels).cuda(), model_file)
        model = DataParallelModel.load(UNet(downsample=downsample,  out_channels=out_channels).cuda(), model_file)
        #model = DataParallelModel.load(UNet(downsample=6).cuda(), model_file)
    else:
        model = DummyModel(Transfer(src_task=tasks.rgb, dest_task=dest_task).load_model())

    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)

    # DATA LOADING 1
    results = []
    final_preds = []
    
    if percep_mode:
        print('Loading percep model...')
        if graph_file is not None and not oldpercep:
            percep_model = graph.edge(dest_task, final_task).load_model()
            percep_model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
        else:
            percep_model = Transfer(src_task=dest_task, dest_task=final_task).load_model()
        percep_model.eval()

    print("Converting...")
    for data, in data_loader:
        preds = model.predict_on_batch(data)[:, -3:].clamp(min=0, max=1)
        results.append(preds.detach().cpu())
        if percep_mode:
            try:
                final_preds += [percep_model.forward(preds[:, -3:]).detach().cpu()]
            except RuntimeError:
                preds = torch.cat([preds] * 3, dim=1)
                final_preds += [percep_model.forward(preds[:, -3:]).detach().cpu()]
        #break

    if percep_mode:
        results = torch.cat(final_preds, dim=0)
    else:
        results = torch.cat(results, dim=0)

    return results


from multiprocessing import dummy as mp

def translate(x):
    return torchvision.utils.save_image(*x)


configs = dict(
      
    # RGB -> Depth -> X (Baselines)
    baseline_rgb2depth_zbuffer2principal_curvature=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.depth_zbuffer, final_task=tasks.principal_curvature,
        percep_mode=True, oldpercep=True),
    baseline_rgb2depth_zbuffer2normal=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.depth_zbuffer, final_task=tasks.normal, percep_mode=True, oldpercep=True,
        out_channels=1),
    baseline_rgb2depth_zbuffer2reshading=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.depth_zbuffer, final_task=tasks.reshading, percep_mode=True,oldpercep=True,
        out_channels=3),
    baseline_rgb2depth_zbuffer2sobel_edges=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.depth_zbuffer, final_task=tasks.sobel_edges, percep_mode=True,oldpercep=True,
        out_channels=1),
    baseline_rgb2depth_zbuffer2edge_occlusion=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.depth_zbuffer, final_task=tasks.edge_occlusion, percep_mode=True,oldpercep=True,
        out_channels=1),
    baseline_rgb2depth_zbuffer2keypoints2d=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.depth_zbuffer, final_task=tasks.keypoints2d, percep_mode=True,oldpercep=True,
        out_channels=1),
    baseline_rgb2depth_zbuffer2keypoints3d=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.depth_zbuffer, final_task=tasks.keypoints3d, percep_mode=True,oldpercep=True,
        out_channels=1),

      
    # RGB -> Depth -> X (Consistency)
    consistency_rgb2depth_zbuffer2principal_curvature=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.depth_zbuffer, final_task=tasks.principal_curvature,
        percep_mode=True),
    consistency_rgb2depth_zbuffer2normal=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.depth_zbuffer, final_task=tasks.normal, percep_mode=True,
        out_channels=1),
    consistency_rgb2depth_zbuffer2reshading=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.depth_zbuffer, final_task=tasks.reshading, percep_mode=True,
        out_channels=3),
    consistency_rgb2depth_zbuffer2sobel_edges=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.depth_zbuffer, final_task=tasks.sobel_edges, percep_mode=True,
        out_channels=1),
    consistency_rgb2depth_zbuffer2edge_occlusion=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.depth_zbuffer, final_task=tasks.edge_occlusion, percep_mode=True,
        out_channels=1),
    consistency_rgb2depth_zbuffer2keypoints2d=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.depth_zbuffer, final_task=tasks.keypoints2d, percep_mode=True,
        out_channels=1),
    consistency_rgb2depth_zbuffer2keypoints3d=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.depth_zbuffer, final_task=tasks.keypoints3d, percep_mode=True,
        out_channels=1),

    # RGB -> Normal -> X (Baselines)
    baseline_rgb2normal2principal_curvature=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.normal, final_task=tasks.principal_curvature,oldpercep=True,
        percep_mode=True),
    baseline_rgb2normal2depth_zbuffer=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.normal, final_task=tasks.depth_zbuffer, percep_mode=True,oldpercep=True,
        out_channels=1),
    baseline_rgb2normal2reshading=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.normal, final_task=tasks.reshading, percep_mode=True,oldpercep=True,
        out_channels=3),
    baseline_rgb2normal2sobel_edges=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.normal, final_task=tasks.sobel_edges, percep_mode=True,oldpercep=True,
        out_channels=1),
    baseline_rgb2normal2edge_occlusion=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.normal, final_task=tasks.edge_occlusion, percep_mode=True,oldpercep=True,
        out_channels=1),
    baseline_rgb2normal2keypoints2d=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.normal, final_task=tasks.keypoints2d, percep_mode=True,oldpercep=True,
        out_channels=1),
    baseline_rgb2normal2keypoints3d=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        dest_task=tasks.normal, final_task=tasks.keypoints3d, percep_mode=True,oldpercep=True,
        out_channels=1),
    
    # RGB -> Normal -> X (Consistency)
    consistency_rgb2normal2principal_curvature=dict(
        graph_file=f"{SHARED_DIR}/shared_cloud/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth",
        dest_task=tasks.normal, final_task=tasks.principal_curvature, percep_mode=True),
    consistency_rgb2normal2depth_zbuffer=dict(
        graph_file=f"{SHARED_DIR}/shared_cloud/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth",
        dest_task=tasks.normal, final_task=tasks.depth_zbuffer, percep_mode=True,
        out_channels=1),
    consistency_rgb2normal2reshading=dict(
        graph_file=f"{SHARED_DIR}/shared_cloud/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth",
        dest_task=tasks.normal, final_task=tasks.reshading, percep_mode=True,
        out_channels=3),
    consistency_rgb2normal2sobel_edges=dict(
        graph_file=f"{SHARED_DIR}/shared_cloud/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth",
        dest_task=tasks.normal, final_task=tasks.sobel_edges, percep_mode=True,
        out_channels=1),
    consistency_rgb2normal2edge_occlusion=dict(
        graph_file=f"{SHARED_DIR}/shared_cloud/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth",
        dest_task=tasks.normal, final_task=tasks.edge_occlusion, percep_mode=True,
        out_channels=1),
    consistency_rgb2normal2keypoints2d=dict(
        graph_file=f"{SHARED_DIR}/shared_cloud/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth",
        dest_task=tasks.normal, final_task=tasks.keypoints2d, percep_mode=True,
        out_channels=1),
    consistency_rgb2normal2keypoints3d=dict(
        graph_file=f"{SHARED_DIR}/shared_cloud/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth",
        dest_task=tasks.normal, final_task=tasks.keypoints3d, percep_mode=True,
        out_channels=1),

    # Naive approach
    rgb2principal_curvature=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/models/rgb2principal_curvature.pth",
        downsample=5,
        dest_task=tasks.principal_curvature),
    rgb2depth_zbuffer=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/models/rgb2zdepth_buffer.pth",
        downsample=6,
        dest_task=tasks.depth_zbuffer,
        out_channels=1),
    rgb2reshading=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/models/rgb2reshade.pth",
        downsample=5,
        dest_task=tasks.reshading,
        out_channels=3),
    rgb2sobel_edges=dict(
        model_file=None,
        downsample=6,
        dest_task=tasks.sobel_edges,
        out_channels=1),
    rgb2edge_occlusion=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/models/rgb2edge_occlusion.pth",
        downsample=5,
        dest_task=tasks.edge_occlusion,
        out_channels=1),
    rgb2keypoints2d=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/models/rgb2keypoints2d_new.pth",
        downsample=3,
        dest_task=tasks.keypoints2d,
        out_channels=1),
    rgb2keypoints3d=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/models/rgb2keypoints3d.pth",
        downsample=5,
        dest_task=tasks.keypoints3d,
        out_channels=1),

    # RGB -> X -> Normal (consistency)
    finetune_rgb2principal_curvature2normal=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/models/ft_perceptual/rgb2principal_curvature.pth",
        downsample=5, percep_mode=True,
        dest_task=tasks.principal_curvature),
    finetune_rgb2depth_zbuffer2normal=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/models/ft_perceptual/rgb2depth_zbuffer.pth",
        downsample=6, percep_mode=True,
        dest_task=tasks.depth_zbuffer,
        out_channels=1),
    finetune_rgb2reshading2normal=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/models/ft_perceptual/rgb2reshading.pth",
        downsample=5, percep_mode=True,
        dest_task=tasks.reshading,
        out_channels=3),
    finetune_rgb2sobel_edges2normal=dict(
        model_file=None, percep_mode=True,
        downsample=6,
        dest_task=tasks.sobel_edges,
        out_channels=1),
    finetune_rgb2edge_occlusion2normal=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/models/ft_perceptual/rgb2edge_occlusion.pth",
        downsample=5, percep_mode=True,
        dest_task=tasks.edge_occlusion,
        out_channels=1),
    finetune_rgb2keypoints2d2normal=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/models/ft_perceptual/rgb2keypoints2d.pth",
        downsample=3, percep_mode=True,
        dest_task=tasks.keypoints2d,
        out_channels=1),
    consistency_rgb2keypoints3d2normal=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/models/ft_perceptual/rgb2keypoints3d.pth",
        downsample=5, percep_mode=True,
        dest_task=tasks.keypoints3d,
        out_channels=1),
    consistency_rgb2normal_CH_lbp_all_normaltarget_nonormalization_1=dict(
        model_file=f"{SHARED_DIR}/results_CH_lbp_all_normaltarget_nonormalization_1/graph.pth",
        downsample=6, percep_mode=True,
        dest_task=tasks.normal,
        out_channels=3),
    consistency_rgb2depth_zbuffer_CH_lbp_all_depthtarget_nonormalization_1=dict(
        model_file=f"{SHARED_DIR}/results_CH_lbp_all_depthtarget_nonormalization_1/graph.pth",
        downsample=6, percep_mode=True,
        dest_task=tasks.depth_zbuffer,
        out_channels=1),

    # On other tasks
    rgb2normal_consist=dict(
        graph_file=f"{SHARED_DIR}/shared_cloud/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth",
        downsample=6,
        out_channels=3,
        dest_task=tasks.normal),
    rgb2depth_zbuffer_consist=dict(
        graph_file=f"{SHARED_DIR}/shared_cloud/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth",
        downsample=6,
        out_channels=1, old=True,
        dest_task=tasks.depth_zbuffer),
    rgb2keypoints3d_consist=dict(
        graph_file=f"{SHARED_DIR}/shared_cloud/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth",
        downsample=6,
        out_channels=1, old=True,
        dest_task=tasks.keypoints3d),
    rgb2principal_curvature_consist=dict(
        graph_file=f"{SHARED_DIR}/shared_cloud/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth",
        downsample=6,
        out_channels=1, old=True,
        dest_task=tasks.principal_curvature),
    rgb2reshading_consist=dict(
        graph_file=f"{SHARED_DIR}/shared_cloud/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth",
        downsample=6,
        out_channels=1, old=True,
        dest_task=tasks.reshading),    
    rgb2edge_occlusion_consist=dict(
        graph_file=f"{SHARED_DIR}/shared_cloud/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth",
        downsample=6,
        out_channels=1, old=False,
        dest_task=tasks.edge_occlusion), 
    
    
    # Comparison to alternative methods
    rgb2normal_cycle_consist=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/results_BASELINES_cycleconsistency_percepstep0.02_1/n.pth",
        downsample=6, percep_mode=False,
        out_channels=3,
        dest_task=tasks.normal),
    pix2pix_rgb2normal=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/results_BASELINES_doublecycle_percepstep0.02_2/n.pth",
        downsample=6, percep_mode=False,
        out_channels=3,
        dest_task=tasks.normal),
    rgb2normal_multitask=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/models/rgb2normal_multitask.pth",
        downsample=5, percep_mode=False,
        out_channels=6,
        dest_task=tasks.normal),
    rgb2normal_imagenet=dict(
        model_file=f"{SHARED_DIR}/shared_cloud/models/rgb2normal_imagepercep.pth",
        downsample=6, percep_mode=False,
        out_channels=3,
        dest_task=tasks.normal),
    rgb2normal_geonet=dict(
        graph_file=f"{SHARED_DIR}/results_geonet_lr1e5_1gpu_2/graph.pth",
        downsample=6, percep_mode=False,
        out_channels=3,
        dest_task=tasks.normal),
    rgb2normal_baseline=dict(
        graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth",
        downsample=6, percep_mode=False,
        out_channels=3,
        dest_task=tasks.normal),
    # run_eval_suite("r2n-m", model_file=f"{MODELS_DIR}/rgb2normal_multitask.pth", logger=logger, multitask=True)
    # run_eval_suite("cycle", model_file=f"{SHARED_DIR}/results_BASELINES_doublecycle_percepstep0.02_2/n.pth", logger=logger)
    # run_eval_suite("cycle-cons", model_file=f"{SHARED_DIR}/results_BASELINES_cycleconsistency_percepstep0.02_1/n.pth", logger=logger)
    #run_eval_suite("r2n-i", model_file=f"{MODELS_DIR}/rgb2normal_imagepercep.pth", logger=logger)
    # run_eval_suite("r2n-r", model_file=f"{MODELS_DIR}/rgb2normal_random.pth", logger=logger)
    # run_eval_suite("unet-b", graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth", logger=logger)
    # run_eval_suite("unet-pc", model_file=f"{MODELS_DIR}/unet_percepstep_0.01.pth", logger=logger, old=True)
    # run_eval_suite("unet-pe", model_file=f"{MODELS_DIR}/unet_percepstep_0.1.pth", logger=logger, old=True)
    # run_eval_suite("mp-5-avg", graph_file=f"{SHARED_DIR}/results_LBP_multipercep_32/graph.pth", logger=logger)
    # run_eval_suite("unet-b-10k", graph_file=f"{SHARED_DIR}/results_SAMPLEFF_baseline10k_4/graph.pth", logger=logger)
    # run_eval_suite("unet-pc-10k", graph_file=f"{SHARED_DIR}/results_SAMPLEFF_consistency10k_8/graph.pth", logger=logger)
    # run_eval_suite("std-1", model_file=f"{SHARED_DIR}/results_STD_baseline_cont1_3/n.pth", logger=logger)
    # run_eval_suite("std-2", model_file=f"{SHARED_DIR}/results_STD_baseline_cont2_4/n.pth", logger=logger)
    # run_eval_suite("std-3", model_file=f"{SHARED_DIR}/results_STD_baseline_cont3_2/n.pth", logger=logger)
    # run_eval_suite("cycle", model_file=f"{SHARED_DIR}/results_BASELINESV2_cycle_percepstep0.01_cont_2/n.pth", logger=logger)
    #run_eval_suite("cycle-cons", model_file=f"{SHARED_DIR}/results_BASELINESV2_cycleconsistency_percepstep0.01_cont_1/n.pth", logger=logger)
)
def make_video(frame_dir=f"{SHARED_DIR}/assets/input_frames",
               batch_size = 64,
               n_workers  = 16,
               config_to_run='rgb2principal_curvature',
               output_dir=f'{SHARED_DIR}/assets/normal_consistency/rgb2depth_zbuffer'):

    frame_loader = torch.utils.data.DataLoader(
        ImageDataset(tasks=[tasks.rgb(size=256)], data_dir=frame_dir),
        batch_size=batch_size,
        num_workers=n_workers, shuffle=False, pin_memory=True
    )
    print(f"Loaded {len(frame_loader)} batches ({batch_size} each) from directory: {frame_dir}")
    
    with torch.no_grad():
        # results = run_viz_suite("imagepercep", frame_loader, model_file=f"{SHARED_DIR}/shared_cloud/models/rgb2normal_imagepercep.pth", logger=logger, percep_mode=False)
        #results = run_viz_suite("imagepercep",
        #                        frame_loader, model_file=f"{SHARED_DIR}/shared_cloud/models/rgb2principal_curvature.pth",
        #                        logger=None, percep_mode=True, downsample=5,
        #                        dest_task=tasks.principal_curvature)

        #results = run_viz_suite("imagepercep", frame_loader, percep_mode=True, **configs[config_to_run])

        results = run_viz_suite("imagepercep", frame_loader, **configs[config_to_run])


    # start = time.time()
    # p = mp.Pool(8)
    # zipped = [(im, f"{SHARED_DIR}/assets/output_frames/output{i}.png") for i, im in enumerate(results)]
    # p.map(translate, zipped)
    # elapsed = time.time() - start
    # print(f"MP = {elapsed}")

    start = time.time()
    print("Finished conversion: Saving...")
    for i in range(results.size(0)):
        torchvision.utils.save_image(results[i], f"{output_dir}/output{i:05}.png")
    elapsed = time.time() - start
    print(f"Single = {elapsed}")

if __name__ == "__main__":
    Fire(make_video)
    