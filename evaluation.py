
import numpy as np
import os, sys, math, random, glob, time, itertools
from fire import Fire

os.environ['CUDA_VISIBLE_DEVICES']='6,7'

os.environ['CUDA_VISIBLE_DEVICES']='0,1'


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils import *
from models import TrainableModel, DataParallelModel
from task_configs import RealityTask, get_task, get_model, tasks
from transfers import Transfer
from logger import Logger, VisdomLogger
from graph import TaskGraph
from datasets import TaskDataset, ImageDataset, SintelDataset, load_test, load_ood
import transforms

torch.manual_seed(229) # cpu  vars
torch.cuda.manual_seed_all(229) # gpu vars

from modules.unet import UNet, UNetOld, UNetOld2
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





def run_eval_suite(name, dest_task=tasks.normal, graph_file=None, model_file=None, logger=None, sample=800, show_images=False, old=False):
    
    if graph_file is not None:
        graph = TaskGraph(tasks=[tasks.rgb, dest_task], pretrained=False)
        graph.load_weights(graph_file)
        model = graph.edge(tasks.rgb, dest_task).load_model()
    elif old:
        model = DataParallelModel.load(UNetOld().cuda(), model_file)
    elif model_file is not None:
        #model = DataParallelModel.load(UNet(downsample=5).cuda(), model_file)
        model = DataParallelModel.load(UNet(downsample=6).cuda(), model_file)
    else:
        model = Transfer(src_task=tasks.normal, dest_task=dest_task).load_model()

    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)

    dataset = ValidationMetrics("almena", dest_task=dest_task)
    result = dataset.evaluate(model, sample=800)
    logger.text (name + ": " + str(result))


def run_viz_suite(name, data, dest_task=tasks.depth_zbuffer, graph_file=None, model_file=None, logger=None, old=False, multitask=False, percep_mode=None):
    
    if graph_file is not None:
        graph = TaskGraph(tasks=[tasks.rgb, dest_task], pretrained=False)
        graph.load_weights(graph_file)
        model = graph.edge(tasks.rgb, dest_task).load_model()
    elif old:
        model = DataParallelModel.load(UNetOld().cuda(), model_file)
    elif multitask:
        model = DataParallelModel.load(UNet(downsample=5, out_channels=6).cuda(), model_file)
    elif model_file is not None:
        print('here')
        #model = DataParallelModel.load(UNet(downsample=5).cuda(), model_file)
        model = DataParallelModel.load(UNet(downsample=6).cuda(), model_file)
    else:
        model = Transfer(src_task=tasks.rgb, dest_task=dest_task).load_model()

    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)

    # DATA LOADING 1
    results = model.predict(data)[:, -3:].clamp(min=0, max=1)
    if results.shape[1] == 1:
        results = torch.cat([results]*3, dim=1)
    
    if percep_mode:
        percep_model = Transfer(src_task=dest_task, dest_task=tasks.normal).load_model()
        percep_model.eval()
        eval_loader = torch.utils.data.DataLoader(
                 torch.utils.data.TensorDataset(results), batch_size=16,
                 num_workers=16, shuffle=False, pin_memory=True
              )
        final_preds = []
        for preds, in eval_loader:
            print('preds shape', preds.shape)
            final_preds += [percep_model.forward(preds[:, -3:])]
        results = torch.cat(final_preds, dim=0)

    return results


def run_perceptual_eval_suite(name, intermediate_task=tasks.normal, dest_task=tasks.normal, graph_file=None, model_file=None, logger=None, sample=800, show_images=False, old=False, perceptual_transfer=None, multitask=False):
    
    if perceptual_transfer is None:
        percep_model = Transfer(src_task=intermediate_task, dest_task=dest_task).load_model()


    if graph_file is not None:
        graph = TaskGraph(tasks=[tasks.rgb, intermediate_task], pretrained=False)
        graph.load_weights(graph_file)
        model = graph.edge(tasks.rgb, intermediate_task).load_model()
    elif old:
        model = DataParallelModel.load(UNetOld().cuda(), model_file)
    elif multitask:
        print('running multitask')
        model = DataParallelModel.load(UNet(downsample=5, out_channels=6).cuda(), model_file)
    elif model_file is not None:
        #model = DataParallelModel.load(UNet(downsample=5).cuda(), model_file)
        model = DataParallelModel.load(UNet(downsample=6).cuda(), model_file)
    else:
        model = Transfer(src_task=tasks.rgb, dest_task=intermediate_task).load_model()

    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)

    dataset = ValidationMetrics("almena", dest_task=dest_task)
    result = dataset.evaluate_with_percep(model, sample=800, percep_model=percep_model)
    logger.text (name + ": " + str(result))



if __name__ == "__main__":
    logger = VisdomLogger("eval", env=JOB)
    
    # Geonet
    #run_perceptual_eval_suite("geonet-v1", intermediate_task=tasks.normal, dest_task=tasks.depth_zbuffer, graph_file=f"{SHARED_DIR}/results_geonet_lr1e5_1gpu_2/graph.pth", logger=logger, )
    #run_perceptual_eval_suite("geonet-v1", intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal, graph_file=f"{SHARED_DIR}/results_geonet_lr1e5_1gpu_2/graph.pth", logger=logger, )

    # Baseline
    # run_perceptual_eval_suite("unet-b",, intermediate_task=tasks.normal, dest_task=tasks.depth_zbuffer, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth", logger=logger, )
    # run_perceptual_eval_suite("unet-b",, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal, bgraph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth", logger=logger, )
    
    
    
    
    
    SHARED_DIR = SHARED_DIR + "/shared_cloud"
    MODELS_DIR = SHARED_DIR + "/models"
    
    #################################
    # Perceptual metrics
    #################################

    # ------------------------------
    # Depth (via normals)
    # -------------------------------
    #run_perceptual_eval_suite("pix2pix", intermediate_task=tasks.normal, dest_task=tasks.depth_zbuffer, model_file=f"{SHARED_DIR}/results_BASELINES_doublecycle_percepstep0.02_2/n.pth", logger=logger, )
    #run_perceptual_eval_suite("multi-task", intermediate_task=tasks.normal, dest_task=tasks.depth_zbuffer, model_file=f"{MODELS_DIR}/rgb2normal_multitask.pth", logger=logger, multitask=True)
    #run_perceptual_eval_suite("cycle-consistency", intermediate_task=tasks.normal, dest_task=tasks.depth_zbuffer, model_file=f"{SHARED_DIR}/results_BASELINES_cycleconsistency_percepstep0.02_1/n.pth", logger=logger, )
    #run_perceptual_eval_suite("perceptual-imagenet", intermediate_task=tasks.normal, dest_task=tasks.depth_zbuffer, model_file=f"{MODELS_DIR}/rgb2normal_imagepercep.pth", logger=logger, )
    #run_perceptual_eval_suite("perceptual-random", intermediate_task=tasks.normal, dest_task=tasks.depth_zbuffer, model_file=f"{MODELS_DIR}/rgb2normal_random.pth", logger=logger, )
    #run_perceptual_eval_suite("unet-baseline-10k", intermediate_task=tasks.normal, dest_task=tasks.depth_zbuffer, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_baseline10k_4/graph.pth", logger=logger, )
    #run_perceptual_eval_suite("unet-pc-10k", intermediate_task=tasks.normal, dest_task=tasks.depth_zbuffer, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_consistency10k_8/graph.pth", logger=logger, )   
    #run_perceptual_eval_suite("consistency", intermediate_task=tasks.normal, dest_task=tasks.depth_zbuffer, model_file=f"{MODELS_DIR}/unet_percepstep_0.01.pth", old=True, logger=logger, )
    #run_perceptual_eval_suite("consistency", intermediate_task=tasks.normal, dest_task=tasks.depth_zbuffer, model_file=f"{MODELS_DIR}/unet_percepstep_0.1.pth", old=True, logger=logger, )



    # ------------------------------
    # RESHADE (via normals)
    # -------------------------------
    #run_perceptual_eval_suite("geonet-v1", intermediate_task=tasks.normal, dest_task=tasks.reshading, graph_file=f"{SHARED_DIR}/results_geonet_lr1e5_1gpu_2/graph.pth", logger=logger, )
    #run_perceptual_eval_suite("unet-b", intermediate_task=tasks.normal, dest_task=tasks.reshading, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth", logger=logger, )
    #run_perceptual_eval_suite("pix2pix", intermediate_task=tasks.normal, dest_task=tasks.reshading, model_file=f"{SHARED_DIR}/results_BASELINES_doublecycle_percepstep0.02_2/n.pth", logger=logger, )
    #run_perceptual_eval_suite("multi-task", intermediate_task=tasks.normal, dest_task=tasks.reshading, model_file=f"{MODELS_DIR}/rgb2normal_multitask.pth", logger=logger, multitask=True)
    #run_perceptual_eval_suite("cycle-consistency", intermediate_task=tasks.normal, dest_task=tasks.reshading, model_file=f"{SHARED_DIR}/results_BASELINES_cycleconsistency_percepstep0.02_1/n.pth", logger=logger, )
    #run_perceptual_eval_suite("perceptual-imagenet", intermediate_task=tasks.normal, dest_task=tasks.reshading, model_file=f"{MODELS_DIR}/rgb2normal_imagepercep.pth", logger=logger, )
    #run_perceptual_eval_suite("perceptual-random", intermediate_task=tasks.normal, dest_task=tasks.reshading, model_file=f"{MODELS_DIR}/rgb2normal_random.pth", logger=logger, )
    #run_perceptual_eval_suite("unet-baseline-10k", intermediate_task=tasks.normal, dest_task=tasks.reshading, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_baseline10k_4/graph.pth", logger=logger, )
    #run_perceptual_eval_suite("unet-pc-10k", intermediate_task=tasks.normal, dest_task=tasks.reshading, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_consistency10k_8/graph.pth", logger=logger, )   
    #run_perceptual_eval_suite("consistency", intermediate_task=tasks.normal, dest_task=tasks.reshading, model_file=f"{MODELS_DIR}/unet_percepstep_0.01.pth", old=True, logger=logger, )
    #run_perceptual_eval_suite("consistency", intermediate_task=tasks.normal, dest_task=tasks.reshading, model_file=f"{MODELS_DIR}/unet_percepstep_0.1.pth", old=True, logger=logger, )




    # ------------------------------
    # CURVATURE (via normals)
    # -------------------------------
    #run_perceptual_eval_suite("geonet-v1", intermediate_task=tasks.normal, dest_task=tasks.principal_curvature, graph_file=f"{SHARED_DIR}/results_geonet_lr1e5_1gpu_2/graph.pth", logger=logger, )
    #run_perceptual_eval_suite("unet-b", intermediate_task=tasks.normal, dest_task=tasks.principal_curvature, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth", logger=logger, )
    #run_perceptual_eval_suite("multi-task", intermediate_task=tasks.normal, dest_task=tasks.principal_curvature, model_file=f"{MODELS_DIR}/rgb2normal_multitask.pth", logger=logger, multitask=True)
    #run_perceptual_eval_suite("pix2pix", intermediate_task=tasks.normal, dest_task=tasks.principal_curvature, model_file=f"{SHARED_DIR}/results_BASELINES_doublecycle_percepstep0.02_2/n.pth", logger=logger, )
    #run_perceptual_eval_suite("cycle-consistency", intermediate_task=tasks.normal, dest_task=tasks.principal_curvature, model_file=f"{SHARED_DIR}/results_BASELINES_cycleconsistency_percepstep0.02_1/n.pth", logger=logger, )
    #run_perceptual_eval_suite("perceptual-imagenet", intermediate_task=tasks.normal, dest_task=tasks.principal_curvature, model_file=f"{MODELS_DIR}/rgb2normal_imagepercep.pth", logger=logger, )
    #run_perceptual_eval_suite("perceptual-random", intermediate_task=tasks.normal, dest_task=tasks.principal_curvature, model_file=f"{MODELS_DIR}/rgb2normal_random.pth", logger=logger, )
    #run_perceptual_eval_suite("unet-baseline-10k", intermediate_task=tasks.normal, dest_task=tasks.principal_curvature, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_baseline10k_4/graph.pth", logger=logger, )
    #run_perceptual_eval_suite("unet-pc-10k", intermediate_task=tasks.normal, dest_task=tasks.principal_curvature, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_consistency10k_8/graph.pth", logger=logger, )   
    #run_perceptual_eval_suite("consistency", intermediate_task=tasks.normal, dest_task=tasks.principal_curvature, model_file=f"{MODELS_DIR}/unet_percepstep_0.01.pth", old=True, logger=logger, )
    #run_perceptual_eval_suite("consistency 0.1", intermediate_task=tasks.normal, dest_task=tasks.principal_curvature, model_file=f"{MODELS_DIR}/unet_percepstep_0.1.pth", old=True, logger=logger, )


    
    # ------------------------------
    # Normals (via depth)
    # -------------------------------
    # run_perceptual_eval_suite("pix2pix",, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal, graph_file=f"{SHARED_DIR}/shared_cloudresults_SAMPLEFF_full_data_baseline_3/graph.pth", logger=logger, )
    # run_perceptual_eval_suite("multi-task",, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth", logger=logger, 
    # run_perceptual_eval_suite("cycle-consistency",, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal, graph_file=f"{SHARED_DIR}/results_BASELINES_cycleconsistency_percepstep0.02_1/graph.pth", logger=logger, )
    # run_perceptual_eval_suite("perceptual-imagenet",, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth", logger=logger, )
    # run_perceptual_eval_suite("perceptual-random",, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth", logger=logger, )
    # run_perceptual_eval_suite("unet-baseline-10k",, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_baseline10k_4/graph.pth", logger=logger, )
    # run_perceptual_eval_suite("unet-pc-10k",, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_consistency10k_8/graph.pth", logger=logger, )
    # run_perceptual_eval_suite("consistency", intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal, graph_file=f"{SHARED_DIR}/results_LBP_multipercep_latwinrate_depthtarget_3/graph.pth", logger=logger, )


    
    
        
    #################################
    # DIRECT metrics
    #################################

    
    
    # --------
    # Normals
    # --------
    # run_eval_suite("unet-b", graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth", logger=logger)
    # Consistency
    # run_perceptual_eval_suite("consistency", intermediate_task=tasks.normal, dest_task=tasks.depth_zbuffer, model_file=f"{MODELS_DIR}/unet_percepstep_0.01.pth", old=True, logger=logger, )
    #run_perceptual_eval_suite("depth-mp2", intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal, graph_file=f"{SHARED_DIR}/results_LBP_multipercep8_winrate_standardized_depthtarget2_2/graph.pth", logger=logger)
    # run_eval_suite("dorn", graph_file=f"../{SHARED_DIR}/results_depth_baseline_DORN_min1_lr_0.00003_1/graph.th", logger=logger, old=True)
    # run_eval_suite("unet-pc", model_file=f"{MODELS_DIR}/unet_percepstep_0.1.pth", logger=logger, old=True)
    # run_eval_suite("mp-8-lat", graph_file=f"{SHARED_DIR}/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth", logger=logger)


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
    
    #run_eval_suite("geonet-v1", model_file=f"{SHARED_DIR}/results_geonet_lr1e5_1gpu_2/graph_normals.pth", logger=logger)

    # run_eval_suite("geonet-v1", graph_file=f"{SHARED_DIR}/results_geonet_lr1e5_1gpu_2/graph.pth", logger=logger)
    # run_eval_suite("geonet-v1", dest_task=tasks.depth_zbuffer, graph_file=f"{SHARED_DIR}/results_geonet_lr1e5_1gpu_2/graph.pth", logger=logger)



    # logger.images_grouped(images, "grouped_data", resize=256)

    run_perceptual_eval_suite("depth-1",  graph_file=f"{SHARED_DIR}/results_CH_gc_lbp_all_percepsnormalreshadingonly_depthtarget_1/graph.pth", logger=logger, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal)
    run_perceptual_eval_suite("depth-2", graph_file=f"{SHARED_DIR}/results_CH_gc_lbp_lat_winrate_unitmean_depthtarget_1/graph.pth", logger=logger, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal)
    run_perceptual_eval_suite("depth-3",  graph_file=f"{SHARED_DIR}/results_CH_gc_lbp_lat_winrate_5perceps_1/graph.pth", logger=logger, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal)


    
    run_perceptual_eval_suite("depth-1",  graph_file=f"{SHARED_DIR}/results_CH_gc_lbp_lat_winrate_depthtarget_1/graph.pth", logger=logger, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal)
    run_perceptual_eval_suite("depth-2", graph_file=f"{SHARED_DIR}/results_CH_gc_lbp_lat_winrate_5perceps_1/graph.pth", logger=logger, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal)
    #run_eval_suite("depth-3", dest_task=tasks.depth_zbuffer, graph_file=f"{SHARED_DIR}/results_CH_gc_lbp_all_depthtarget_1/graph.pth", logger=logger)
    run_perceptual_eval_suite("depth-4", graph_file=f"{SHARED_DIR}/results_CH_gc_lbp_lat_winrate_5percepsk2_nonormalization_depthtarget_1/graph.pth", logger=logger, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal)
    run_perceptual_eval_suite("depth-5", graph_file=f"{SHARED_DIR}/results_CH_gc_lbp_all_winrate_depthtarget_5perceps_1/graph.pth", logger=logger, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal)
    run_perceptual_eval_suite("depth-6", graph_file=f"{SHARED_DIR}/results_CH_lbp_all_depthtarget_nonormalization_1/graph.pth", logger=logger, intermediate_task=tasks.depth_zbuffer, dest_task=tasks.normal)


    # run_eval_suite("depth-b", dest_task=tasks.depth_zbuffer, model_file=f"{MODELS_DIR}/rgb2zdepth_buffer.pth", logger=logger)
    # run_eval_suite("depth-mp", dest_task=tasks.depth_zbuffer, graph_file=f"{SHARED_DIR}/results_LBP_multipercep_latwinrate_depthtarget_3/graph.pth", logger=logger)
    # run_eval_suite("depth-mp2", dest_task=tasks.depth_zbuffer, graph_file=f"{SHARED_DIR}/results_LBP_multipercep8_winrate_standardized_depthtarget2_2/graph.pth", logger=logger)

    
    #run_eval_suite("reshade-b", dest_task=tasks.reshading, model_file=f"{MODELS_DIR}/rgb2reshade.pth", logger=logger)
    #run_eval_suite("reshade-mp", dest_task=tasks.reshading, graph_file=f"{SHARED_DIR}/results_LBP_multipercep_latwinrate_reshadingtarget_6/graph.pth", logger=logger)
    # # # run_eval_suite("apc-tr", graph_file=f"{SHARED_DIR}/results_2FF_conservative_full_triangle_4/graph.pth", logger=logger, old=True)

    # run_eval_suite("unet-b-old", model_file=f"{MODELS_DIR}/unet_baseline.pth", logger=logger, old=True)
    # run_eval_suite("cycle-cons", model_file=f"{SHARED_DIR}/results_BASELINES_cycleconsistency_percepstep0.02_1/n.pth", logger=logger)
    # run_eval_suite("cycle", model_file=f"{SHARED_DIR}/results_BASELINES_doublecycle_percepstep0.02_2/n.pth", logger=logger)
    
    # run_eval_suite("r2n-d", model_file=f"{MODELS_DIR}/rgb2normal_discriminator.pth", logger=logger)
    # run_eval_suite("r2n-i", model_file=f"{MODELS_DIR}/rgb2normal_imagepercep.pth", logger=logger)
    # run_eval_suite("r2n-r", model_file=f"{MODELS_DIR}/rgb2normal_random.pth", logger=logger)
    # # run_eval_suite("r2n-m", model_file=f"{MODELS_DIR}/rgb2normal_multitask.pth", logger=logger)

    # run_eval_suite("unet-pe", graph_file=f"{SHARED_DIR}/results_LBP_percepedge2_4/graph.pth", logger=logger)
    # run_eval_suite("unet-pc", model_file=f"{MODELS_DIR}/unet_percepstep_0.1.pth", logger=logger, old=True)
    # run_eval_suite("unet-pk3", model_file=f"{SHARED_DIR}/results_LBP_percepkeypoints3d_6/graph.pth", logger=logger, old=True)
    
    # run_eval_suite("unet-pe", graph_file=f"{SHARED_DIR}/results_LBP_percepedge2_4/graph.pth", logger=logger)
    # run_eval_suite("unet-pc", model_file=f"{MODELS_DIR}/unet_percepstep_0.1.pth", logger=logger, old=True)
    # run_eval_suite("unet-pk3", graph_file=f"{SHARED_DIR}/results_LBP_percepkeypoints3d_6/graph.pth", logger=logger, old=True)

    # run_eval_suite("mp-5-avg", graph_file=f"{SHARED_DIR}/results_LBP_multipercep_32/graph.pth", logger=logger)
    # run_eval_suite("mp-8-lat", graph_file=f"{SHARED_DIR}/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth", logger=logger)
    # run_eval_suite("mp-8-rnd", graph_file=f"{SHARED_DIR}/results_LBP_multipercep_rndv2_2/graph.pth", logger=logger)

    # run_eval_suite("unet-p0.1-graph", graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_percepstep0.1_15/graph.pth", logger=logger)
    # run_eval_suite("unet-b", graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth", logger=logger)

    

    # test_set = load_test([tasks.rgb(size=320), tasks.normal(size=320)], sample=8)
    # ood_set = load_ood([tasks.rgb(size=320), tasks.normal(size=320)], sample=12)
    
    # ood_loader = torch.utils.data.DataLoader(
    #     ImageDataset(tasks=[tasks.rgb(size=320), tasks.normal(size=320)], data_dir=f"{SHARED_DIR}/ood_images"),
    #     batch_size=16,
    #     num_workers=16, shuffle=False, pin_memory=True
    # )
    # dataset = list(ood_loader)
    # data = [test_set[0][:, 0:3], ood_set[0][:, 0:3], 
    #     dataset[0][0][:, 0:3], dataset[1][0][:, 0:3],
    #     dataset[2][0][:, 0:3], dataset[3][0][:, 0:3],
    #     dataset[4][0][:, 0:3], dataset[5][0][:, 0:3],
    # ]
    # target = [test_set[1][:, 0:3], ood_set[1][:, 0:3], 
    #     dataset[0][1][:, 0:3], dataset[1][1][:, 0:3],
    #     dataset[2][1][:, 0:3], dataset[3][1][:, 0:3],
    #     dataset[4][1][:, 0:3], dataset[5][1][:, 0:3],
    # ]
    # images = torch.cat(data, dim=0)
    # targets = torch.cat(target, dim=0)

    # images = [images, targets]
    # images += [run_viz_suite("r2n-m", data, model_file=f"{MODELS_DIR}/rgb2normal_multitask.pth", logger=logger, multitask=True)]
    # # images += [run_viz_suite("cycle", data, model_file=f"{SHARED_DIR}/results_BASELINES_doublecycle_percepstep0.02_2/n.pth", logger=logger)]
    # images += [run_viz_suite("cycle-cons", data, model_file=f"{SHARED_DIR}/results_BASELINES_cycleconsistency_percepstep0.02_1/n.pth", logger=logger)]
    # images += [run_viz_suite("r2n-i", data, model_file=f"{MODELS_DIR}/rgb2normal_imagepercep.pth", logger=logger)]
    # images += [run_viz_suite("r2n-r", data, model_file=f"{MODELS_DIR}/rgb2normal_random.pth", logger=logger)]
    # images += [run_viz_suite("unet-b", data, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth", logger=logger)]
    # images += [run_viz_suite("unet-pc", data, model_file=f"{MODELS_DIR}/unet_percepstep_0.1.pth", logger=logger, old=True)]
    # images += [run_viz_suite("unet-pe", data, model_file=f"{MODELS_DIR}/unet_percepstep_0.1.pth", logger=logger, old=True)]
    # images += [run_viz_suite("mp-5-avg", data, graph_file=f"{SHARED_DIR}/results_LBP_multipercep_32/graph.pth", logger=logger)]
    # images += [run_viz_suite("mp-8-lat", data, graph_file=f"{SHARED_DIR}/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth", logger=logger)]
    # # run_eval_suite("unet-b-1m", model_file=f"{SHARED_DIR}/results_SAMPLEFF_baseline1m_3/n.pth", logger=logger)
    # # run_eval_suite("unet-pc-1m", graph_file=f"{SHARED_DIR}/results_SAMPLEFF_consistency1m_25/graph.pth", logger=logger)
    # logger.images_grouped(images, "r2n-multi, cycle-cons, r2n-img, r2n-rnd, unet-base, unet-pc, unet-pe, mp-5-avg, mp-8-lat-win", resize=320)
    
    # run_eval_suite("reshade-b", data, dest_task=tasks.reshading, model_file=f"{MODELS_DIR}/rgb2reshade.pth", logger=logger)
    # run_eval_suite("reshade-mp", data, dest_task=tasks.reshading, graph_file=f"{SHARED_DIR}/results_LBP_multipercep_latwinrate_reshadingtarget_6/graph.pth", logger=logger)
    # run_eval_suite("apc-tr", data, graph_file=f"{SHARED_DIR}/results_2FF_conservative_full_triangle_4/graph.pth", logger=logger, old=True)

    # run_eval_suite("unet-b-old", model_file=f"{MODELS_DIR}/unet_baseline.pth", logger=logger, old=True)
    # run_eval_suite("cycle-cons", model_file=f"{SHARED_DIR}/results_BASELINES_cycleconsistency_percepstep0.02_1/n.pth", logger=logger)
    
    
    # run_eval_suite("r2n-d", model_file=f"{MODELS_DIR}/rgb2normal_discriminator.pth", logger=logger)
    # run_eval_suite("r2n-i", model_file=f"{MODELS_DIR}/rgb2normal_imagepercep.pth", logger=logger)
    # run_eval_suite("r2n-r", model_file=f"{MODELS_DIR}/rgb2normal_random.pth", logger=logger)

    # run_eval_suite("unet-pe", graph_file=f"{SHARED_DIR}/results_LBP_percepedge2_4/graph.pth", logger=logger)
    # run_eval_suite("unet-pc", model_file=f"{MODELS_DIR}/unet_percepstep_0.1.pth", logger=logger, old=True)
    # run_eval_suite("unet-pk3", model_file=f"{SHARED_DIR}/results_LBP_percepkeypoints3d_6/graph.pth", logger=logger, old=True)
    
    # run_eval_suite("unet-pe", graph_file=f"{SHARED_DIR}/results_LBP_percepedge2_4/graph.pth", logger=logger)
    # run_eval_suite("unet-pc", model_file=f"{MODELS_DIR}/unet_percepstep_0.1.pth", logger=logger, old=True)
    # run_eval_suite("unet-pk3", graph_file=f"{SHARED_DIR}/results_LBP_percepkeypoints3d_6/graph.pth", logger=logger, old=True)

    # run_eval_suite("mp-5-avg", graph_file=f"{SHARED_DIR}/results_LBP_multipercep_32/graph.pth", logger=logger)
    # run_eval_suite()
    # run_eval_suite("mp-8-rnd", graph_file=f"{SHARED_DIR}/results_LBP_multipercep_rndv2_2/graph.pth", logger=logger)

    # run_eval_suite("unet-p0.1-graph", graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_percepstep0.1_15/graph.pth", logger=logger)
    # run_eval_suite("unet-b", graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth", logger=logger)

    # run_eval_suite("unet-b-10k", graph_file=f"{SHARED_DIR}/results_SAMPLEFF_baseline10k_4/graph.pth", logger=logger)
    # run_eval_suite("unet-pc-10k", graph_file=f"{SHARED_DIR}/results_SAMPLEFF_consistency10k_8/graph.pth", logger=logger)
    

    
    
    # Fire(run_eval_suite)


    exit(0)
    test_set = load_test([tasks.rgb(size=320), tasks.normal(size=320)], sample=8)
    ood_set = load_ood([tasks.rgb(size=320), tasks.normal(size=320)], sample=12)
    
    ood_loader = torch.utils.data.DataLoader(
        ImageDataset(tasks=[tasks.rgb(size=320), tasks.normal(size=320)], data_dir=f"{SHARED_DIR}/ood_images"),
        batch_size=16,
        num_workers=16, shuffle=False, pin_memory=True
    )
    dataset = list(ood_loader)


    data = [test_set[0][:, 0:3], ood_set[0][:, 0:3], 
        dataset[0][0][:, 0:3], dataset[1][0][:, 0:3],
        dataset[2][0][:, 0:3], dataset[3][0][:, 0:3],
        dataset[4][0][:, 0:3], dataset[5][0][:, 0:3],
    ]
    target = [test_set[1][:, 0:3], ood_set[1][:, 0:3], 
        dataset[0][1][:, 0:3], dataset[1][1][:, 0:3],
        dataset[2][1][:, 0:3], dataset[3][1][:, 0:3],
        dataset[4][1][:, 0:3], dataset[5][1][:, 0:3],
    ]
    images = torch.cat(data, dim=0)
    targets = torch.cat(target, dim=0)

    if targets.shape[1] == 1:
        targets = torch.cat([targets]*3, dim=1)
        print(targets.shape)
    print(targets.shape)
    images = [images, targets]
    images += [run_viz_suite("imagepercep", data, model_file=f"{MODELS_DIR}/rgb2normal_imagepercep.pth", logger=logger, percep_mode=False)]
    #images += [run_viz_suite("unet-b", data, graph_file=f"{SHARED_DIR}/results_SAMPLEFF_full_data_baseline_3/graph.pth", logger=logger)]
    #images += [run_viz_suite("unet-pc", data, model_file=f"{MODELS_DIR}/unet_percepstep_0.1.pth", logger=logger, old=True)]
    #images += [run_viz_suite("geonet", data, graph_file=f"{SHARED_DIR}/results_geonet_lr1e5_1gpu_2/graph.pth", logger=logger)]
    # run_eval_suite("unet-b-1m", model_file=f"{SHARED_DIR}/results_SAMPLEFF_baseline1m_3/n.pth", logger=logger)
    # run_eval_suite("unet-pc-1m", graph_file=f"{SHARED_DIR}/results_SAMPLEFF_consistency1m_25/graph.pth", logger=logger)
    for image in images:
        print(image.shape)
    logger.images_grouped(images, "imagepercep", resize=320)