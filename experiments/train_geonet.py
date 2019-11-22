import os, sys, math, random, itertools
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from plotting import *
from energy import get_energy_loss
from graph import GeoNetTaskGraph, TaskGraph, Discriminator
from logger import Logger, VisdomLogger
from datasets import TaskDataset, load_train_val, load_test, load_ood
from task_configs import tasks, RealityTask
from transfers import functional_transfers
from evaluation import run_eval_suite

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from functools import partial
from fire import Fire

import IPython

def main(
    loss_config="geonet", mode="geonet", visualize=False,
    fast=False, batch_size=None, 
    subset_size=None, early_stopping=float('inf'),
    max_epochs=800, **kwargs,
):

    print(kwargs)
    # CONFIG
    batch_size = batch_size or (4 if fast else 64)
    energy_loss = get_energy_loss(config=loss_config, mode=mode, **kwargs)

    # DATA LOADING

    train_dataset, val_dataset, train_step, val_step = load_train_val(
        energy_loss.get_tasks("train"),
        batch_size=batch_size, fast=fast,
        subset_size=subset_size,
    )
    test_set = load_test(energy_loss.get_tasks("test"))
    ood_set = load_ood(energy_loss.get_tasks("ood"))
    print (train_step, val_step)

    
    # GRAPH
    print(energy_loss.tasks)
    
    print('train tasks', energy_loss.get_tasks("train"))
    train = RealityTask("train", train_dataset, batch_size=batch_size, shuffle=True)
    print('val tasks', energy_loss.get_tasks("val"))
    val = RealityTask("val", val_dataset, batch_size=batch_size, shuffle=True)
    print('test tasks', energy_loss.get_tasks("test"))
    test = RealityTask.from_static("test", test_set, energy_loss.get_tasks("test"))
    print('ood tasks', energy_loss.get_tasks("ood"))
    ood = RealityTask.from_static("ood", ood_set, energy_loss.get_tasks("ood"))
    print('done')


    # GRAPH
    realities = [train, val, test, ood]
#     graph = GeoNetTaskGraph(tasks=energy_loss.tasks, realities=realities, pretrained=False)

    graph = GeoNetTaskGraph(tasks=energy_loss.tasks, realities=realities, pretrained=True)


    # n(x)/norm(n(x))
    # (f(n(x)) / RC(x)) 
    #graph.compile(torch.optim.Adam, grad_clip=2.0, lr=1e-5, weight_decay=0e-6, amsgrad=True)
    graph.compile(torch.optim.Adam, grad_clip=5.0, lr=4e-5, weight_decay=2e-6, amsgrad=True)
    #graph.compile(torch.optim.Adam, grad_clip=5.0, lr=1e-6, weight_decay=2e-6, amsgrad=True)
    #graph.compile(torch.optim.Adam, grad_clip=5.0, lr=1e-5, weight_decay=2e-6, amsgrad=True)


    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
    energy_loss.logger_hooks(logger)
    best_val_loss, stop_idx = float('inf'), 0

    # TRAINING
    for epochs in range(0, max_epochs):

        logger.update("epoch", epochs)
        try:
            energy_loss.plot_paths(graph, logger, realities, prefix="start" if epochs == 0 else "")
        except:
            pass
        if visualize: return

        graph.train()
        print('training for', train_step, 'steps')
        for _ in range(0, train_step):
            try:
                train_loss = energy_loss(graph, realities=[train])
                train_loss = sum([train_loss[loss_name] for loss_name in train_loss])

                graph.step(train_loss)
                train.step()
                logger.update("loss", train_loss)
            except NotImplementedError:
                pass

        graph.eval()
        for _ in range(0, val_step):
            try:
                with torch.no_grad():
                    val_loss = energy_loss(graph, realities=[val])
                    val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
                val.step()
                logger.update("loss", val_loss)
            except NotImplementedError:
                pass

        energy_loss.logger_update(logger)
        logger.step()

        stop_idx += 1 
        try:
            curr_val_loss = (logger.data["val_mse : N(rgb) -> normal"][-1] + logger.data["val_mse : D(rgb) -> depth"][-1])
            if curr_val_loss < best_val_loss:
                print ("Better val loss, reset stop_idx: ", stop_idx)
                best_val_loss, stop_idx = curr_val_loss, 0
                energy_loss.plot_paths(graph, logger, realities, prefix="best")
                graph.save(f"{RESULTS_DIR}/graph.pth")
        except NotImplementedError:
            pass
    
        if stop_idx >= early_stopping:
            print ("Stopping training now")
            return

if __name__ == "__main__":
    Fire(main)


