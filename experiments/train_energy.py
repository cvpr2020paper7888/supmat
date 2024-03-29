import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from plotting import *
from energy import get_energy_loss
from graph import TaskGraph, Discriminator
from logger import Logger, VisdomLogger
from datasets import TaskDataset, load_train_val, load_test, load_ood
from task_configs import tasks, RealityTask
from evaluation import run_eval_suite

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from functools import partial
from fire import Fire

import IPython

def main(
	loss_config="conservative_full", mode="standard", visualize=False,
	pretrained=True, finetuned=False, fast=False, batch_size=None, 
	ood_batch_size=None, subset_size=None,
	cont=f"{MODELS_DIR}/conservative/conservative.pth", 
	max_epochs=800, **kwargs,
):
	
	# CONFIG
	batch_size = batch_size or (4 if fast else 64)
	energy_loss = get_energy_loss(config=loss_config, mode=mode, **kwargs)

	# DATA LOADING

	train_dataset, val_dataset, train_step, val_step = load_train_val(
		energy_loss.get_tasks("train"),
		batch_size=batch_size, fast=fast,
		subset_size=subset_size
	)
	test_set = load_test(energy_loss.get_tasks("test"))
	ood_set = load_ood(energy_loss.get_tasks("ood"))
	train_step, val_step = 4, 4
	print (train_step, val_step)
	
	train = RealityTask("train", train_dataset, batch_size=batch_size, shuffle=True)
	val = RealityTask("val", val_dataset, batch_size=batch_size, shuffle=True)
	test = RealityTask.from_static("test", test_set, energy_loss.get_tasks("test"))
	ood = RealityTask.from_static("ood", ood_set, energy_loss.get_tasks("ood"))

	# GRAPH
	realities = [train, val, test, ood]
	graph = TaskGraph(tasks=energy_loss.tasks + realities, 
		freeze_list=energy_loss.freeze_list, finetuned=finetuned)
	graph.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
	if not USE_RAID: graph.load_weights(cont)

	# LOGGING
	logger = VisdomLogger("train", env=JOB)
	logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
	logger.add_hook(lambda _, __: graph.save(f"{RESULTS_DIR}/graph.pth"), feature="epoch", freq=1)
	energy_loss.logger_hooks(logger)
	best_ood_val_loss = float('inf')

	# TRAINING
	for epochs in range(0, max_epochs):

		logger.update("epoch", epochs)
		energy_loss.plot_paths(graph, logger, realities, prefix=f"epoch_{epochs}")
		if visualize: return

		graph.eval()
		for _ in range(0, val_step):
			with torch.no_grad():
				val_loss = energy_loss(graph, realities=[val])
				val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
			val.step()
			logger.update("loss", val_loss)

		energy_loss.select_losses(val)
		if epochs != 0: 
			energy_loss.logger_update(logger)
		else:
			energy_loss.metrics = {}
		logger.step()

		logger.text(f"Chosen losses: {energy_loss.chosen_losses}")
		logger.text(f"Percep winrate: {energy_loss.percep_winrate}")
		graph.train()
		for _ in range(0, train_step):
			train_loss2 = energy_loss(graph, realities=[train])
			train_loss = sum(train_loss2.values())
 
			graph.step(train_loss)
			train.step()

			logger.update("loss", train_loss)

		# if logger.data["val_mse : y^ -> n(~x)"][-1] < best_ood_val_loss:
		# 	best_ood_val_loss = logger.data["val_mse : y^ -> n(~x)"][-1]
		# 	energy_loss.plot_paths(graph, logger, realities, prefix="best")

if __name__ == "__main__":
	Fire(main)
