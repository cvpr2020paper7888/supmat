import os, sys, math, random, itertools
import numpy as np
import scipy
from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'

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
	cont=f"{BASE_DIR}/shared/results_LBP_multipercep_lat_winrate_8/graph.pth", 
	cont_gan=None, pre_gan=None, max_epochs=800, use_baseline=False, **kwargs,
):
	
	# CONFIG
	batch_size = batch_size or (4 if fast else 64)
	energy_loss = get_energy_loss(config=loss_config, mode=mode, **kwargs)

	# DATA LOADING
	train_dataset, val_dataset, train_step, val_step = load_train_val(
		energy_loss.get_tasks("train"), energy_loss.get_tasks("val"),
		batch_size=batch_size, fast=fast,
	)
	train_subset_dataset, _, _, _ = load_train_val(
		energy_loss.get_tasks("train_subset"),
		batch_size=batch_size, fast=fast, subset_size=subset_size
	)
	if not fast:
		train_step, val_step = train_step // (16*4), val_step // (16)
	test_set = load_test(energy_loss.get_tasks("test"))
	ood_set = load_ood(energy_loss.get_tasks("ood"))
	
	train = RealityTask("train", train_dataset, batch_size=batch_size, shuffle=True)
	train_subset = RealityTask("train_subset", train_subset_dataset, batch_size=batch_size, shuffle=True)
	val = RealityTask("val", val_dataset, batch_size=batch_size, shuffle=True)
	test = RealityTask.from_static("test", test_set, energy_loss.get_tasks("test"))
	# ood = RealityTask.from_static("ood", ood_set, [energy_loss.get_tasks("ood")])

	# GRAPH
	realities = [train, val, test,] + [train_subset] #[ood]
	graph = TaskGraph(tasks=energy_loss.tasks + realities, finetuned=finetuned, freeze_list=energy_loss.freeze_list)
	graph.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
	graph.load_weights(cont)

	# LOGGING
	logger = VisdomLogger("train", env=JOB)
	logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
	# logger.add_hook(lambda _, __: graph.save(f"{RESULTS_DIR}/graph.pth"), feature="epoch", freq=1)
	energy_loss.logger_hooks(logger)

	best_ood_val_loss = float('inf')
	energy_losses = []
	mse_losses = []
	pearsonr_vals = []
	percep_losses = defaultdict(list)
	pearson_percep = defaultdict(list)
	# # TRAINING
	# for epochs in range(0, max_epochs):

	# 	logger.update("epoch", epochs)
	# 	if epochs == 0:
	# 		energy_loss.plot_paths(graph, logger, realities, prefix="start" if epochs == 0 else "")
	# 	# if visualize: return

	# 	graph.eval()
	# 	for _ in range(0, val_step):
	# 		with torch.no_grad():
	# 			losses = energy_loss(graph, realities=[val])
	# 			all_perceps = [losses[loss_name] for loss_name in losses if 'percep' in loss_name ]
	# 			energy_avg = sum(all_perceps) / len(all_perceps)
	# 			for loss_name in losses:
	# 				if 'percep' not in loss_name: continue
	# 				percep_losses[loss_name] += [losses[loss_name].data.cpu().numpy()]
	# 			mse = losses['mse']
	# 			energy_losses.append(energy_avg.data.cpu().numpy())
	# 			mse_losses.append(mse.data.cpu().numpy())
				
	# 		val.step()
	# 	mse_arr = np.array(mse_losses)
	# 	energy_arr = np.array(energy_losses)	
	# 	# logger.scatter(mse_arr - mse_arr.mean() / np.std(mse_arr), \
	# 	# 	energy_arr - energy_arr.mean() / np.std(energy_arr), \
	# 	# 	'unit_normal_all', opts={'xlabel':'mse','ylabel':'energy'})
	# 	logger.scatter(mse_arr, energy_arr, \
	# 		'mse_energy_all', opts={'xlabel':'mse','ylabel':'energy'})
	# 	pearsonr, p = scipy.stats.pearsonr(mse_arr, energy_arr)
	# 	logger.text(f'pearsonr = {pearsonr}, p = {p}')
	# 	pearsonr_vals.append(pearsonr)
	# 	logger.plot(pearsonr_vals, 'pearsonr_all')
	# 	for percep_name in percep_losses:
	# 		percep_loss_arr = np.array(percep_losses[percep_name])
	# 		logger.scatter(mse_arr, percep_loss_arr, f'mse_energy_{percep_name}', \
	# 			opts={'xlabel':'mse','ylabel':'energy'})
	# 		pearsonr, p = scipy.stats.pearsonr(mse_arr, percep_loss_arr)
	# 		pearson_percep[percep_name] += [pearsonr]
	# 		logger.plot(pearson_percep[percep_name], f'pearson_{percep_name}')


		# energy_loss.logger_update(logger)
		# if logger.data['val_mse : n(~x) -> y^'][-1] < best_ood_val_loss:
		# 	best_ood_val_loss = logger.data['val_mse : n(~x) -> y^'][-1]
		# 	energy_loss.plot_paths(graph, logger, realities, prefix="best")

	energy_mean_by_blur = []
	energy_std_by_blur = []
	mse_mean_by_blur = []
	mse_std_by_blur = []
	for blur_size in np.arange(0, 10, 0.5):
		tasks.rgb.blur_radius = blur_size if blur_size > 0 else None
		train_subset.step()
		# energy_loss.plot_paths(graph, logger, realities, prefix="start" if epochs == 0 else "")

		energy_losses = []
		mse_losses = []
		for epochs in range(subset_size // batch_size):
			with torch.no_grad():
				flosses = energy_loss(graph, realities=[train_subset], reduce=False)
				losses = energy_loss(graph, realities=[train_subset], reduce=False)
				all_perceps = np.stack([losses[loss_name].data.cpu().numpy() for loss_name in losses if 'percep' in loss_name])
				energy_losses += list(all_perceps.mean(0))
				mse_losses += list(losses['mse'].data.cpu().numpy())
			train_subset.step()
		mse_losses = np.array(mse_losses)
		energy_losses = np.array(energy_losses)
		logger.text(f'blur_radius = {blur_size}, mse = {mse_losses.mean()}, energy = {energy_losses.mean()}')
		logger.scatter(mse_losses, energy_losses, \
			f'mse_energy, blur = {blur_size}', opts={'xlabel':'mse','ylabel':'energy'})

		energy_mean_by_blur += [energy_losses.mean()]
		energy_std_by_blur += [np.std(energy_losses)]
		mse_mean_by_blur += [mse_losses.mean()]
		mse_std_by_blur += [np.std(mse_losses)]

	logger.plot(energy_mean_by_blur, f'energy_mean_by_blur')
	logger.plot(energy_std_by_blur, f'energy_std_by_blur')
	logger.plot(mse_mean_by_blur, f'mse_mean_by_blur')
	logger.plot(mse_std_by_blur, f'mse_std_by_blur')

if __name__ == "__main__":
	Fire(main)