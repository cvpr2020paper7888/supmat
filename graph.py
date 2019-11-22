import os, sys, math, random, itertools, heapq
from collections import namedtuple, defaultdict
from functools import partial, reduce
import numpy as np
import IPython

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from models import TrainableModel, WrapperModel
from datasets import TaskDataset
from task_configs import get_task, task_map, tasks, get_model, RealityTask
from task_configs import tasks as TASKS
from transfers import Transfer, RealityTransfer, get_transfer_name, pretrained_transfers
import transforms

from modules.gan_dis import GanDisNet


import os, sys, math, random, itertools, heapq
from collections import namedtuple, defaultdict
from functools import partial, reduce
import numpy as np
import IPython

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from models import TrainableModel, WrapperModel
from datasets import TaskDataset
from task_configs import get_task, task_map, tasks, get_model, RealityTask
from transfers import Transfer, RealityTransfer, get_transfer_name
import transforms

from modules.gan_dis import GanDisNet

class TaskGraph(TrainableModel):
    """Basic graph that encapsulates set of edge constraints. Can be saved and loaded
    from directories."""

    def __init__(
        self, tasks=tasks, edges=None, edges_exclude=None, 
        pretrained=True, finetuned=False,
        reality=[], task_filter=[tasks.segment_semantic, tasks.class_scene],
        freeze_list=[], lazy=False,
    ):

        super().__init__()
        self.tasks = list(set(tasks) - set(task_filter))
        self.tasks += [task.base for task in self.tasks if hasattr(task, "base")]
        self.edge_list, self.edge_list_exclude = edges, edges_exclude
        self.pretrained, self.finetuned = pretrained, finetuned
        self.edges, self.adj, self.in_adj = [], defaultdict(list), defaultdict(list)
        self.edge_map, self.reality = {}, reality
        print('graph tasks', self.tasks)
        self.params = {}

        # construct transfer graph
        for src_task, dest_task in itertools.product(self.tasks, self.tasks):
            key = (src_task, dest_task)
            if edges is not None and key not in edges: continue
            if edges_exclude is not None and key in edges_exclude: continue
            if src_task == dest_task: continue
            if isinstance(dest_task, RealityTask): continue
            transfer = None
            if isinstance(src_task, RealityTask):
                if dest_task not in src_task.tasks: continue
                transfer = RealityTransfer(src_task, dest_task)
            else:
                transfer = Transfer(src_task, dest_task, 
                    pretrained=pretrained, finetuned=finetuned
                )
                transfer.name = get_transfer_name(transfer)
            if transfer.model_type is None: 
                continue
            self.edges += [transfer]
            self.adj[src_task.name] += [transfer]
            self.in_adj[dest_task.name] += [transfer]
            self.edge_map[str((src_task.name, dest_task.name))] = transfer
            if isinstance(transfer, nn.Module):
                if str((src_task.name, dest_task.name)) not in freeze_list:
                    self.params[str((src_task.name, dest_task.name))] = transfer
                else:
                    print("freezing " + str((src_task.name, dest_task.name)))
                try:
                    if not lazy: transfer.load_model()
                except:
                    print('Cound not load model:', str((src_task.name, dest_task.name)))
                    IPython.embed()

        self.params = nn.ModuleDict(self.params)
    
    def edge(self, src_task, dest_task):
        key1 = str((src_task.name, dest_task.name))
        key2 = str((src_task.kind, dest_task.kind))
        if key1 in self.edge_map: return self.edge_map[key1]
        return self.edge_map[key2]

    def sample_path(self, path, reality=None, use_cache=False, cache={}):
        print('cache keys', list(cache.keys()), ' | path:', path)
        path = [reality or self.reality[0]] + path
        x = None
        for i in range(1, len(path)):
            try:
#                 print(path[i-1], path[i])
                # if x is not None: print (x.shape)
                # print (self.edge(path[i-1], path[i]))
                x = cache.get(tuple(path[0:(i+1)]), 
                    self.edge(path[i-1], path[i])(x)
                )
            except KeyError:
                return None
            except Exception as e:
                IPython.embed()
                
            if use_cache: cache[tuple(path[0:(i+1)])] = x
        return x

    def save(self, weights_file=None, weights_dir=None):

        ### TODO: save optimizers here too
        if weights_file:
            torch.save({
                key: model.state_dict() for key, model in self.edge_map.items() \
                if not isinstance(model, RealityTransfer)
            }, weights_file)

        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)
            for key, model in self.edge_map.items():
                if isinstance(model, RealityTransfer): continue
                if not isinstance(model.model, TrainableModel): continue
                model.model.save(f"{weights_dir}/{model.name}.pth")
            torch.save(self.optimizer, f"{weights_dir}/optimizer.pth")


    def load_weights(self, weights_file=None):
        for key, state_dict in torch.load(weights_file).items():
            if key in self.edge_map:
                self.edge_map[key].load_model()
                self.edge_map[key].load_state_dict(state_dict)


class Discriminator(object):

    def __init__(self, loss_config, frac=None, size=224, sigma=12, use_patches=False):
        super(Discriminator, self).__init__()
        self.size = size
        self.frac = frac
        self.sigma = sigma
        self.use_patches = use_patches
        self.discriminator = GanDisNet(size=size)
        self.discriminator.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)

    def train(self):
        self.discriminator.train()

    def eval(self):
        self.discriminator.eval()

    def step(self, loss):
        self.discriminator.step(-sum(loss[dis_key] for dis_key in loss if 'disgan' in dis_key))

    def sample_patches(self, X, mean=1, sigma=None):
        sigma = sigma or self.sigma
        N, C, H, W = X.shape
        def crop(x, size):
            a, b = random.randint(0, H-size), random.randint(0, W-size)
            x = x[:,a:a+size,b:b+size]
            x = nn.functional.interpolate(x.unsqueeze(0), size=(mean, mean))
            return x[0]

        def sample(x, size, n):
            return torch.cat([crop(x, size) for _ in range(n)], dim=1).view(-1, 3, 3)

        sizes = np.clip(np.random.randn(N) * sigma, sigma*2, sigma*-2).astype(int) + mean
        sizes = np.clip(sizes, a_min=1, a_max=(min(H, W)))
        patches = torch.stack([ crop(x, size) for x, size in zip(X, sizes) ])
        # patches = torch.stack([ sample(x, size, 9) for x, size in zip(X, sizes)])
        return patches

    def __call__(self, x):
        size = int(self.frac * x.shape[2]) if self.frac is not None else self.size
        x = self.sample_patches(x, mean=size) if self.use_patches else x
        return self.discriminator(x)

    def save(self, weights_file=None):
        weights = self.discriminator.state_dict()
        optimizer = self.discriminator.optimizer.state_dict()
        torch.save({'weights':weights, 'optimizer':optimizer}, weights_file)


    def load_weights(self, weights_file=None):
        file_ = torch.load(weights_file)
        self.discriminator.load_state_dict(flie_['weights'])
        self.discriminator.optimizer.load_state_dict(flie_['optimizer'])
        
        
class GeoNetTaskGraph(TaskGraph):
    def __init__(
        self, tasks=tasks, realities=None, edges=None, edges_exclude=None, 
        pretrained=True, finetuned=False,
        reality=[], task_filter=[tasks.segment_semantic, tasks.class_scene],
        freeze_list=[], lazy=False,
    ):

        super().__init__(tasks=[])
        self.tasks = list(set(tasks) - set(task_filter))
        self.tasks += [task.base for task in self.tasks if hasattr(task, "base")]
        self.edge_list, self.edge_list_exclude = edges, edges_exclude
        self.pretrained, self.finetuned = pretrained, finetuned
        self.edges, self.adj, self.in_adj = [], defaultdict(list), defaultdict(list)
        self.edge_map, self.reality = {}, reality
        print('graph tasks!', self.tasks)
        self.params = nn.ModuleDict()
        self.realities = realities
        
        # RGB -> Normal
        transfer = Transfer(TASKS.rgb, TASKS.normal, pretrained=pretrained, finetuned=finetuned)
        transfer.name = get_transfer_name(transfer)
        self.params[str((TASKS.rgb.name, TASKS.normal.name))] = transfer
        self.edge_map[str((TASKS.rgb.name, TASKS.normal.name))] = transfer
        self.edges += [transfer]
        try:
            if not lazy: transfer.load_model()
        except:
            print('Cound not load model:', str((TASKS.rgb.name, TASKS.normal.name)))
            IPython.embed()

        # RGB -> Depth
        transfer = Transfer(TASKS.rgb, TASKS.depth_zbuffer, pretrained=pretrained, finetuned=finetuned)
        transfer.name = get_transfer_name(transfer)
        self.params[str((TASKS.rgb.name, TASKS.depth_zbuffer.name))] = transfer
        self.edge_map[str((TASKS.rgb.name, TASKS.depth_zbuffer.name))] = transfer
        try:
            if not lazy: transfer.load_model()
        except:
            print('Cound not load model:', str((TASKS.rgb.name, TASKS.depth_zbuffer.name)))
            IPython.embed()

        # Depth -> Normals
        src_task = (TASKS.depth_zbuffer.name, TASKS.FoV.name, TASKS.normal.name)
        target_task = TASKS.normal
        transfer_name = str((src_task, target_task.name))
        model_type, path = pretrained_transfers[(src_task, target_task.name)]
        transfer = Transfer(src_task, target_task, pretrained=pretrained,
                            model_type=model_type, path=path,  checkpoint=False,
                            finetuned=finetuned, name=f"{src_task}2{target_task.name}")
        transfer.name = transfer_name
        self.params[transfer_name] = transfer
        self.edge_map[transfer_name] = transfer
        try:
            if not lazy: transfer.load_model()
        except:
            print('Cound not load model:', transfer_name)
            IPython.embed()

        # Normal -> Depth
        src_task = (TASKS.depth_zbuffer.name, TASKS.FoV.name, TASKS.normal.name)
        target_task = TASKS.depth_zbuffer
        transfer_name = str((src_task, target_task.name))
        model_type, path = pretrained_transfers[(src_task, target_task.name)]
        transfer = Transfer(src_task, target_task, pretrained=pretrained,
                            model_type=model_type, path=path, checkpoint=False,
                            finetuned=finetuned, name=f"{src_task}2{target_task.name}")
        transfer.name = transfer_name
        self.params[transfer_name] = transfer
        self.edge_map[transfer_name] = transfer
        try:
            if not lazy: transfer.load_model()
        except:
            print('Cound not load model:', transfer_name)
            IPython.embed()

        for src_task, dest_task in itertools.product(self.realities, self.tasks):
            key = (src_task, dest_task)
            if edges is not None and key not in edges: continue
            if edges_exclude is not None and key in edges_exclude: continue
            if src_task == dest_task: continue
            if isinstance(dest_task, RealityTask): continue
            transfer = None
            if isinstance(src_task, RealityTask):
                if dest_task not in src_task.tasks: continue
                transfer = RealityTransfer(src_task, dest_task)
            else:
                transfer = Transfer(src_task, dest_task, 
                    pretrained=pretrained, finetuned=finetuned
                )
                transfer.name = get_transfer_name(transfer)
            if transfer.model_type is None: 
                continue
            self.edges += [transfer]
            self.adj[src_task.name] += [transfer]
            self.in_adj[dest_task.name] += [transfer]
            self.edge_map[str((src_task.name, dest_task.name))] = transfer
            if isinstance(transfer, nn.Module):
                if str((src_task.name, dest_task.name)) not in freeze_list:
                    self.params[str((src_task.name, dest_task.name))] = transfer
                else:
                    print("freezing " + str((src_task.name, dest_task.name)))
                try:
                    if not lazy: transfer.load_model()
                except:
                    print('Cound not load model:', str((src_task.name, dest_task.name)))
                    IPython.embed()

            
#     def sample_path(self, path, reality=None, use_cache=False, cache={}):
#         print(cache)
#         path = [reality or self.reality[0]] + path
#         x = None
#         for i in range(1, len(path)):
#             try:
#                 # if x is not None: print (x.shape)
#                 # print (self.edge(path[i-1], path[i]))
#                 x = cache.get(tuple(path[0:(i+1)]), 
#                     self.edge(path[i-1], path[i])(x)
#                 )
#             except KeyError:
#                 return None
#             except Exception as e:
#                 IPython.embed()
                
#             if use_cache: cache[tuple(path[0:(i+1)])] = x
#         return x


    def compute_all_paths(self, reality=None):
        rgb = self.edge(reality, TASKS.rgb)(None)

        can_compute_consistency = True
        try:
            fov = self.edge(reality, TASKS.FoV)(None)
        except KeyError:
            can_compute_consistency = False

        # RGB -> Depth
        initial_depth = self.edge(TASKS.rgb, TASKS.depth_zbuffer)(rgb)
        
        # RGB -> Normal
        initial_normal = self.edge(TASKS.rgb, TASKS.normal)(rgb)

        if can_compute_consistency:
            # (Depth, FoV, Normal) -> Normal
            src_task = (TASKS.depth_zbuffer.name, TASKS.FoV.name, TASKS.normal.name)
            target_task = TASKS.normal
            transfer_name = str((src_task, target_task.name))
            consistency_normal = self.edge_map[transfer_name]((initial_depth, fov, initial_normal))
            
            # (Depth, FoV, Normal) -> Depth
            src_task = (TASKS.depth_zbuffer.name, TASKS.FoV.name, TASKS.normal.name)
            target_task = TASKS.depth_zbuffer
            transfer_name = str((src_task, target_task.name))
            consistency_depth = self.edge_map[transfer_name]((initial_depth, fov, initial_normal))

        returned_paths = {
            "rgb": rgb,
            "N(rgb)": initial_normal,
            "D(rgb)": initial_depth,
        }
        
        if can_compute_consistency:
            returned_paths.update({
                "normal": self.edge(reality, TASKS.normal)(None),
                "depth": self.edge(reality, TASKS.depth_zbuffer)(None),
                "N(D(rgb))": consistency_normal,
                "D(N(rgb))": consistency_depth,
            })
#             print('reality:', reality)
#             print('\tnormal:', returned_paths['normal'].min(), returned_paths['normal'].max())
#             print('\tdepth:', returned_paths['depth'].min(), returned_paths['depth'].max())
#             print('rgb shape:', rgb.shape)
#             print('fov shape:', fov.shape)
#             print('depth shape:', initial_depth.min(), initial_depth.max())
#             print('normal shape:', initial_normal.min(),initial_normal.max(),)
#             print('consistency normal', consistency_normal.min(), consistency_normal.max())
#             print('consistency depth', consistency_depth.min(), consistency_depth.max())

        return returned_paths
