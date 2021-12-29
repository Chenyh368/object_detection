import os
from tqdm import tqdm
import torch
from torch.utils.data.distributed import DistributedSampler
import collections
# from utils.experiman import manager
from utils.misc import AverageMeter, MovingAverageMeter, ScalerMeter, PerClassMeter


"""
manager API: 
Setting: 
self.manager = manager

---------- For multi gpu ------------
self.is_master = manager.is_master()

---------- For log command line info ------------
if self.is_master:
    self.logger = manager.get_logger()
    self.logger.info("*")

---------- For getting ckpt dir ------------
model_path = os.path.join(
    self.manager.get_checkpoint_dir(), name)

---------- For log metric to tensorboard --------
self.manager.log_metric(name, value,
    global_iter, [epoch_id](Not used), split='train')
    
---------- For log metric to tensorboard --------
self.manager.log_metric(name, value,
    global_iter, [epoch_id](Not used), split='train')
    
"""
class Faster_RCNN_Trainer(object):
    def __init__(self, manager):
        self.manager = manager
