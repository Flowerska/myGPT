import os
import time
import math
import pickle 
from contextlib import nullcontext

import numpy as np
import torch 
from  torch.nn.parallel import DistributedDataParallel as DDP
from  torch. distributed import init_process_group, destroy_process_group

# default config values designed to train a gpt2
out_dir = 'out'
eval_interval = 2000
log_interval = 100
eval_iters = 200
eval_only = False # if True, script exits right after the first evaluation
always_save_checkpoint = True # if True, always save a checkpoint after each evaluation
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt
