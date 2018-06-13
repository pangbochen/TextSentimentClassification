# -*- coding: UTF-8 -*-

# -*- coding: UTF-8 -*-

import torch
import torch.nn.functional as F
import torchtext
import time
import opts
from utils import load_data, clip_gradient, evaluate, validate
import models

# get option

opt = opts.parse_opt()

opt.label_size = 100
opt.vocab_size = 100
opt.embedding_dim = 100 # 300 as default
opt.embeddings = None

# select model
opt.model = 'lstm'

model = models.init(opt)

model.cuda()
