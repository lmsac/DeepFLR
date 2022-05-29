import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
import json

from fastNLP.core.metrics import MetricBase,seq_len_to_mask
from fastNLP.core.losses import LossBase
from preprocess import NPPeptidePipe,PPeptidePipe
from torch.nn import CosineSimilarity
from model import *
from model import _2deepdiaModelms2
from model import _2deepchargeModelirt
from model import _2deepchargeModelirt_cls
from model import _2deepchargeModelirt_ll
from model import _2deepchargeModelirt_zero_all
from model import _2deepchargeModelirt_ll_sigmoid
####################Const
from utils_irt import *
####################
set_seed(seed)


################data
from datapath import *

###################################################################################################PATH
fpath=os.path.join(path0,allirtphosTstd2)
###################################################################################################

databundle=PPeptidePipe(vocab=vocab).process_from_file(paths=fpath)
totaldata=databundle.get_dataset("train")
traindata,devdata=totaldata.split(0.10)
###########model
deepirt=_2deepchargeModelirt_zero_all(maxlength,acid_size,embed_size,nhead,num_layers,dropout=dropout,num_col=num_col)
###############################read pretrained model
bestmodelpath="AllphosTirt/best__2deepchargeModelirt_zero_all_meanl1loss_2021-03-20-14-32-17-116522"
bestmodel=torch.load(bestmodelpath).state_dict()
deepirt.load_state_dict(bestmodel)
###########Tester



from fastNLP import Const

pccmetrics=PearsonCCMetricfortest(target="irt")
from fastNLP import MSELoss,L1Loss
loss=MSELoss(pred=Const.OUTPUT,target="irt")
loss1=L1Loss(pred=Const.OUTPUT,target="irt")
############trainer
from fastNLP import Tester


pptester=Tester(model=deepirt,device=device,data=devdata,
                  loss=loss1,metrics=pccmetrics,
                   batch_size=BATCH_SIZE)
pptester.test()