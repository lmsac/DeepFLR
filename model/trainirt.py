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
from model import _2deepchargeModelirt_zero
from model import _2deepchargeModelirt_zero_all
from model import _2deepchargeModelirt_ll_sigmoid
from Bertmodel import _2deepchargeModelms2_bert_irt,_2deepchargeModelms2_roberta,_2deepchargeModelms2_bert_ss,_2deepchargeModelms2_roberta_ss,_2deepchargeModelms2_bert_ss_contrast

####################Const
from utils_irt import *
from utils import WandbCallback
####################
set_seed(seed)
warmupsteps=3000
batch_size=128
################data
from datapath import *

###################################################################################################PATH
fpath=os.path.join(path0,allirtphosTstd2)
###################################################################################################

databundle=PPeptidePipe(vocab=vocab).process_from_file(paths=fpath)
totaldata=databundle.get_dataset("train")
traindata,devdata=totaldata.split(0.10)
###########model
print("数据总量:",len(totaldata))
# deepirt=_2deepchargeModelirt_zero_all(maxlength,acid_size,embed_size,nhead,num_layers,dropout=dropout,num_col=num_col)
###############################read pretrained model
# bestmodelpath="phosT/best__2deepchargeModelms2_mediancos_2021-01-28-14-45-59-411833"
# bestmodel=torch.load(bestmodelpath).state_dict()
# deepirt.load_state_dict(bestmodel)

pretrainmodel="bert-base-uncased"

deepirt=_2deepchargeModelms2_bert_irt.from_pretrained(pretrainmodel)
###########Trainer



from fastNLP import Const
metrics=CossimilarityMetric(pred=Const.OUTPUT,target=Const.TARGET,seq_len='seq_len',num_col=num_col)
pccmetrics=PearsonCCMetric(pred="predirt",target="irt")
from fastNLP import MSELoss,L1Loss
loss=MSELoss(pred="predirt",target="irt")
loss1=L1Loss(pred="predirt",target="irt")
import torch.optim as optim
optimizer=optim.AdamW(deepirt.parameters(),lr=lr)
from fastNLP import WarmupCallback,SaveModelCallback
save_path=os.path.join(path0,"irt/checkpoints/"+pretrainmodel+"/pretrained_trainall_ss")
callback=[WarmupCallback(warmupsteps)]
callback.append(WandbCallback(project="Finalcutirt",name=save_path,config={"lr":lr,"seed":seed,
"Batch_size":BATCH_SIZE,"warmupsteps":warmupsteps,"temperature":None,"weight_decay":None}))
callback.append(SaveModelCallback(save_path,top=3))

############trainer
from fastNLP import Trainer


pptrainer=Trainer(model=deepirt,train_data=traindata,device=device,dev_data=devdata,save_path=save_path,
                  loss=loss1,metrics=pccmetrics,callbacks=callback,
                   optimizer=optimizer,n_epochs=N_epochs,batch_size=batch_size,update_every=int(BATCH_SIZE/batch_size),dev_batch_size=batch_size)
pptrainer.train()