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
import fastNLP
from transformers import BertConfig,RobertaConfig
from model import *
from model import _2deepdiaModelms2
from model import _2deepchargeModelms2_all
from Bertmodel import _2deepchargeModelms2_bert,_2deepchargeModelms2_roberta,_2deepchargeModelms2_bert_ss,_2deepchargeModelms2_roberta_ss,_2deepchargeModelms2_bert_ss_contrast
####################Const
from utils import *
####################预处理数据集
# trainpathcsv="/remote-home/yxwang/Finalcut/fintune/MSMS——fintune/mouse_fintune_2/mouse_fintune_training_dataset.csv"
# devpathcsv="/remote-home/yxwang/Finalcut/fintune/MSMS——fintune/mouse_fintune_2/mouse_fintune_test_dataset.csv"
# traindatajson=trainpathcsv[:-4]+".json"
# devdatajson=devpathcsv[:-4]+".json"
# os.system("python matrixwithdict.py \
# --do_ms2 \
# --DDAfile {} \
# --outputfile {}".format(trainpathcsv,traindatajson))

# os.system("python matrixwithdict.py \
# --do_ms2 \
# --DDAfile {} \
# --outputfile {}".format(devpathcsv,devdatajson))
##############
set_seed(seed)
weight_decay=1e-2
batch_size=16
################data
from datapath import *

###########traindata
filename="/remote-home/yxwang/Finalcut/fintune/DeepFLR/fintune/mouse_fintune/Model_mouse_fintune_training_dataset.json"
databundle=PPeptidePipe(vocab=vocab).process_from_file(paths=filename)
totaldata=databundle.get_dataset("train")
vocab=databundle.get_vocab("peptide_tokens")


traindata,devdata=totaldata.split(0.1)

###########testdata
testfile="/remote-home/yxwang/Finalcut/fintune/DeepFLR/fintune/mouse_fintune/Model_mouse_fintune_test_dataset.json"
testdatabundle=PPeptidePipe(vocab=vocab).process_from_file(paths=testfile)
testdata=testdatabundle.get_dataset("train")
###########model
pretrainmodel="bert-base-uncased"

# deepms2=_2deepchargeModelms2_bert.from_pretrained(pretrainmodel)

# deepms2=_2deepchargeModelms2_all(maxlength,acid_size,embed_size,nhead,num_layers,dropout=dropout,num_col=num_col)
# ##############################read pretrained model
config=BertConfig.from_pretrained("bert-base-uncased")
# bestmodelpath="/remote-home/yxwang/Finalcut/checkpoints/bert-base-uncased/pretrained_trainall_ss/furthermann/best__2deepchargeModelms2_bert_mediancos_2021-09-30-13-31-45-442035"###mannfintuned
bestmodelpath="/remote-home/yxwang/Finalcut/checkpoints/pretrainedbertbaseconfig_trainall/best__2deepchargeModelms2_bert_mediancos_2021-09-20-01-17-50-729399"##bertpretrained
deepms2=_2deepchargeModelms2_bert(config)
bestmodel=torch.load(bestmodelpath).state_dict()
deepms2.load_state_dict(bestmodel)
###########Trainer


from fastNLP import Const
metrics=CossimilarityMetricfortest(savename=testfile,pred=Const.OUTPUT,target=Const.TARGET,seq_len='seq_len',
                                        num_col=num_col,sequence='sequence',charge="charge",decoration="decoration")
from fastNLP import MSELoss
loss=MSELoss(pred=Const.OUTPUT,target=Const.TARGET)
import torch.optim as optim
optimizer=optim.AdamW(deepms2.parameters(),lr=lr)
from fastNLP import WarmupCallback,SaveModelCallback
save_path=filename[:-5]+"/checkpoints"
# save_path=os.path.join(path0,"checkpoints/"+pretrainmodel+"/pretrained_trainall_ss/combinemann_all")
callback=[WarmupCallback(warmupsteps)]
callback.append(WandbCallback(project="Finalcut",name=save_path,config={"lr":lr,"seed":seed,
"Batch_size":BATCH_SIZE,"warmupsteps":warmupsteps,"temperature":None,"weight_decay":None}))
callback.append(SaveModelCallback(save_path,top=3))
############trainer
from fastNLP import Trainer


if vocab_save:
    vocab.save(os.path.join(save_path,"vocab"))

pptrainer=Trainer(model=deepms2,    train_data=totaldata,
                    device=device,  dev_data=testdata,
                save_path=save_path,
                  loss=loss,metrics=metrics,callbacks=callback,
                   optimizer=optimizer,n_epochs=finetune_epoch,batch_size=batch_size,update_every=int(BATCH_SIZE/batch_size),dev_batch_size=batch_size)
pptrainer.train()
