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
import fastNLP.core.dataset
from transformers import BertConfig,RobertaConfig
from model import *
from model import _2deepdiaModelms2
from model import _2deepchargeModelms2_all
from Bertmodel import _2deepchargeModelms2_bert,_2deepchargeModelms2_roberta,_2deepchargeModelms2_bert_ss,_2deepchargeModelms2_roberta_ss,_2deepchargeModelms2_bert_ss_contrast
####################Const
from utils import *
####################预处理数据集
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        "--inputfile",
        default=None,
        type=str,
        required=True,
        help="inputfile used for finetune or retrain DeepFLR ",
    )
parser.add_argument(
        "--modelpath",
        default="phosT/best__2deepchargeModelms2_bert_mediancos_2021-09-20-01-17-50-729399",
        type=str,
        required=False,
        help="model path",
    )
parser.add_argument(
        "--type",
        default="finetune",
        type=str,
        required=False,
        help="finetune/train",
        choices=["finetune","train"]
    )

args = parser.parse_args()

inputfile=args.inputfile
bestmodelpath=args.modelpath
type=args.type

trainpathcsv=inputfile
traindatajson=trainpathcsv[:-4]+".json"

os.system("python matrixwithdict.py \
--do_ms2 \
--DDAfile {} \
--outputfile {}".format(trainpathcsv,traindatajson))

##############
set_seed(seed)
weight_decay=1e-2
batch_size=16
################data
from datapath import *

###########traindata
filename=traindatajson
databundle=PPeptidePipe(vocab=vocab).process_from_file(paths=filename)
totaldata=databundle.get_dataset("train")
vocab=databundle.get_vocab("peptide_tokens")


traindata,devdata=totaldata.split(0.1)
def savingFastnlpdataset_DataFrame(dataset):
    dataset_field=dataset.field_arrays.keys()
    frame=pd.DataFrame(columns=dataset_field)
    for i in range(len(dataset)):
        c_list=[]
        for name in dataset_field:
            target=dataset.field_arrays[name][i]
            if name=="target":
                c_list.append(target.cpu().numpy().tolist())
            else:
                c_list.append(target)

            
        frame.loc[i]=c_list

    
    return frame

# devframe=savingFastnlpdataset_DataFrame(devdata)
# devframe.to_json("AllphosT/Tprocessed2_testdata.json")

# trainframe=savingFastnlpdataset_DataFrame(traindata)
# trainframe.to_json("AllphosT/Tprocessed2_traindata.json")


###########model
if type=="train":
    pretrainmodel="bert-base-uncased"
    deepms2=_2deepchargeModelms2_bert.from_pretrained(pretrainmodel,cache_dir="./")

# deepms2=_2deepchargeModelms2_all(maxlength,acid_size,embed_size,nhead,num_layers,dropout=dropout,num_col=num_col)
# ##############################read pretrained model
if type=="finetune":
    config=BertConfig.from_pretrained("bert-base-uncased")
    bestmodelpath=bestmodelpath
    deepms2=_2deepchargeModelms2_bert(config)
    bestmodel=torch.load(bestmodelpath).state_dict()
    deepms2.load_state_dict(bestmodel)
###########Trainer

from fastNLP import Const
metrics=CossimilarityMetricfortest(savename=None,pred=Const.OUTPUT,target=Const.TARGET,seq_len='seq_len',
                                        num_col=num_col,sequence='sequence',charge="charge",decoration="decoration")
from fastNLP import MSELoss
loss=MSELoss(pred=Const.OUTPUT,target=Const.TARGET)
import torch.optim as optim
optimizer=optim.AdamW(deepms2.parameters(),lr=lr)
from fastNLP import WarmupCallback,SaveModelCallback
save_path=filename[:-5]+"/checkpoints"
############trainer
from fastNLP import Trainer


if vocab_save:
    vocab.save(os.path.join(save_path,"vocab"))

pptrainer=Trainer(model=deepms2,    train_data=traindata,
                    device=device,  dev_data=devdata,
                save_path=save_path,
                  loss=loss,metrics=metrics,
                   optimizer=optimizer,n_epochs=finetune_epoch,batch_size=batch_size,update_every=int(BATCH_SIZE/batch_size),dev_batch_size=batch_size)
pptrainer.train()
