import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
import json
from preprocess import *
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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--inputfile",
    default="zhaodan_only_phos_DDA_exp(1)processed.json",
    # default="./output_train/train_pred_gold_ner.json",
    type=str,
    required=True,
    help="DDA processed file,if using no_target ,use zz csv file",
)
parser.add_argument(
    "--outputfile",
    default="repsequenceirt.json",
    # default="./output_train/train_pred_gold_ner.json",
    type=str,
    # required=True,
    help="output filename irt list",
)
args = parser.parse_args()

processeddata=args.inputfile
save_path=args.outputfile
set_seed(seed)
class PearsonCCMetricforreplace(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self, pred=None, target=None,seq_len='seq_len',
                                   sequence='sequence',charge="charge",decoration="decoration"):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len,sequence=sequence,charge=charge,decoration=decoration)
        self.prelist=[]
        self.targetlist=[]
        self.nj=0
        self.repsequence=pd.DataFrame(columns=['repsequence','decoration','irt'])

    def evaluate(self, pred, target, seq_len=None,
                                   sequence=None,charge=None,decoration=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        N=pred.size(0)
        self.prelist+=pred.cpu().numpy().tolist()
        self.targetlist+=target.cpu().numpy().tolist()
        for i in range(N):
            il = seq_len[i]
            isequence=sequence[i][:il]
            icharge=charge[i]
            idecoration=decoration[i][:il]
            iirt=pred[i]

            self.repsequence.loc[self.nj] = [isequence.cpu().numpy().tolist(),
                                         idecoration.cpu().numpy().tolist(),

                                         iirt.cpu().numpy().tolist()]
            self.nj+=1


    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        cos=CosineSimilarity(dim=0)
        MAE=np.mean(np.abs(np.array(self.prelist)-np.array(self.targetlist)))
        Pprelist=self.prelist-np.mean(self.prelist)
        Ptargetlist=self.targetlist-np.mean(self.targetlist)
        PCC=cos(torch.Tensor(Pprelist),torch.Tensor(Ptargetlist))
        PCC=PCC.item()
        outdata=pd.DataFrame(columns=["pred_irt","exp_irt"])
        outdata["pred_irt"]=self.prelist
        outdata["exp_irt"]=self.targetlist
        outdata.to_csv("irt_pred_experiment.csv",index=False)
        self.repsequence.to_json(save_path)
        evaluate_result = {'PCC':round(PCC,6),"meanl1loss":round(MAE,6)

                           }
        if reset:
            self.prelist = []
            self.targetlist=[]
        return evaluate_result

################data
from datapath import *

###################################################################################################PATH
fpath=processeddata
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

pccmetrics=PearsonCCMetricforreplace(target="irt")
from fastNLP import MSELoss,L1Loss
loss=MSELoss(pred=Const.OUTPUT,target="irt")
loss1=L1Loss(pred=Const.OUTPUT,target="irt")
############trainer
from fastNLP import Tester


pptester=Tester(model=deepirt,device=device,data=totaldata,
                  loss=loss1,metrics=pccmetrics,
                   batch_size=BATCH_SIZE)
pptester.test()