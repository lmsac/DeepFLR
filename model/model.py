import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import pandas as pd
import numpy as np
import json


from fastNLP.core.metrics import MetricBase,seq_len_to_mask
from fastNLP.core.losses import LossBase
from preprocess import NPPeptidePipe,PPeptidePipe
from torch.nn import CosineSimilarity

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def attentionmask(seq_len, max_len=None):
    r"""

    将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
    转变 1-d seq_len到2-d mask.

    .. code-block::
        #
        # >>> seq_len = torch.arange(2, 16)
        # >>> mask = seq_len_to_mask(seq_len)
        # >>> print(mask.size())
        # torch.Size([14, 15])
        # >>> seq_len = np.arange(2, 16)
        # >>> mask = seq_len_to_mask(seq_len)
        # >>> print(mask.shape)
        # (14, 15)
        # >>> seq_len = torch.arange(2, 16)
        # >>> mask = seq_len_to_mask(seq_len, max_len=100)
        # >>>print(mask.size())
        torch.Size([14, 100])

    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.ge(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask
class CossimilarityMetric(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self, pred=None, target=None, seq_len=None,num_col=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self.bestcos=0
        self.total = 0
        self.cos = 0
        self.listcos=[]
        self.num_col=num_col

    def evaluate(self, pred, target, seq_len=None):
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
        L_1=pred.size(1)
        if seq_len is not None and target.dim() > 1:
            max_len = target.size(1)
            masks = seq_len_to_mask(seq_len=(seq_len-1)*int(self.num_col))
        else:
            masks = None


        cos=CosineSimilarity(dim=1)
        if masks is not None:
            pred=pred.masked_fill(masks.eq(False), 0)
            self.cos += torch.sum(cos(pred,target)).item()
            self.total += pred.size(0)
            self.bestcos=max(self.bestcos,torch.max(cos(pred,target)).item())
            self.listcos += cos(pred, target).reshape(N, ).cpu().numpy().tolist()
        else:

            self.cos += torch.sum(cos(pred,target)).item()
            self.total += pred.size(0)
            self.bestcos = max(self.bestcos, torch.max(cos(pred,target)).item())
            self.listcos +=cos(pred, target).reshape(N,).cpu().numpy().tolist()
    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """

        evaluate_result = {'mediancos':round(np.median(self.listcos),6),

                           'coss': round(float(self.cos) / (self.total + 1e-12), 6),
                           'bestcoss':round(self.bestcos, 6),
                           }
        if reset:
            self.cos = 0
            self.total = 0
            self.bestcos=0
            self.listcos=[]
        return evaluate_result

class CossimilarityMetricfortest(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self,savename, pred=None, target=None, seq_len=None,num_col=None,sequence=None,charge=None,decoration=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len,sequence=sequence,charge=charge,decoration=decoration,_id="_id")
        self.bestcos=0
        self.total = 0
        self.cos = 0
        self.listcos=[]
        self.nan=0
        self.bestanswer=0
        self.nansequence=pd.DataFrame(columns=['nansequence','charge','decoration'])
        self.num_col=num_col
        self.savename=savename if savename else ""
        self.id_list=[]
    def evaluate(self, pred, target, seq_len=None,sequence=None,charge=None,decoration=None,_id=None):
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
        L_1=pred.size(1)
        if seq_len is not None and target.dim() > 1:
            max_len = target.size(1)
            masks = seq_len_to_mask(seq_len=(seq_len-1)*int(self.num_col))
        else:
            masks = None
        cos=CosineSimilarity(dim=1)
        # self.id_list+=_id.cpu().numpy().tolist()
        if masks is not None:
            s=torch.sum(cos(pred, target)).item()
            pred=pred.masked_fill(masks.eq(False), 0)
            if math.isnan(s):
                
                self.nansequence.loc[self.nan] = [sequence.cpu().numpy().tolist(),charge.cpu().numpy().tolist(),decoration.cpu().numpy().tolist()]
                self.nan+=1

            else:

                self.cos += torch.sum(cos(pred,target)).item()
                self.total += pred.size(0)
                self.bestcos=max(self.bestcos,torch.max(cos(pred,target)).item())
                self.listcos += cos(pred, target).reshape(N, ).cpu().numpy().tolist()
            
        else:
            s=torch.sum(cos(pred, target)).item()
            print(s)
            if math.isnan(s):
                print("getnan:{}".format(_id.cpu().numpy().tolist()[0]))
                self.nansequence.loc[self.nan] = [sequence.cpu().numpy().tolist(),charge.cpu().numpy().tolist(),decoration.cpu().numpy().tolist()]
                self.nan+=1

            else:

                self.cos += s
                self.total += pred.size(0)
                self.bestcos = max(self.bestcos, torch.max(cos(pred,target)).item())
                self.listcos +=cos(pred, target).reshape(N,).cpu().numpy().tolist()
    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        data=pd.Series(self.listcos)
        mediancos=np.median(self.listcos)
        if mediancos>self.bestanswer:
            data.to_csv(self.savename+"Cossimilaritylist.csv",index=False)
        
        if self.nan>0:
            self.nansequence.to_json(self.savename+"nansequence.json")
        evaluate_result = {'mediancos':round(mediancos,6),
                            'nan number':self.nan,
                           'coss': round(float(self.cos) / (self.total + 1e-12), 6),
                           'bestcoss':round(self.bestcos, 6),
                           }
        if reset:
            self.cos = 0
            self.total = 0
            self.bestcos=0
            self.listcos=[]
        return evaluate_result

class PearsonCCMetric(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self, pred=None, target=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target)
        self.prelist=[]
        self.targetlist=[]

    def evaluate(self, pred, target, seq_len=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        self.prelist+=pred.cpu().numpy().tolist()
        self.targetlist+=target.cpu().numpy().tolist()



    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        cos=CosineSimilarity(dim=0)
        MAE=-np.mean(np.abs(np.array(self.prelist)-np.array(self.targetlist)))
        Pprelist=self.prelist-np.mean(self.prelist)
        Ptargetlist=self.targetlist-np.mean(self.targetlist)
        PCC=cos(torch.Tensor(Pprelist),torch.Tensor(Ptargetlist))
        PCC=PCC.item()
        evaluate_result = {"meanl1loss":round(MAE,6),
                           'PCC':round(PCC,6),

                           }
        if reset:
            self.prelist = []
            self.targetlist=[]
        return evaluate_result
class PearsonCCMetricfortest(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self, pred=None, target=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target)
        self.prelist=[]
        self.targetlist=[]

    def evaluate(self, pred, target, seq_len=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        self.prelist+=pred.cpu().numpy().tolist()
        self.targetlist+=target.cpu().numpy().tolist()



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
        evaluate_result = {"meanl1loss":round(MAE,6),
                           'PCC':round(PCC,6),

                           }
        if reset:
            self.prelist = []
            self.targetlist=[]
        return evaluate_result
######################################################################################positionembedding
class PositionEmbedding(nn.Module):#input:sequence:N*L(N:batch)
    def __init__(self,emb_size,maxlength):
        super().__init__()
        pe=torch.arange(0,maxlength)
        pe.requires_grad=False
        pe = pe.unsqueeze(0)
        self.embedding=nn.Embedding(maxlength,emb_size)
        self.register_buffer('pe', pe)#1LE
    def forward(self,x,device):
        pe=self.embedding(self.pe[:,:x.size(1)])
        return pe
import math
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
##################################################################################### model
class deepdiaModelms2(nn.Module):#input:sequence:N*L(N:batch)
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.num_col=int(num_col)
        self.edim=embed_dim
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(2,embed_dim)#只有两种磷酸化情况
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,phos=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        device=peptide_tokens.device
        ll=torch.LongTensor(range(L))
        slengths=ll.expand(N,L)
        slengths=slengths.to(device)
        sequence=peptide_tokens

        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(slengths)#NLE

        if phos:
            assert sequence.size(0) == phos.size(0)
            phos_embed = self.phos_embedding(phos)
            ninput=pos_embed+a_embed+phos_embed#NLE
        else:
            ninput = pos_embed + a_embed   #NLE
        key_padding_mask=attentionmask(peptide_length-1)
        ninput=self.activation(self.conv(ninput.permute(0,2,1)))#NE(L-1)
        output =self.transformer(ninput.permute(2,0,1),src_key_padding_mask=key_padding_mask)#(L-1)NE
        outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1
        outputms=self.dropout(self.mslinear(output))#(L-1)*N*12
        outputms=self.activation(outputms)
        outputms=outputms.permute(1,0,2).reshape(N,-1)#N*((L-1)*12)
        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(torch.sum(outputms))
        return {'pred':outputms}
class _2deepdiaModelms2(nn.Module):#input:sequence:N*L(N:batch)
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=PositionEmbedding(embed_dim,maxlength)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)

        sequence=peptide_tokens
        device=peptide_tokens.device
        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(peptide_tokens,device)#NLE


        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)
        ninput=pos_embed+a_embed+phos_embed#NLE

        key_padding_mask=attentionmask(peptide_length)

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1
        output=self.activation(self.conv(output.permute(1,2,0)))# NE(L-1)
        output=output.permute(0,2,1)#N*(L-1)*E
        outputms=self.dropout(self.mslinear(output))#N*(L-1)*24
        outputms=self.activation(outputms)
        outputms=outputms.reshape(N,-1)#N*((L-1)*24)
        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(torch.sum(outputms))
        return {'pred':outputms}
####################################################################################################charge embedding
class _2deepchargeModelms2(nn.Module):#input:sequence:N*L(N:batch)
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(4,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        sequence=peptide_tokens
        device=peptide_length.device
        assert N==peptide_length.size(0)
        ll = torch.arange(0, L, device=device).unsqueeze(0)#1*L
        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        charge_embed=self.charge_embedding(charge.unsqueeze(1).expand(N,L))

        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)

        ninput=pos_embed+a_embed+phos_embed+charge_embed#NLE
        # ninput=self.dropout(ninput)
        key_padding_mask=attentionmask(peptide_length)

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1

        output=self.activation(self.conv(output.permute(1,2,0)))# NE(L-1)
        output=output.permute(0,2,1)#N*(L-1)*E
        outputms=self.dropout(self.mslinear(output))#N*(L-1)*24

        outputms=self.activation(outputms)

        outputms=outputms.reshape(N,-1)#N*((L-1)*24)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(outputms)
        # print(torch.sum(outputms))
        return {'pred':outputms,'sequence':sequence,'charge':charge,"decoration":decoration,"seq_len":peptide_length}
# irt******************************************************************************************************
class _2deepchargeModelirt_ll(nn.Module):#input:sequence:N*L(N:batch)###不使用charge
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear1=nn.Linear(embed_dim,256)
        self.rtlinear2=nn.Linear(256,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        #sequence前面加入[CLS]的embedding
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        device=peptide_tokens.device
        ll=torch.arange(0,L,device=device).unsqueeze(0)
        sequence=peptide_tokens
        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)
        ninput=pos_embed+a_embed+phos_embed#NLE
        # ninput=self.dropout(ninput)

        key_padding_mask=attentionmask(peptide_length)#

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        output=torch.max(output,dim=0).values#maxpooling #N*E
        output = self.activation(self.rtlinear1(output))
        outputrt=self.activation(self.rtlinear2(output).squeeze())#N*1
        # outputrt=outputrt
        # print(torch.sum(outputms))
        return {'pred':outputrt}
##########sigmoid
class _2deepchargeModelirt_ll_sigmoid(nn.Module):#input:sequence:N*L(N:batch)###不使用charge
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear1=nn.Linear(embed_dim,256)
        self.rtlinear2=nn.Linear(256,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        #sequence前面加入[CLS]的embedding
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        device=peptide_tokens.device
        ll=torch.arange(0,L,device=device).unsqueeze(0)
        sequence=peptide_tokens
        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)
        ninput=pos_embed+a_embed+phos_embed#NLE
        # ninput=self.dropout(ninput)

        key_padding_mask=attentionmask(peptide_length)#

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        output=torch.max(output,dim=0).values#maxpooling #N*E
        output = self.activation(self.rtlinear1(output))
        outputrt=torch.sigmoid(self.rtlinear2(output).squeeze())#N*1
        # outputrt=outputrt
        # print(torch.sum(outputms))
        return {'pred':outputrt}
####################clsembedding
class _2deepchargeModelirt_cls(nn.Module):#input:sequence:N*L(N:batch)
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.cls_embedding=nn.Embedding(2,embed_dim)##
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        #sequence前面加入[CLS]的embedding
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        device=peptide_tokens.device
        ll=torch.arange(0,L+1,device=device).unsqueeze(0)##L+1
        sequence=peptide_tokens
        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        cls_embed=self.cls_embedding(torch.ones(N,device=device,dtype=int)).unsqueeze(1)#N*1*E

        pos_embed=self.pos_embedding(ll)#1(L+1)E

        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)#NLE
        ninput=a_embed+phos_embed#NLE
        ninput=torch.cat([cls_embed,ninput],dim=1)+pos_embed#N(L+1)E
        # ninput=self.dropout(ninput)

        key_padding_mask=attentionmask(peptide_length+1)#

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L+1)NE
        output=output[0].squeeze()#maxpooling #N*E

        outputrt=self.rtlinear(output).squeeze()#N*1
        # outputrt=outputrt
        # print(torch.sum(outputms))
        return {'pred':outputrt}
#######################
class _2deepchargeModelirt(nn.Module):#input:sequence:N*L(N:batch)###不使用charge
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        #sequence前面加入[CLS]的embedding
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        device=peptide_tokens.device
        ll=torch.arange(0,L,device=device).unsqueeze(0)
        sequence=peptide_tokens
        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)
        ninput=pos_embed+a_embed+phos_embed#NLE
        # ninput=self.dropout(ninput)

        key_padding_mask=attentionmask(peptide_length)#

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        output=torch.max(output,dim=0).values#maxpooling #N*E

        outputrt=self.rtlinear(output).squeeze()#N*1
        # outputrt=outputrt
        # print(torch.sum(outputms))
        return {'pred':outputrt}
##################################################################取0位置linear
class _2deepchargeModelirt_zero(nn.Module):#input:sequence:N*L(N:batch)###不使用charge
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(4,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        #sequence前面加入[CLS]的embedding
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        device=peptide_tokens.device
        ll=torch.arange(0,L,device=device).unsqueeze(0)
        sequence=peptide_tokens
        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)
        ninput=pos_embed+a_embed+phos_embed#NLE
        # ninput=self.dropout(ninput)

        key_padding_mask=attentionmask(peptide_length)#

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        output=output[0].squeeze()#maxpooling #N*E

        outputrt=self.rtlinear(output).squeeze()#N*1
        # outputrt=outputrt
        # print(torch.sum(outputms))
        return {'pred':outputrt,'sequence':sequence,'charge':charge,"decoration":decoration,"seq_len":peptide_length}

######完整版模型#############################
class _2deepchargeModelirt_zero_all(nn.Module):#input:sequence:N*L(N:batch)###不使用charge
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        #sequence前面加入[CLS]的embedding
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        device=peptide_tokens.device
        ll=torch.arange(0,L,device=device).unsqueeze(0)
        sequence=peptide_tokens
        assert N==peptide_length.size(0)

        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)
        ninput=pos_embed+a_embed+phos_embed#NLE
        # ninput=self.dropout(ninput)

        key_padding_mask=attentionmask(peptide_length)#

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        output=output[0].squeeze()#maxpooling #N*E

        outputrt=self.rtlinear(output).squeeze()#N*1
        # outputrt=outputrt
        # print(torch.sum(outputms))
        return {'pred':outputrt,'sequence':sequence,'charge':charge,"decoration":decoration,"seq_len":peptide_length}


class _2deepchargeModelms2_all(nn.Module):#input:sequence:N*L(N:batch)
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration,pnumber):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        sequence=peptide_tokens
        device=peptide_length.device
        assert N==peptide_length.size(0)
        ll = torch.arange(0, L, device=device).unsqueeze(0)#1*L
        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        charge_embed=self.charge_embedding(charge.unsqueeze(1).expand(N,L))

        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)

        ninput=pos_embed+a_embed+phos_embed+charge_embed#NLE
        # ninput=self.dropout(ninput)
        key_padding_mask=attentionmask(peptide_length)

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1

        output=self.activation(self.conv(output.permute(1,2,0)))# NE(L-1)
        output=output.permute(0,2,1)#N*(L-1)*E
        outputms=self.dropout(self.mslinear(output))#N*(L-1)*24

        outputms=self.activation(outputms)

        outputms=outputms.reshape(N,-1)#N*((L-1)*num_col)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(outputms)
        # print(torch.sum(outputms))
        return {'pred':outputms,'sequence':sequence,'charge':charge,"decoration":decoration,"seq_len":peptide_length,
                'pnumber':pnumber}

########################contrast training for decoy
class _2deepchargeModelms2_all_contrast(nn.Module):#input:sequence:N*L(N:batch)
    def __init__(self,maxlength,acid_size,embed_dim,nhead,num_layers,dropout=0.2,num_col=12):
        super().__init__()
        self.edim=embed_dim
        self.num_col=int(num_col)
        self.conv=nn.Conv1d(embed_dim,embed_dim,2)
        self.pos_embedding=nn.Embedding(maxlength,embed_dim)
        self.charge_embedding=nn.Embedding(10,embed_dim,padding_idx=0)
        self.a_embedding=nn.Embedding(acid_size,embed_dim,padding_idx=0)
        self.phos_embedding=nn.Embedding(5,embed_dim)#修饰三种加上padding###完整版4种修饰
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,nhead,dropout=dropout)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers)
        self.rtlinear=nn.Linear(embed_dim,1)
        self.activation=nn.ReLU()
        self.mslinear=nn.Linear(embed_dim,num_col)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,peptide_tokens,peptide_length,charge,decoration,pnumber,false_samples=None):
        #input:sequence:N*L(N:batch)
        # lengths:N*L(N:batch)
        # phos:N*L(N:batch)
        #charge:N*1
        #false_samples:N*F(number of decoy of every sample)*L 这里面只存了phos的数据.
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        sequence=peptide_tokens
        device=peptide_length.device
        assert N==peptide_length.size(0)
        ll = torch.arange(0, L, device=device).unsqueeze(0)#1*L
        a_embed=self.a_embedding(sequence)#NLE

        pos_embed=self.pos_embedding(ll)#1LE
        charge_embed=self.charge_embedding(charge.unsqueeze(1).expand(N,L))

        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)

        ninput=pos_embed+a_embed+phos_embed+charge_embed#NLE

        # ninput=self.dropout(ninput)
        key_padding_mask=attentionmask(peptide_length)

        output =self.transformer(ninput.permute(1,0,2),src_key_padding_mask=key_padding_mask)#(L)NE
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1

        output=self.activation(self.conv(output.permute(1,2,0)))# NE(L-1)
        output=output.permute(0,2,1)#N*(L-1)*E
        outputms=self.dropout(self.mslinear(output))#N*(L-1)*24

        outputms=self.activation(outputms)

        outputms=outputms.reshape(N,-1)#N*((L-1)*num_col)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)
        if false_samples is None:
            return {'pred': outputms, 'sequence': sequence, 'charge': charge, "decoration": decoration,
                    "seq_len": peptide_length,
                    'pnumber': pnumber}
        # print(outputms)
        # print(torch.sum(outputms))
        else:
            false_phos_embed = self.phos_embedding(false_samples)
            false_ninput = pos_embed + a_embed + charge_embed  ##NLE
            false_ninput=false_ninput.unsqueeze(1)
            # import ipdb
            # ipdb.set_trace()
            false_ninput = false_ninput + false_phos_embed
            F = false_ninput.size(1)
            false_ninput = false_ninput.reshape(N * F, L, -1)
            false_peplen = peptide_length.expand(F, N).T.reshape(N * F)
            false_key_padding_mask = attentionmask(false_peplen)

            false_output = self.transformer(false_ninput.permute(1, 0, 2),
                                            src_key_padding_mask=false_key_padding_mask)  # (L)(N*F)E

            false_output = self.activation(self.conv(false_output.permute(1, 2, 0)))  # (N*F)E(L-1)
            false_output = false_output.permute(0, 2, 1)  # (N*F)*(L-1)*E
            false_outputms = self.dropout(self.mslinear(false_output))  # (N*F)*(L-1)*24

            false_outputms = self.activation(false_outputms)
            false_outputms=false_outputms.reshape(N*F,-1)   # (N*F)*((L-1)*num_col)
            false_seq_len = ((peptide_length - 1) * self.num_col).expand(F, N).T.reshape(N * F)

            false_masks = seq_len_to_mask(seq_len=false_seq_len)  ##加上mask

            false_outputms = false_outputms.masked_fill(false_masks.eq(False), 0)
            false_outputms = false_outputms.reshape(N, F, -1)  # N*F*((L-1)*num_col)
            return {'pred': outputms, 'sequence': sequence, 'charge': charge, "decoration": decoration,
                    "seq_len": peptide_length,
                    'pnumber': pnumber, "false_outputms": false_outputms}
