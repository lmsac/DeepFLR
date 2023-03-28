import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP import DataSet
import torch
import os
import numpy as np
from fastNLP import BucketSampler
import pandas as pd
import json
from fastNLP.core.metrics import MetricBase,seq_len_to_mask
from fastNLP import DataSet
from fastNLP import  Instance
from fastNLP import BucketSampler
from fastNLP import DataSetIter
from fastNLP.io import Loader
from fastNLP.io import Pipe
from fastNLP import Vocabulary
from fastNLP.io import DataBundle
def peptideload(fpath):##优化
    fields=["peptide","charge","ions","qvalue"]
    datadict={}
    for field in fields:
        datadict[field]=[]
    with open(fpath,'r') as f:
        data=json.load(f)
    for i in data:
        for field in fields:
            datadict[field].append(i[field])
    return DataSet(datadict)
class NPPeptideLoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, fpath: str):
        fields = ["peptide", "charge", "ions", "qvalue"]
        datadict = {}
        for field in fields:
            datadict[field] = []
        with open(fpath, 'r') as f:
            data = json.load(f)
        for i in data:
            for field in fields:
                datadict[field].append(i[field])
        return DataSet(datadict)

    def download(self, dev_ratio=0.1, re_download=False) -> str:
       pass

class NPPeptidePipe(Pipe):
    def __init__(self,vocab=None):
        self.fields=["peptide", "charge", "ions", "qvalue"]
        self.vocab=vocab
        pass

    def _tokenize(self, data_bundle):
        for name, dataset in data_bundle.datasets.items():

            def tokenize(raw_text):
                output = list(raw_text)
                return output
            dataset.apply_field(tokenize, field_name='peptide',
                                new_field_name='peptide_tokens')
            dataset.apply_field(lambda x:len(x), field_name='peptide',
                                new_field_name='peptide_length')


        return data_bundle



    def process(self, data_bundle: DataBundle) -> DataBundle:
        self._tokenize(data_bundle)
        ionsby=sorted(data_bundle.get_dataset('train')[0]["ions"].keys())
        tensorfields = ['peptide_tensor','ions_tensor']

        if self.vocab==None:
            vocab = Vocabulary(unknown='<unk>', padding='<pad>')
        else:
            vocab=self.vocab
        vocab.from_dataset(data_bundle.get_dataset('train'),
                           field_name="peptide_tokens",
                           )
        vocab.index_dataset(*data_bundle.datasets.values(), field_name='peptide_tokens')
        data_bundle.set_vocab(vocab, 'peptide_tokens')
        #处理ions，用顺序排列转成矩阵
        def process_ions(ions):
            ions_tensor=[]
            for by in ionsby:
                ions_tensor.append(ions[by])
            ions_tensor=torch.Tensor(ions_tensor)
            max=ions_tensor.max().item()
            ions_tensor=ions_tensor/max

            return torch.log2(ions_tensor.transpose(0,1).reshape(-1,)+1)
        data_bundle.apply_field(process_ions,'ions','target')
        # data_bundle.apply_field(lambda x:len(x), 'ions_tensor', 'ions_tensor_length')
        #set input and output
        data_bundle.set_input('peptide_tokens','peptide_length')
        data_bundle.set_target('target')
        # data_bundle.set_target('tgt_tokens', 'tgt_seq_len')

        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = NPPeptideLoader().load(paths)
        return self.process(data_bundle)
#######################################################################################磷酸化数据读取，自己写的json所以读取方式和上面非磷酸化不太一样
class PPeptideLoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, fpath: str):
        fields = ["peptide", "charge", "ions","decoration",'irt',"_id"]
        datadict = {}
        for field in fields:
            datadict[field] = []
        data=pd.read_json(fpath)
        datadict["peptide"]+=list(data["sequence"])
        datadict["charge"] += list(data["charge"])
        datadict["ions"] += list(data["ions"])
        datadict["decoration"] += list(data["decoration"])
        datadict["_id"]=range(len(datadict["peptide"]))
        try:
            datadict["irt"] += list(data["irt"])

            return DataSet(datadict)
        except:
            datadict["irt"]=range(len(datadict["ions"]))
            return DataSet(datadict)


    def download(self, dev_ratio=0.1, re_download=False) -> str:
       pass

class PPeptidePipe(Pipe):
    def __init__(self,vocab=None):
        self.fields=["peptide", "charge", "ions", "decoration",'irt',"_id"]
        self.vocab=vocab
        pass

    def _tokenize(self, data_bundle):
        for name, dataset in data_bundle.datasets.items():

            def tokenize(raw_text):
                output = list(raw_text)
                return output
            dataset.apply_field(tokenize, field_name='peptide',
                                new_field_name='peptide_tokens')
            dataset.apply_field(lambda x:len(x), field_name='peptide',
                                new_field_name='peptide_length')


        return data_bundle



    def process(self, data_bundle: DataBundle) -> DataBundle:
        self._tokenize(data_bundle)
        if self.vocab==None:

            vocab = Vocabulary(unknown='<unk>', padding='<pad>')
        else:
            vocab=self.vocab
        vocab.from_dataset(data_bundle.get_dataset('train'),
                           field_name="peptide_tokens",
                           )
        vocab.index_dataset(*data_bundle.datasets.values(), field_name='peptide_tokens')
        data_bundle.set_vocab(vocab, 'peptide_tokens')
        #处理ions，用顺序排列转成矩阵
        def process_ions(ions_tensor):

            ions_tensor=torch.Tensor(ions_tensor)
            max=ions_tensor.max().item()
            ions_tensor=ions_tensor/max

            return torch.log2(ions_tensor.reshape(-1,)+1)
        data_bundle.apply_field(process_ions,'ions','target')
        def countphos(lis):
            return lis.count(1)
        data_bundle.apply_field(countphos,field_name='decoration',new_field_name='pnumber')
        def add_cls_sep(instance):
            peptide_tokens=instance["peptide_tokens"]
            clsindex=len(vocab)
            sepindex=len(vocab)+1
            input_ids=[]
            decoration_ids=[]
            decoration=instance["decoration"]
            for i,token in enumerate(peptide_tokens):
                if i==0:
                    input_ids.append(clsindex)
                    input_ids.append(token)
                    decoration_ids.append(0)
                    decoration_ids.append(decoration[i])
                else:
                    input_ids.append(sepindex)
                    input_ids.append(token)
                    decoration_ids.append(0)
                    decoration_ids.append(decoration[i])
            return {"input_ids":input_ids,"decoration_ids":decoration_ids}
        data_bundle.apply_more(add_cls_sep)
        # data_bundle.apply_field(normalize, 'irt', 'irt')
        # data_bundle.apply_field(lambda x:len(x), 'ions_tensor', 'ions_tensor_length')
        #set input and output
        data_bundle.set_input("decoration_ids",'peptide_tokens','peptide_length','decoration','charge','pnumber',"input_ids","_id")
        data_bundle.set_target('target','irt')
        # data_bundle.set_target('tgt_tokens', 'tgt_seq_len')
        data_bundle.set_pad_val("decoration",0)
        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = PPeptideLoader().load(paths)
        return self.process(data_bundle)
################################################################################只有肽段和charge没有target的产生数据
class PPeptideLoader_notarget(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, fpath: str):
        fields = ["peptide", "charge", "ions","decoration",'irt']
        datadict = {}
        for field in fields:
            datadict[field] = []
        data=pd.read_csv(fpath)
        datadict["peptide"]+=list(data["key"])
        datadict["charge"] += list(data["PP.Charge"])
        ############下面三个是没有的，通过后续操作产生
        datadict["ions"] += range(len(data["key"]))
        datadict["decoration"] += list(data["key"])
        datadict["irt"]=range(len(data["key"]))
        return DataSet(datadict)


    def download(self, dev_ratio=0.1, re_download=False) -> str:
       pass
def countdecoration(presequence):
    ##example:  input:SDL1K2FJ4NL
    ##          output:00120400
    phos = []
    number = "0123456789"
    l = len(presequence)
    sig=0
    if "4" in presequence:
        if presequence.index("4") !=0:
            presequence=presequence.replace('4','')
            presequence='4'+presequence
            # print(presequence)
    for i in range(l):
        if presequence[i]=='4':
            assert i==0
            sig=1
            phos.append(4)
            continue
        elif presequence[i] in number:
            phos[-1] = int(presequence[i])
            continue

        phos.append(0)
        if sig:
            sig=0
            phos.pop()
    return phos

def dropnumber(string:str):
    for i in range(5):
        string=string.replace(str(i),"")
    return string
def createzerotarget(peptide_tokens,num_col=36):
    return np.zeros(((len(peptide_tokens)-1)*num_col))
class PPeptidePipe_notarget(Pipe):
    def __init__(self,vocab=None):
        self.fields=["peptide", "charge", "ions", "decoration",'irt']
        self.vocab=vocab
        pass

    def _tokenize(self, data_bundle):
        for name, dataset in data_bundle.datasets.items():

            def tokenize(raw_text):
                output = list(raw_text)
                return output
            dataset.apply_field(tokenize, field_name='peptide',
                                new_field_name='peptide_tokens')
            dataset.apply_field(lambda x:len(x), field_name='peptide',
                                new_field_name='peptide_length')


        return data_bundle



    def process(self, data_bundle: DataBundle) -> DataBundle:
        data_bundle.apply_field(countdecoration,'peptide','decoration')#decoration搞定
        data_bundle.apply_field(dropnumber, 'peptide', 'peptide')#peptide搞定
        self._tokenize(data_bundle)
        if self.vocab==None:

            vocab = Vocabulary(unknown='<unk>', padding='<pad>')
        else:
            vocab=self.vocab
        vocab.from_dataset(data_bundle.get_dataset('train'),
                           field_name="peptide_tokens",
                           )
        vocab.index_dataset(*data_bundle.datasets.values(), field_name='peptide_tokens')
        data_bundle.set_vocab(vocab, 'peptide_tokens')



        def countphos(lis):
            return lis.count(1)
        data_bundle.apply_field(countphos,field_name='decoration',new_field_name='pnumber')
        data_bundle.apply_field(createzerotarget, field_name='peptide_tokens', new_field_name='target')
        def add_cls_sep(instance):
            peptide_tokens=instance["peptide_tokens"]
            clsindex=len(vocab)
            sepindex=len(vocab)+1
            input_ids=[]
            decoration_ids=[]
            decoration=instance["decoration"]
            for i,token in enumerate(peptide_tokens):
                if i==0:
                    input_ids.append(clsindex)
                    input_ids.append(token)
                    decoration_ids.append(0)
                    decoration_ids.append(decoration[i])
                else:
                    input_ids.append(sepindex)
                    input_ids.append(token)
                    decoration_ids.append(0)
                    decoration_ids.append(decoration[i])
            return {"input_ids":input_ids,"decoration_ids":decoration_ids}
        data_bundle.apply_more(add_cls_sep)
        # data_bundle.apply_field(normalize, 'irt', 'irt')
        # data_bundle.apply_field(lambda x:len(x), 'ions_tensor', 'ions_tensor_length')
        #set input and output
        data_bundle.set_input("decoration_ids",'peptide_tokens','peptide_length','decoration','charge','pnumber',"input_ids")
        # data_bundle.apply_field(normalize, 'irt', 'irt')
        # data_bundle.apply_field(lambda x:len(x), 'ions_tensor', 'ions_tensor_length')
        #set input and output

        data_bundle.set_target('target','irt')
        # data_bundle.set_target('tgt_tokens', 'tgt_seq_len')
        data_bundle.set_pad_val("decoration",0)
        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = PPeptideLoader_notarget().load(paths)
        return self.process(data_bundle)

if __name__=="__main__":
    databundle = PPeptidePipe_notarget(vocab=None).process_from_file(paths="synthetic/syntheticpep_Sigma_pycharm_monophos.csv"
                                                                           )
    totaldata = databundle.get_dataset("train")
    print(totaldata)