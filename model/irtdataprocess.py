import torch
from torch import Tensor as tt
import torch.nn as nn
import math
import numpy as np
from torch.nn import CosineSimilarity
from matrixwithdict import countdecoration
from fastNLP import Vocabulary
import pandas as pd
import re
a=pd.read_json("AllphosTirt/Tprocessed.json")
print("总长度：",len(a))
stddict={}
irtdict={}
print(a.columns)
for x,y in a.groupby("PP.PIMID"):#######计算每个肽段对应的std和平均irt
    std1=np.std(y["irt0"])
    irt=np.mean(y["irt0"])
    stddict[x]=std1
    irtdict[x]=irt
def getppmid(row):
    return row["PP.PIMID"]
a['std']=a.apply(lambda row:stddict[getppmid(row)],axis=1)
a['irt']=a.apply(lambda row:irtdict[getppmid(row)],axis=1)
a=a.drop_duplicates(subset=["PP.PIMID"])
print("afterlen:",len(a))
a.to_json("AllphosTirt/irtprocessedstdall.json")
b=a.loc[a['std']<=1]
print("std<=1:",len(b))
b.to_json("AllphosTirt/irtprocessed<=1.json")

c=a.loc[a['std']<=2]
print("std<=2:",len(c))
c.to_json("AllphosTirt/irtprocessed<=2.json")
