import pandas as pd
import numpy as np
from datapath import *
file=allirtphosTstd2
datata=pd.read_json(file)
def countphos(lis):
    return lis.count(1)

count=datata['decoration'].apply(countphos)
print('maxlength:',max(datata['sequence'].apply(len)))
print("总数：",len(count))
print("无磷酸化：",(count==0).sum())
print("单磷酸化：",(count==1).sum())
print("多磷酸化：",(count>1).sum())