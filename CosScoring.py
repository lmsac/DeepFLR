import pandas as pd
import numpy as np
import re
import argparse
############
parser=argparse.ArgumentParser()
parser.add_argument(
        "--scoringfile",
        default="1.pdeep2.ions.json",
        # default="./output_train/train_pred_gold_ner.json",
        type=str,
        # required=True,
        help="pdeep2 predicted json file",
    )
parser.add_argument(
        "--DDAprocessedfile",
        default="mouse_monophosprocessed.json",
        # default="./output_train/train_pred_gold_ner.json",
        type=str,
        # required=True,
        help="DDA processed file",
    )
args = parser.parse_args()
########
pdeep2=pd.read_json(args.scoringfile)
print(pdeep2.columns)
def combine(sequence,decoration):
    key = ''
    length = len(sequence)
    assert length == len(decoration)
    for i in range(length):
        if decoration[i] == 4:
            key += str(decoration[i])
            key += sequence[i]
            continue
        else:
            key += sequence[i]
            if decoration[i] == 0:
                continue
            else:
                key += str(decoration[i])
    return key
# for i in range(len(pdeep2)):
#     print(pdeep2["peptide"][i])
#     print(pdeep2["ions"][i])
#     print(pdeep2["modification"][i])

def modification2key(row):
    decorationdict = {}
    decorationdict['Phospho'] = 1
    decorationdict['Oxidation'] = 2
    decorationdict['Carbamidomethyl'] = 3
    decorationdict['Acetyl'] = 4
    modification=row["modification"]
    sequence=row["peptide"]
    length=len(row["peptide"])
    decoration=[0]*length

    for mod in modification.split(";"):
        ind=int(re.findall('(\d+)',mod)[0])
        dtype=re.findall(r'[(](.*?)[)]',mod,re.S)[0]
        decoration[ind-1]=decorationdict[dtype]
    ####decoration OK
    key=combine(sequence,decoration)
    charge = "charge" + str(row["charge"])
    key=key+charge
    return key
def countdecoration(presequence):
    ##example:  input:SDL1K2FJ4NL
    ##          output:00120400
    phos = []
    number = "0123456789"
    l = len(presequence)
    sig=0
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
def ions(row):
    length=len(row["peptide"])
    ionname = "b1,bn1,bo1,b2,bn2,bo2,y1,yn1,yo1,y2,yn2,yo2,bp1,bnp1,bop1,bp2,bnp2,bop2,yp1,ynp1,yop1,yp2,ynp2,yop2," \
                          "b2p1,bn2p1,bo2p1,b2p2,bn2p2,bo2p2,y2p1,yn2p1,yo2p1,y2p2,yn2p2,yo2p2".split(
                    ',')
    ions = np.zeros((length - 1, len(ionname)))
    for key,item in row["ions"].items():
        j=ionname.index(key)
        if "y" in key:

            ions[:, j] = np.array(item[::-1])
        else:
            ions[:, j] = np.array(item)
    return ions
pdeep2["_ions"]=pdeep2["ions"]
pdeep2["_key"]=pdeep2.apply(lambda x:x["_psequence"]+"charge"+str(x["charge"]),axis=1)
# print(pdeep2["_ions"][0])
# print(pdeep2["_key"][0])
rawdata=pd.read_json(args.DDAprocessedfile)
rawdata["_key"]=rawdata.apply(lambda x:x["_psequence"]+"charge"+str(x["charge"]),axis=1)
rawdata["_ions"]=rawdata["ions"]
print(rawdata["_key"][0])
print(rawdata["_ions"][0])
datadict={}
for i in range(len(rawdata)):
    datadict[rawdata["_key"][i]]=rawdata["_ions"][i]

def addrawdata(row):
    try:
        return datadict[row["_key"]]
    except:
        return np.zeros(row["_ions"].shape())
pdeep2["t_ions"]=pdeep2.apply(addrawdata,axis=1)
def countscore(row):
    import torch
    from torch import Tensor
    return torch.cosine_similarity(Tensor(row["_ions"]).reshape(-1),Tensor(row["t_ions"]).reshape(-1),dim=-1).item()
pdeep2["score"]=pdeep2.apply(countscore,axis=1)
scorelist=list(pdeep2["score"])
if 0 in scorelist:
    scorelist.remove(0)
print(scorelist)
print(np.mean(scorelist))
print(np.median(scorelist))
pdeep2["score"].to_csv("CosScoringtestresult.csv",index=False)