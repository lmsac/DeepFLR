from datapath import *
import pandas as pd
from utils import *
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
def MPIMID(key):
    from bidict import bidict
    decorationdict=bidict({'[Phospho (STY)]':1,'[Oxidation (M)]':2,'[Carbamidomethyl (C)]':3,'[Acetyl (Protein N-term)]':4})
    for p in range(1, 5):
        key = key.replace(str(p), decorationdict.inverse[p])
    PIMID = "_" + key + "_"
    return PIMID
nanfile="/remote-home/yxwang/Finalcut/fintune/20200413_145038_PXD14525_fintune_Fragment_Report_20200420_120949nansequence.json"
jsonfile="/remote-home/yxwang/Finalcut/fintune/20200413_145038_PXD14525_fintune_Fragment_Report_20200420_120949.json"
csvfile="/remote-home/yxwang/Finalcut/fintune/20200413_145038_PXD14525_fintune_Fragment_Report_20200420_120949.csv"

# nanseq=pd.read_json(nanfile)
# for col in nanseq.columns:
#     nanseq[col]=nanseq.apply(lambda x:x.loc[col][0],axis=1)
# nanseq["_sequence"]=nanseq.apply(lambda x:"".join([vocab.to_word(i) for i in x.loc["nansequence"]]),axis=1)
# nanseq["_PIMID_"]=nanseq.apply(lambda x:MPIMID(combine(x["_sequence"],x["decoration"])),axis=1)
# print(nanseq)
# nanseq.to_json(nanfile[:-5]+"PIMID.json")


nanseq=pd.read_json(nanfile[:-5]+"PIMID.json")
b=pd.read_json(jsonfile)

d=pd.read_csv("/remote-home/yxwang/Finalcut/fintune/maxquant_result_biological_sample.csv")
# print(all["sequence"]==nan1)
# print(all["sequence"]==nan2)
# print(len(b))
# for ind in range(len(nanseq)):
#     nani=nanseq.loc[ind,"_sequence"]
#     charge=nanseq.loc[ind,"charge"]
#     decoration=nanseq.loc[ind,'decoration']
    
#     b=b.loc[~((b["sequence"]==nani)&(b["charge"]==charge))]


# print(len(b))
# b.to_json(jsonfile[0:-5]+"dropnan.json")

# c=pd.read_csv(csvfile)
# print(len(c))
# for ind in range(len(nanseq)):
#     PIMID=nanseq.loc[ind,"_PIMID_"]
#     charge=nanseq.loc[ind,"charge"]
#     decoration=nanseq.loc[ind,'decoration']
    
#     c=c.loc[~((c["PP.PIMID"]==PIMID)&(c["PP.Charge"]==charge))]
#     print("ind:"+str(ind)+"lenc:"+str(len(c)))

# print(len(c))
# c.to_csv(csvfile[0:-4]+"dropnan.csv")

print(len(d))
for ind in range(len(nanseq)):
    import ipdb
    ipdb.set_trace()
    sequence=nanseq.loc[ind,"_sequence"]
    charge=nanseq.loc[ind,"charge"]
    decoration=nanseq.loc[ind,'decoration']
    
    d=d.loc[~((d["exp_strip_sequence"]==sequence)&(d["PP.Charge"]==charge))]
    print("ind:"+str(ind)+" lenc:"+str(len(d)))

print(len(d))
d.to_csv("fintune/maxquantdropnan.csv")