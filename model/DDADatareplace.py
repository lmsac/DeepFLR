import torch
import fastNLP
import pandas as pd
import numpy as np
from model import *
from fastNLP import Vocabulary
import re
from utils import *
##########ms2dict.对每行碎片找到位置用预测的dia替代
def transformoutput(x):
    #训练的时候ms2使用了log2（x+1)，这里变回来
    return np.exp2(x)-1
def frame2dict(ms2frame:pd.DataFrame):
    ms2dict={}
    for index,row in ms2frame.iterrows():
        sequence="".join([vocab.to_word(i) for i in row['repsequence']])
        decoration=row["decoration"]
        key=''
        length=len(sequence)
        assert length==len(decoration)
        for i in range(length):
            if decoration[i] == 4:
                key += str(decoration[i])
                key += sequence[i]
                continue
            else:
                key += sequence[i]
                if decoration[i]==0:
                    continue
                else:
                    key+=str(decoration[i])


        charge="charge"+str(row["charge"])
        key=key+charge
        ms2=transformoutput(np.array(row["ms2"]))
        ms2dict[key]=ms2
    print("making dictionary success.total number of keys:",len(ms2dict.keys()))
    return ms2dict

def frame2irtdict(irtframe:pd.DataFrame):
    irtdict={}
    for index,row in irtframe.iterrows():
        sequence="".join([vocab.to_word(i) for i in row['repsequence']])
        decoration=row["decoration"]
        key=''
        length=len(sequence)
        assert length==len(decoration)
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



        irt=row["irt"]

        irtdict[key]=irt
    print("making dictionary success.total number of keys:",len(irtdict.keys()))
    return irtdict

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
def replace(ddaframe,ms2dict):####对ddaframe进行替换,返回新的frame
    decorationvocab = Vocabulary(unknown=None)
    decorationvocab.add('[Phospho (STY)]')
    decorationvocab.add('[Oxidation (M)]')
    decorationvocab.add('[Carbamidomethyl (C)]')
    lossdict = {"noloss": '', "H2O": 'o', "NH3": 'n', "H3PO4": 'p', '1(+H2+O)1(+H3+O4+P)': 'op',
                '1(+H3+N)1(+H3+O4+P)': 'np'}
    l=len(ddaframe.index)
    print("total number to replace:",l)
    count=0
    error=0

    for index, row in ddaframe.iterrows():
        psequence=row["PP.PIMID"]
        _psequence = psequence[1:-1]
        sequence = list(row['PEP.StrippedSequence'])[0]

        #####decoration

        match = re.findall(r'\[.*?\]', _psequence)
        for _ in match:
            _psequence = _psequence.replace(_, str(decorationvocab.to_index(_)))

        print("{} in total {}".format(count+1,l))

        length = len(sequence)
        #######decoration///
        # charge = list(frame['PP.Charge'])[0]
        ###########ions
        key=_psequence+"charge"+str(row["PP.Charge"])
        ionname = "b1,bn1,bo1,b2,bn2,bo2,y1,yn1,yo1,y2,yn2,yo2,bp1,bnp1,bop1,bp2,bnp2,bop2,yp1,ynp1,yop1,yp2,ynp2,yop2".split(
            ',')
        try:
            ions = ms2dict[key]
        except:
            error+=1
            continue
        try:
                ion = row['FI.FrgType'] + lossdict[row['FI.LossType']] + str(row['FI.Charge'])

                j = ionname.index(ion)
        except:
            continue
        i = row['FI.FrgNum'] - 1 if row['FI.FrgType'] == 'b' else length - row['FI.FrgNum'] - 1
        # print(i,j)
        # print("len:",length)



        ddaframe.loc[index,'FI.Intensity'] = ions[i, j]
        # print(ddaframe.loc[index]['FI.Intensity'] )
        count+=1
    print("successfuly replaced number:",count)
    print("failure number:",error)
    return ddaframe

def fastreplace(ddaframe,ms2dict):####对ddaframe进行替换,返回新的frame
    decorationvocab = Vocabulary(unknown=None)
    decorationvocab.add('[Phospho (STY)]')
    decorationvocab.add('[Oxidation (M)]')
    decorationvocab.add('[Carbamidomethyl (C)]')
    decorationvocab.add('[Acetyl (Protein N-term)]')
    lossdict = {"noloss": '', "H2O": 'o', "NH3": 'n', "H3PO4": 'p', '1(+H2+O)1(+H3+O4+P)': 'op',
                '1(+H3+N)1(+H3+O4+P)': 'np',"2(+H3+O4+P)": '2p', '1(+H2+O)2(+H3+O4+P)': 'o2p',
                '1(+H3+N)2(+H3+O4+P)': 'n2p'}
    l=len(ddaframe.index)
    print("total number to replace:",l)

    def rowreplace(frame):##这里的frame其实是row
        psequence = frame["PP.PIMID"]
        _psequence = psequence[1:-1]
        sequence = list(frame['PEP.StrippedSequence'])[0]

        #####decoration

        match = re.findall(r'\[.*?\]', _psequence)
        for _ in match:
            _psequence = _psequence.replace(_, str(decorationvocab.to_index(_)))


        length = len(sequence)
        #######decoration///
        # charge = list(frame['PP.Charge'])[0]
        ###########ions
        key = _psequence + "charge" + str(frame["PP.Charge"])
        ionname = "b1,bn1,bo1,b2,bn2,bo2,y1,yn1,yo1,y2,yn2,yo2,bp1,bnp1,bop1,bp2,bnp2,bop2,yp1,ynp1,yop1,yp2,ynp2,yop2," \
                  "b2p1,bn2p1,bo2p1,b2p2,bn2p2,bo2p2,y2p1,yn2p1,yo2p1,y2p2,yn2p2,yo2p2".split(
            ',')
        try:
            ions = ms2dict[key]
        except:
            key=key[1:]
            ions = ms2dict[key]

        try:
            ion = frame['FI.FrgType'] + lossdict[frame['FI.LossType']] + str(frame['FI.Charge'])

            j = ionname.index(ion)
        except:
            return frame['FI.Intensity']
        i = frame['FI.FrgNum'] - 1 if frame['FI.FrgType'] == 'b' else length - frame['FI.FrgNum'] - 1
        # print(i,j)
        # print("len:",length)

        return ions[i, j]
    ddaframe['FI.Intensity']=ddaframe.apply(lambda row:rowreplace(row),axis=1)
    return ddaframe

def fastreplaceirt(ddaframe,irtdict):####对ddaframe进行替换,返回新的frame
    decorationvocab = Vocabulary(unknown=None)
    decorationvocab.add('[Phospho (STY)]')
    decorationvocab.add('[Oxidation (M)]')
    decorationvocab.add('[Carbamidomethyl (C)]')
    decorationvocab.add('[Acetyl (Protein N-term)]')
    l=len(ddaframe.index)
    print("total number to replace:",l)

    def rowreplace(frame):##这里的frame其实是row
        psequence = frame["PP.PIMID"]
        _psequence = psequence[1:-1]
        sequence = list(frame['PEP.StrippedSequence'])[0]

        #####decoration

        match = re.findall(r'\[.*?\]', _psequence)
        for _ in match:
            _psequence = _psequence.replace(_, str(decorationvocab.to_index(_)))


        length = len(sequence)
        #######decoration///
        # charge = list(frame['PP.Charge'])[0]
        ###########ions
        key = _psequence
        try:
            irtdict[key]
        except:
            key=key[1:]

        return irtdict[key]
    ddaframe["PP.iRTEmpirical"]=ddaframe.apply(lambda row:rowreplace(row),axis=1)
    return ddaframe
def resultfastreplace(ddaframe,ms2dict):####对ddaframe进行替换,返回新的frame
    decorationvocab = Vocabulary(unknown=None)
    decorationvocab.add('[Phospho (STY)]')
    decorationvocab.add('[Oxidation (M)]')
    decorationvocab.add('[Carbamidomethyl (C)]')
    lossdict = {"noloss": '', "H2O": 'o', "NH3": 'n', "H3PO4": 'p', '1(+H2+O)1(+H3+O4+P)': 'op',
                '1(+H3+N)1(+H3+O4+P)': 'np'}
    l=len(ddaframe.index)
    print("total number to replace:",l)

    def rowreplace(frame):##这里的frame其实是row
        psequence = frame["ModifiedPeptide"]
        _psequence = psequence[1:-1]
        sequence = list(frame['StrippedPeptide'])[0]

        #####decoration

        match = re.findall(r'\[.*?\]', _psequence)
        for _ in match:
            _psequence = _psequence.replace(_, str(decorationvocab.to_index(_)))


        length = len(sequence)
        #######decoration///
        # charge = list(frame['PP.Charge'])[0]
        ###########ions
        key = _psequence + "charge" + str(frame["PrecursorCharge"])
        ionname = "b1,bn1,bo1,b2,bn2,bo2,y1,yn1,yo1,y2,yn2,yo2,bp1,bnp1,bop1,bp2,bnp2,bop2,yp1,ynp1,yop1,yp2,ynp2,yop2".split(
            ',')
        try:
            ions = ms2dict[key]
        except:
            return frame['RelativeIntensity']

        try:
            ion = frame['FragmentType'] + lossdict[frame['FragmentLossType']] + str(frame['FragmentCharge'])

            j = ionname.index(ion)
        except:
            return frame['RelativeIntensity']
        i = frame['FragmentNumber'] - 1 if frame['FragmentType'] == 'b' else length - frame['FragmentNumber'] - 1

        # print("len:",length)

        return ions[i, j]
    ddaframe['RelativeIntensity']=ddaframe.apply(lambda row:rowreplace(row),axis=1)
    return ddaframe

def resultfastreplaceirt(ddaframe,irtdict):####对ddaframe进行替换,返回新的frame
    decorationvocab = Vocabulary(unknown=None)
    decorationvocab.add('[Phospho (STY)]')
    decorationvocab.add('[Oxidation (M)]')
    decorationvocab.add('[Carbamidomethyl (C)]')

    l=len(ddaframe.index)
    print("total number to replace:",l)

    def rowreplace(frame):##这里的frame其实是row
        psequence = frame["ModifiedPeptide"]
        _psequence = psequence[1:-1]
        sequence = list(frame['StrippedPeptide'])[0]

        #####decoration

        match = re.findall(r'\[.*?\]', _psequence)
        for _ in match:
            _psequence = _psequence.replace(_, str(decorationvocab.to_index(_)))


        length = len(sequence)
        #######decoration///
        # charge = list(frame['PP.Charge'])[0]
        ###########ions
        key = _psequence
        return irtdict[key]
    ddaframe["iRT"]=ddaframe.apply(lambda row:rowreplace(row),axis=1)
    return ddaframe

################data
if __name__=="__main__":
    repsequence="repsequence.json"
    repsequenceirt="repsequenceirt.json"
    rawdata="zhaodan_only_phos_DDA_exp.csv"
    raw=pd.read_csv(rawdata)
    drop1=raw.drop(["PP.EmpiricalRT"],axis=1)
    drop1.to_csv(rawdata[:-4]+"droprt.csv",index=False)
    drop2=drop1[drop1["FI.Charge"]<2.9]
    drop2.to_csv(rawdata[:-4]+"droprtcharge<2.csv",index=False)
    ms2frame=pd.read_json(repsequence)

    ms2dict=frame2dict(ms2frame)
    print(ms2dict.keys())
    ddaframe=drop2
    replaced=fastreplace(ddaframe,ms2dict)
    print(replaced['FI.Intensity'])
    replaced.to_csv(rawdata[:-4]+"droprtcharge<2ms2replaced.csv", index=False)

    irtframe=pd.read_json(repsequenceirt)
    irtdict=frame2irtdict(irtframe)
    print(irtdict.keys())
    ddaframe=replaced
    print("readfinish")
    replaced=fastreplaceirt(ddaframe,irtdict)
    print(replaced["PP.iRTEmpirical"])
    replaced.to_csv(rawdata[:-4]+"droprtcharge<2ms2irtreplaced.csv", index=False)