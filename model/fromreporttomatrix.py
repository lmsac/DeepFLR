import numpy as np
import pandas as pd
from fastNLP import Vocabulary
import os

import re

#############
r"""    预处理。删除多个磷酸化位点数据，删除含有acetyl的数据
PEP.StrippedSequence 原始sequence
PP.PIMID
charge=2和3的分开做
        建立修饰toindex的Vocabulary
        对PP.PIMID 找到[修饰]替换为对应数字,放到新的field里面。找到数字位置，数字位置减一就是修饰位置。
        对FI.Charge	FI.FrgNum	FI.FrgType	FI.LossType 
        。 
        type=b1,bn1,bo1,bp1,bnp1,bop1,b2,bn2,bo2,bp2,bnp2,bop2;y1,yn1,yo1,yp1,ynp1,yop1,y2,yn2,yo2,yp2,ynp2,yop2(和b相反的number)
        产生新的dataframe如下fields：sequence,decoration,charge,b1,bn1,bo1,bp1,b2,bn2,bo2,bp2,y1,yn1,yo1,yp1,y2,yn2,yo2,yp2
        """


def countdecoration(presequence):
    ##example:  input:SDL1K2FJ4NL
    ##          output:00120400
    phos = []
    number = "0123456789"
    l = len(presequence)

    for i in range(l):

        if presequence[i] in number:
            phos[-1] = int(presequence[i])
            continue

        phos.append(0)
    return phos
def report2matrix(filename):#处理csv文件
    fields = "sequence,decoration,charge,ions,psmscore,_psequence,irt".split(',')
    fieldslen = range(len(fields))
    outputdataframe = pd.DataFrame(columns=fields)

    print(outputdataframe)
    if isinstance(filename,str):
        file = filename
        path0 = os.getcwd()
        filepath = os.path.join(path0, file)
        testdata = pd.read_csv(filepath)
    elif isinstance(filename,pd.DataFrame):
        testdata=filename
        file = "rawdata.csv"
    else:
        raise Exception
    down = 0
    decorationvocab = Vocabulary(unknown=None)
    decorationvocab.add('[Phospho (STY)]')
    decorationvocab.add('[Oxidation (M)]')
    decorationvocab.add('[Carbamidomethyl (C)]')
    lossdict = {"noloss": '', "H2O": 'o', "NH3": 'n', "H3PO4": 'p', '1(+H2+O)1(+H3+O4+P)': 'op',
                '1(+H3+N)1(+H3+O4+P)': 'np'}
    for charge, chargedata in testdata.groupby("PP.Charge"):
        for psequence, iframe in chargedata.groupby("PP.PIMID"):
            psmscore, frame = list(iframe.groupby("PSM.Score"))[-1]
            # for psmscore,frame in iframe.groupby("PSM.Score"):

            _psequence = psequence[1:-1]
            sequence = list(frame['PEP.StrippedSequence'])[0]

            #####decoration

            match = re.findall(r'\[.*?\]', _psequence)
            for _ in match:
                _psequence = _psequence.replace(_, str(decorationvocab.to_index(_)))

            print(_psequence)

            decoration = countdecoration(_psequence)
            length = len(decoration)
            #######decoration///
            # charge = list(frame['PP.Charge'])[0]
            ###########ions
            ionname = "b1,bn1,bo1,b2,bn2,bo2,y1,yn1,yo1,y2,yn2,yo2,bp1,bnp1,bop1,bp2,bnp2,bop2,yp1,ynp1,yop1,yp2,ynp2,yop2".split(
                ',')
            ions = np.zeros((length - 1, len(ionname)))

            for index, row in frame.iterrows():
                try:
                    ion = row['FI.FrgType'] + lossdict[row['FI.LossType']] + str(row['FI.Charge'])


                    j = ionname.index(ion)
                except:
                    continue
                i = row['FI.FrgNum'] - 1 if row['FI.FrgType'] == 'b' else length - row['FI.FrgNum'] - 1
                # print(i,j)
                # print("len:",length)
                ions[i, j] = row['FI.Intensity']
            irt=list(frame["PP.iRTEmpirical"])[0]
            #############ions///
            outputdataframe.loc[down] = [sequence, decoration, charge, ions, psmscore, _psequence,irt]
            down += 1
    outputdataframe.to_json(file[0:-4] + "processed.json")
    # decorationvocab.save("decorationvocab")
    # print(decorationvocab)
    print("finish")
    # print(outputdataframe)
if __name__=='__main__':

    a = pd.read_csv("DDAtestData/20200219_123137_DDALIBRARY_Fragment Report_20200219_190531.csv")  # excel的文件名###除去N乙酰化以及磷酸化大于1的肽段
    print("read data ok")
    b = a[~a["PP.PIMID"].str.contains('Acetyl')]
    b = b.loc[b["PP.PIMID"].str.count("Phospho") <= 1]
    b = b.loc[b['FI.Charge'] <= 2]
    print("finish")
    b.to_csv("DDAtestData/DDALIBRARY3nocharge>2.csv")
    # report2matrix("DDAtestData/DDALIBRARY3.csv")