##############将模型出现的结果json填入一开始的肽段文件中

import pandas as pd
import numpy as np
from utils import *
from bidict import bidict
import time
import os
def drop(row):
        key = row["key"]
        key = list(key)
        by = row["FI.FrgType"]
        loss = row["FI.LossType"]
        FrgNum = row["FI.FrgNum"]
        FrgNum = int(FrgNum)
        qienum = 0
        pure = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "U", "W", "Y", "V"]
        FrgNumC = FrgNum
        if by == "b":
            keyq = key[:FrgNumC]
            for x in pure:
                qienum += keyq.count(x)
            while qienum != FrgNum:
                FrgNumC += 1
                keyq = key[:FrgNumC]
                qienum = 0
                for x in pure:
                    qienum += keyq.count(x)
            if key[FrgNumC] in ["1", "2", "3"]:
                FrgNumC += 1
                keyq = key[:FrgNumC]
            if loss in ["2(+H3+O4+P)", '1(+H2+O)2(+H3+O4+P)', '1(+H3+N)2(+H3+O4+P)']:
                if keyq.count("1") != 2:
                    return True
            if loss in ["H3PO4", '1(+H2+O)1(+H3+O4+P)', '1(+H3+N)1(+H3+O4+P)']:
                if keyq.count("1") == 0:
                    return True
        if by == "y":
            FrgNumindex = len(key) - FrgNum
            keyq = key[FrgNumindex:]
            qienum = 0
            for x in pure:
                qienum += keyq.count(x)
            while qienum != FrgNum:
                FrgNumindex = FrgNumindex - 1
                keyq = key[FrgNumindex:]
                qienum = 0
                for x in pure:
                    qienum += keyq.count(x)
            if key[FrgNumindex] in ["Ace"]:
                FrgNumindex = FrgNumindex - 1
                keyq = key[FrgNumindex:]
            if loss in ["2(+H3+O4+P)", '1(+H2+O)2(+H3+O4+P)', '1(+H3+N)2(+H3+O4+P)']:
                if keyq.count("1") != 2:
                    return True
            if loss in ["H3PO4", '1(+H2+O)1(+H3+O4+P)', '1(+H3+N)1(+H3+O4+P)']:
                if keyq.count("1") == 0:
                    return True
def transformoutput(x):
    #训练的时候ms2使用了log2（x+1)，这里变回来
    return np.exp2(np.array(x))-1
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

def countloss(j):
    ionname = "b1,bn1,bo1,b2,bn2,bo2,y1,yn1,yo1,y2,yn2,yo2,bp1,bnp1,bop1,bp2,bnp2,bop2,yp1,ynp1,yop1,yp2,ynp2,yop2," \
              "b2p1,bn2p1,bo2p1,b2p2,bn2p2,bo2p2,y2p1,yn2p1,yo2p1,y2p2,yn2p2,yo2p2".split(
        ',')
    lossdict = bidict({"noloss": '', "H2O": 'o', "NH3": 'n', "H3PO4": 'p', '1(+H2+O)1(+H3+O4+P)': 'op',
                       '1(+H3+N)1(+H3+O4+P)': 'np', "2(+H3+O4+P)": '2p', '1(+H2+O)2(+H3+O4+P)': 'o2p',
                       '1(+H3+N)2(+H3+O4+P)': 'n2p'})
    l = len(ionname[j])
    if l == 2:
        return "noloss"
    else:
        return lossdict.inverse[ionname[j][1:-1]]

def combinerr(rawdataname,ms2framename):
    rawdata=pd.read_csv(rawdataname)
    ms2frame=pd.read_json(ms2framename)
    ms2frame["ms2"]=ms2frame.apply(lambda row:transformoutput(row["ms2"]),axis=1)
    def tokey(row):
        sequence = "".join([vocab.to_word(i) for i in row['repsequence']])
        decoration = row["decoration"]
        key = combine(sequence, decoration)
        return key
    ms2frame["key"]=ms2frame.apply(tokey,axis=1)
    # rawdata.columns=['SourceFile', 'Spectrum', 'PP.Charge', 'key', 'stripsequence']
    ms2frame=ms2frame.sort_values(by=["key","charge"],ascending=True,ignore_index=True)
    rawdata=rawdata.sort_values(by=["key","PP.Charge"],ascending=True,ignore_index=True)
    ccdata=pd.concat([rawdata,ms2frame],axis=1)
    # ccdata.to_csv("FDR/ccdata.csv")
    print("combinedata finished")
    return ccdata

def splitmakelibraryms2(name):######ms2frame就是上述输出的ccframe

    decorationdict=bidict({'[Phospho (STY)]':1,'[Oxidation (M)]':2,'[Carbamidomethyl (C)]':3,'[Acetyl (Protein N-term)]':4})

    lossdict = bidict({"noloss": '', "H2O": 'o', "NH3": 'n', "H3PO4": 'p', '1(+H2+O)1(+H3+O4+P)': 'op',
                '1(+H3+N)1(+H3+O4+P)': 'np',"2(+H3+O4+P)": '2p', '1(+H2+O)2(+H3+O4+P)': 'o2p',
                '1(+H3+N)2(+H3+O4+P)': 'n2p'})

    columns=["SourceFile","Spectrum","PP.PIMID",'PEP.StrippedSequence',"PrecurserMz","FragmentMz","PP.Charge",'FI.FrgType','FI.LossType','FI.Charge','FI.Intensity',
         'FI.FrgNum',"PP.iRTEmpirical","PG.UniprotIds","key","indexnumber","countphos",'length',"iii","Fspectrum"]
    ionname = "b1,bn1,bo1,b2,bn2,bo2,y1,yn1,yo1,y2,yn2,yo2,bp1,bnp1,bop1,bp2,bnp2,bop2,yp1,ynp1,yop1,yp2,ynp2,yop2," \
              "b2p1,bn2p1,bo2p1,b2p2,bn2p2,bo2p2,y2p1,yn2p1,yo2p1,y2p2,yn2p2,yo2p2".split(
        ',')
    
    # outframe=combine_in_memory()
    # outframe.drop(index=(outframe.loc[(outframe['FI.Intensity'] < 0.0001)].index), inplace=True)

    def combine_disk_frame(filepath):
        outframe=pd.DataFrame(columns=["SourceFile","Spectrum","PP.PIMID",'PEP.StrippedSequence',"PrecurserMz","FragmentMz","PP.Charge",'FI.FrgType','FI.LossType','FI.Charge','FI.Intensity',
         'FI.FrgNum',"PP.iRTEmpirical","PG.UniprotIds","key","indexnumber","countphos",'length',"iii","Fspectrum"])
        for idx,file in enumerate(os.listdir(filepath)):
            if idx <70:
                continue
            if file.endswith("json"):

                print("reading file with index : ",str(idx)," ",str(file))
                filename=os.path.join(filepath,file)
                concatframe=pd.read_json(filename)
                
                concatframe['i'] = concatframe['indexnumber'].apply(lambda x: x // 36)
                concatframe['j'] = concatframe['indexnumber'].apply(lambda x: x % 36)
                concatframe['FI.FrgType'] = concatframe["j"].apply(lambda x: ionname[x][0])
                concatframe['FI.Charge'] = concatframe["j"].apply(lambda x: ionname[x][-1])
                concatframe['FI.LossType'] = concatframe["j"].apply(lambda x: countloss(x))
                concatframe['FI.FrgNum'] = concatframe.apply(
                lambda row: row['i'] + 1 if row['FI.FrgType'] == 'b' else row['length'] - row['i'] - 1,
                axis=1)
                concatframe["drop"] = concatframe.apply(drop, axis=1)
                concatframe = concatframe[~concatframe['drop'].isin([True])]
                concatframe.drop(['drop'], axis=1, inplace=True)
                concatframe.reset_index(drop=True, inplace=True)

                # concatframe.to_json(os.path.join(filepath,"drops",file))
                # import ipdb
                # ipdb.set_trace()

                outframe=pd.concat([outframe, concatframe], axis=0, ignore_index=True)
            
        return outframe
    temppath="./TempSavingframe"
    outframe=combine_disk_frame(temppath)
    print("drop zero intensity finished...............")



    # if down>=1000:
    #     break
    print("saving...")
    outframe.to_csv(name, index=False)
    print("successfully saved.")
    return outframe
import argparse
def main():
    print("You are combining jsons from disk.##################")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        default=None,
        # default="./output_train/train_pred_gold_ner.json",
        type=str,
        required=True,
        help="DDA raw file",
    )
    parser.add_argument(
        "--repsequencefile",
        default=None,
        # default="./output_train/train_pred_gold_ner.json",
        type=str,
        required=True,
        help="output filename",
    )
    parser.add_argument(
        "--outputfile",
        default="combineoutput.csv",
        # default="./output_train/train_pred_gold_ner.json",
        type=str,
        # required=True,
        help="output filename",
    )
    args = parser.parse_args()
    df=splitmakelibraryms2(
                        args.inputfile[0:-4]+"model.csv")
    print("combinerawresult finished")
    outputfilename=args.inputfile[0:-4]+"model.csv"
    needmzdrop = outputfilename
    savename = needmzdrop[0:-4]+"monomz70-final.csv"
    # df = pd.read_csv(needmzdrop)
    # df.drop(['Unnamed: 0'], axis = 1,inplace=True)
    # df.drop(['kkk'], axis = 1,inplace=True)
    # df.to_csv("makingms2irtSigmamonomz.csv",index=False)
    print(df["key"])


    

    dict = {"A": 71.037114, "R": 156.101111, "N": 114.042927, "D": 115.026943, "C": 103.009185, "E": 129.042593,
            "Q": 128.058578, "G": 57.021464, \
            "H": 137.058912, "I": 113.084064, "L": 113.084064, "K": 128.094963, "M": 131.040485, "F": 147.068414,
            "P": 97.052764, "S": 87.032028, \
            "T": 101.047679, "U": 150.95363, "W": 186.079313, "Y": 163.06332, "V": 99.068414, \
            "Phos": 79.966331, "Ox": 15.994915, "Car": 57.021464, "Ace": 42.010565, \
            "H20": 18.010565, "NH3": 17.026549, "H3PO4": 97.976897, "proton": 1.007276}

    # H的质量，氢的原子质量减去一个电子的质量，如果纯粹质子算出来却是1.007100
    def PrecurserMzzzbb(row):
        key = row["key"]
        key = list(key)
        phosz = ["1"]
        oxz = ["2"]
        carz = ["3"]
        acez = ["4"]
        key = ["Phos" if x in phosz else x for x in key]
        key = ["Ox" if x in oxz else x for x in key]
        key = ["Car" if x in carz else x for x in key]
        key = ["Ace" if x in acez else x for x in key]
        a = 0
        for i in key:
            a += dict[i]
        charge = row["PP.Charge"]
        charge = int(charge)
        weight = a + dict["H20"] + charge * dict["proton"]
        mz = weight / charge
        return mz

    df['PrecurserMz'] = df.apply(PrecurserMzzzbb, axis=1)

    # 碎片mz
    def FragmentMzzzbb(row):
        key = row["key"]
        key = list(key)
        phosz = ["1"]
        oxz = ["2"]
        carz = ["3"]
        acez = ["4"]
        key = ["Phos" if x in phosz else x for x in key]
        key = ["Ox" if x in oxz else x for x in key]
        key = ["Car" if x in carz else x for x in key]
        key = ["Ace" if x in acez else x for x in key]

        by = row["FI.FrgType"]
        loss = row["FI.LossType"]
        losstype = {"noloss": 0, "H2O": 18.010565, "NH3": 17.026549, "H3PO4": 97.976897,
                    '1(+H2+O)1(+H3+O4+P)': 115.987462, \
                    '1(+H3+N)1(+H3+O4+P)': 115.003446, "2(+H3+O4+P)": 195.953794, '1(+H2+O)2(+H3+O4+P)': 213.964359,
                    '1(+H3+N)2(+H3+O4+P)': 212.980343}
        ficharge = row["FI.Charge"]
        ficharge = int(ficharge)
        # fragnum有几就表示从哪里切的过程有几个氨基酸 左或右
        FrgNum = row["FI.FrgNum"]
        FrgNum = int(FrgNum)
        qienum = 0
        pure = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "U", "W", "Y", "V"]
        FrgNumC = FrgNum
        if by == "b":
            keyq = key[:FrgNumC]
            for x in pure:
                qienum += keyq.count(x)
            while qienum != FrgNum:
                FrgNumC += 1
                keyq = key[:FrgNumC]
                qienum = 0
                for x in pure:
                    qienum += keyq.count(x)
            if key[FrgNumC] in ["Ox", "Phos", "Car"]:
                FrgNumC += 1
                keyq = key[:FrgNumC]
            b = 0
            for i in keyq:
                b += dict[i]
            FIweight = b + ficharge * dict["proton"] - losstype[loss]
            fimz = FIweight / ficharge
            # print(fimz)
            return fimz
        if by == "y":
            FrgNumindex = len(key) - FrgNum
            keyq = key[FrgNumindex:]
            qienum = 0
            for x in pure:
                qienum += keyq.count(x)
            while qienum != FrgNum:
                FrgNumindex = FrgNumindex - 1
                keyq = key[FrgNumindex:]
                qienum = 0
                for x in pure:
                    qienum += keyq.count(x)
            if key[FrgNumindex] in ["Ace"]:
                FrgNumindex = FrgNumindex - 1
                keyq = key[FrgNumindex:]
            c = 0
            for i in keyq:
                c += dict[i]
            FIweight = c + ficharge * dict["proton"] + dict["H20"] - losstype[loss]
            fimz = FIweight / ficharge
            return fimz

    df["FragmentMz"] = df.apply(FragmentMzzzbb, axis=1)
    df.to_csv(savename, index=None)
if __name__=="__main__":
    main()