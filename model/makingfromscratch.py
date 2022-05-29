import pandas as pd
import numpy as np
from utils import *
from bidict import bidict
import time


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

def makelibraryms2(ms2frame:pd.DataFrame,name):
    ms2frame["ms2"]=ms2frame.apply(lambda row:transformoutput(row["ms2"]),axis=1)
    decorationdict=bidict({'[Phospho (STY)]':1,'[Oxidation (M)]':2,'[Carbamidomethyl (C)]':3,'[Acetyl (Protein N-term)]':4})

    lossdict = bidict({"noloss": '', "H2O": 'o', "NH3": 'n', "H3PO4": 'p', '1(+H2+O)1(+H3+O4+P)': 'op',
                '1(+H3+N)1(+H3+O4+P)': 'np',"2(+H3+O4+P)": '2p', '1(+H2+O)2(+H3+O4+P)': 'o2p',
                '1(+H3+N)2(+H3+O4+P)': 'n2p'})

    columns=["PP.PIMID",'PEP.StrippedSequence',"PrecurserMz","FragmentMz","PP.Charge",'FI.FrgType','FI.LossType','FI.Charge','FI.Intensity',
         'FI.FrgNum',"PP.iRTEmpirical","PG.UniprotIds","key","indexnumber","countphos",'length']
    ionname = "b1,bn1,bo1,b2,bn2,bo2,y1,yn1,yo1,y2,yn2,yo2,bp1,bnp1,bop1,bp2,bnp2,bop2,yp1,ynp1,yop1,yp2,ynp2,yop2," \
              "b2p1,bn2p1,bo2p1,b2p2,bn2p2,bo2p2,y2p1,yn2p1,yo2p1,y2p2,yn2p2,yo2p2".split(
        ',')

    outframe=pd.DataFrame(columns=columns)

    N=len(ms2frame)
    start=time.time()
    si=-1
    for index in range(N):
        row=ms2frame.loc[index]
        framei=pd.DataFrame(columns=columns)
        sequence = "".join([vocab.to_word(i) for i in row['repsequence']])
        decoration = row["decoration"]
        countphos=decoration.count(1)
        key=combine(sequence,decoration)
        PIMID = key
        for p in range(1, 5):
            PIMID = PIMID.replace(str(p), decorationdict.inverse[p])
        PIMID = "_" + PIMID + "_"
        charge = row["charge"]
        ms2=np.array(row['ms2'])
        length=len(sequence)
        nnn=(length-1)*36

        ###################################framei的几列
        indexnumber = range(nnn)
        isequence=[sequence]*nnn
        ikey=[key]*nnn
        icharge=[charge]*nnn
        ims2=ms2.reshape(nnn)
        icountphos=[countphos]*nnn
        iPIMID=[PIMID]*nnn
        il=[length]*nnn

        framei['PEP.StrippedSequence']=isequence
        framei["PP.PIMID"]=iPIMID
        framei["PP.Charge"]=icharge
        framei['FI.Intensity']=ims2
        framei['key']=ikey
        framei['countphos']=icountphos
        framei['indexnumber']=indexnumber
        framei['length']=il
        # assert length==len(row['ms2'])-1
        outframe=pd.concat([outframe,framei],axis=0,ignore_index=True)
        median = time.time()
        print( "making" + str(index) + "/total" + str(
            N) + "  estimated time:" + str(round((median - start) / (int(index) - si) * (N - si), 1)) + "s")
    outframe.to_csv("tempdata.csv",index=False)
    outframe.drop(index=(outframe.loc[(outframe['FI.Intensity']==0)].index),inplace=True)


    print("drop zero intensity finished...............")
    def otherattri(row):

        j = row['indexnumber'] % 36
        i = row['indexnumber'] // 36

        FrgType = ionname[j][0]
        FIcharge = ionname[j][-1]

        LossType = countloss(j)
        FIfrgnum = i + 1 if FrgType == 'b' else row['length'] - i - 1
        return FrgType,FIcharge,LossType,FIfrgnum
    outframe['i']=outframe['indexnumber'].apply(lambda x:x//36)
    outframe['j']=outframe['indexnumber'].apply(lambda x:x%36)
    outframe['FI.FrgType']=outframe["j"].apply(lambda x:ionname[x][0])
    outframe['FI.Charge'] = outframe["j"].apply(lambda x: ionname[x][-1])
    outframe['FI.LossType'] = outframe["j"].apply(lambda x: countloss(x))
    outframe['FI.FrgNum'] = outframe.apply(lambda row: row['i']+1 if row['FI.FrgType']=='b' else row['length']-row['i']-1,
                                           axis=1)


        # if down>=1000:
        #     break
    print("saving...")
    outframe.to_csv(name, index=False)
    print("successfully saved.")
    return outframe


def makelibraryirt(outframe,irtframe,name):
    #######makingdict
    def frame2irtdict(irtframe: pd.DataFrame):
        irtdict = {}
        for index, row in irtframe.iterrows():
            sequence = "".join([vocab.to_word(i) for i in row['repsequence']])
            decoration = row["decoration"]
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

            irt = row["irt"]

            irtdict[key] = irt 
        print("making dictionary success.total number of keys:", len(irtdict.keys()))
        return irtdict
    irtdict=frame2irtdict(irtframe)
    #######通用key
    outframe['PP.iRTEmpirical']=outframe.apply(lambda x:irtdict[x['key']],axis=1)
    print("saving...")
    outframe.to_csv(name, index=False)
    print("successfully saved.")
    return outframe
def dropandms2(outframe:pd.DataFrame,name):
    print(outframe.columns)
    lossdict = bidict({"noloss": '', "H2O": 'o', "NH3": 'n', "H3PO4": 'p', '1(+H2+O)1(+H3+O4+P)': 'op',
                       '1(+H3+N)1(+H3+O4+P)': 'np', "2(+H3+O4+P)": '2p', '1(+H2+O)2(+H3+O4+P)': 'o2p',
                       '1(+H3+N)2(+H3+O4+P)': 'n2p'})
    ionname = "b1,bn1,bo1,b2,bn2,bo2,y1,yn1,yo1,y2,yn2,yo2,bp1,bnp1,bop1,bp2,bnp2,bop2,yp1,ynp1,yop1,yp2,ynp2,yop2," \
              "b2p1,bn2p1,bo2p1,b2p2,bn2p2,bo2p2,y2p1,yn2p1,yo2p1,y2p2,yn2p2,yo2p2".split(
        ',')
    outframe.drop(index=(outframe.loc[(outframe['FI.Intensity'] ==0)].index), inplace=True)

    print("drop zero intensity finished...............")

    outframe['i'] = outframe['indexnumber'].apply(lambda x: x // 36)
    outframe['j'] = outframe['indexnumber'].apply(lambda x: x % 36)
    outframe['FI.FrgType'] = outframe["j"].apply(lambda x: ionname[x][0])
    outframe['FI.Charge'] = outframe["j"].apply(lambda x: ionname[x][-1])
    outframe['FI.LossType'] = outframe["j"].apply(lambda x: countloss(x))
    outframe['FI.FrgNum'] = outframe.apply(
        lambda row: row['i'] + 1 if row['FI.FrgType'] == 'b' else row['length'] - row['i'] - 1,axis=1)

    # if down>=1000:
    #     break
    print("saving...")
    outframe.to_csv(name, index=False)
    print("successfully saved.")
    return outframe
def splitmakelibraryms2(ms2frame:pd.DataFrame,name):
    ms2frame["ms2"]=ms2frame.apply(lambda row:transformoutput(row["ms2"]),axis=1)
    decorationdict=bidict({'[Phospho (STY)]':1,'[Oxidation (M)]':2,'[Carbamidomethyl (C)]':3,'[Acetyl (Protein N-term)]':4})

    lossdict = bidict({"noloss": '', "H2O": 'o', "NH3": 'n', "H3PO4": 'p', '1(+H2+O)1(+H3+O4+P)': 'op',
                '1(+H3+N)1(+H3+O4+P)': 'np',"2(+H3+O4+P)": '2p', '1(+H2+O)2(+H3+O4+P)': 'o2p',
                '1(+H3+N)2(+H3+O4+P)': 'n2p'})

    columns=["PP.PIMID",'PEP.StrippedSequence',"PrecurserMz","FragmentMz","PP.Charge",'FI.FrgType','FI.LossType','FI.Charge','FI.Intensity',
         'FI.FrgNum',"PP.iRTEmpirical","PG.UniprotIds","key","indexnumber","countphos",'length',"iii"]
    ionname = "b1,bn1,bo1,b2,bn2,bo2,y1,yn1,yo1,y2,yn2,yo2,bp1,bnp1,bop1,bp2,bnp2,bop2,yp1,ynp1,yop1,yp2,ynp2,yop2," \
              "b2p1,bn2p1,bo2p1,b2p2,bn2p2,bo2p2,y2p1,yn2p1,yo2p1,y2p2,yn2p2,yo2p2".split(
        ',')
    mout=pd.DataFrame(columns=columns)
    outframe=pd.DataFrame(columns=columns)

    N=len(ms2frame)
    start=time.time()
    si=-1
    for index in range(N):
        row=ms2frame.loc[index]
        framei=pd.DataFrame(columns=columns)
        sequence = "".join([vocab.to_word(i) for i in row['repsequence']])
        decoration = row["decoration"]
        countphos=decoration.count(1)
        key=combine(sequence,decoration)
        PIMID = key
        for p in range(1, 5):
            PIMID = PIMID.replace(str(p), decorationdict.inverse[p])
        PIMID = "_" + PIMID + "_"
        charge = row["charge"]
        ms2=np.array(row['ms2'])
        length=len(sequence)
        nnn=(length-1)*36

        ###################################framei的几列
        indexnumber = range(nnn)
        isequence=[sequence]*nnn
        ikey=[key]*nnn
        icharge=[charge]*nnn
        ims2=ms2.reshape(nnn)
        icountphos=[countphos]*nnn
        iPIMID=[PIMID]*nnn
        il=[length]*nnn
        iii=[index]*nnn

        framei['PEP.StrippedSequence']=isequence
        framei["PP.PIMID"]=iPIMID
        framei["PP.Charge"]=icharge
        framei['FI.Intensity']=ims2
        framei['key']=ikey
        framei['countphos']=icountphos
        framei['indexnumber']=indexnumber
        framei['length']=il
        framei['iii']=iii
        # assert length==len(row['ms2'])-1
        mout=pd.concat([mout,framei],axis=0,ignore_index=True)
        median = time.time()
        print( "making" + str(index) + "/total" + str(
            N) + "  estimated time:" + str(round((median - start) / (int(index) - si) * (N - si), 1)) + "s")
        if index % 200==0:

            # mout.to_csv(name[0:-4]+str(index//1000)+".csv",index=False)
            outframe=pd.concat([outframe,mout],axis=0,ignore_index=True)
            mout=pd.DataFrame(columns=columns)
            print(str(index)+"lines finished saving.")
    outframe = pd.concat([outframe, mout], axis=0, ignore_index=True)##########把最后残余的一部分拼接
    outframe.drop(index=(outframe.loc[(outframe['FI.Intensity'] < 0.0001)].index), inplace=True)

    print("drop zero intensity finished...............")

    def otherattri(row):

        j = row['indexnumber'] % 36
        i = row['indexnumber'] // 36

        FrgType = ionname[j][0]
        FIcharge = ionname[j][-1]

        LossType = countloss(j)
        FIfrgnum = i + 1 if FrgType == 'b' else row['length'] - i - 1
        return FrgType, FIcharge, LossType, FIfrgnum

    outframe['i'] = outframe['indexnumber'].apply(lambda x: x // 36)
    outframe['j'] = outframe['indexnumber'].apply(lambda x: x % 36)
    outframe['FI.FrgType'] = outframe["j"].apply(lambda x: ionname[x][0])
    outframe['FI.Charge'] = outframe["j"].apply(lambda x: ionname[x][-1])
    outframe['FI.LossType'] = outframe["j"].apply(lambda x: countloss(x))
    outframe['FI.FrgNum'] = outframe.apply(
        lambda row: row['i'] + 1 if row['FI.FrgType'] == 'b' else row['length'] - row['i'] - 1,
        axis=1)

    # if down>=1000:
    #     break
    print("saving...")
    outframe.to_csv(name, index=False)
    print("successfully saved.")
    return outframe


if __name__=="__main__":
    ms2frame = pd.read_json("repsequence.json")

    outframe=splitmakelibraryms2(ms2frame,"modeldata.csv")
    # outframe=dropandms2(pd.read_csv("tempdata.csv"),name="synthetic/makingms2JPT.csv")
    irtframe=pd.read_json("repsequenceirt.json")
    makelibraryirt(outframe,irtframe,"modelirtdata.csv")