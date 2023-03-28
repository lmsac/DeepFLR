import pandas as pd
needmzdrop="modelirtdata.csv"
savename="modelirtdatamonomz.csv"
df=pd.read_csv(needmzdrop)
def drop(row):
    key=row["key"]
    key=list(key)
    by = row["FI.FrgType"]
    loss = row["FI.LossType"]
    FrgNum=row["FI.FrgNum"]
    FrgNum=int(FrgNum)
    qienum=0
    pure = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "U", "W", "Y", "V"]
    FrgNumC=FrgNum
    if by=="b":
        keyq=key[:FrgNumC]
        for x in pure:
            qienum+=keyq.count(x)
        while qienum != FrgNum:
            FrgNumC+=1
            keyq = key[:FrgNumC]
            qienum=0
            for x in pure:
                qienum += keyq.count(x)
        if key[FrgNumC] in ["1","2","3"]:
            FrgNumC+=1
            keyq = key[:FrgNumC]
        if loss in ["2(+H3+O4+P)",'1(+H2+O)2(+H3+O4+P)','1(+H3+N)2(+H3+O4+P)']:
            if keyq.count("1") !=2:
                return True
        if loss in ["H3PO4",'1(+H2+O)1(+H3+O4+P)','1(+H3+N)1(+H3+O4+P)']:
            if keyq.count("1") ==0:
                return True
    if by=="y":
        FrgNumindex=len(key)-FrgNum
        keyq = key[FrgNumindex:]
        qienum=0
        for x in pure:
            qienum+=keyq.count(x)
        while qienum != FrgNum:
            FrgNumindex=FrgNumindex-1
            keyq = key[FrgNumindex:]
            qienum=0
            for x in pure:
                qienum += keyq.count(x)
        if key[FrgNumindex] in ["Ace"]:
            FrgNumindex=FrgNumindex-1
            keyq = key[FrgNumindex:]
        if loss in ["2(+H3+O4+P)",'1(+H2+O)2(+H3+O4+P)','1(+H3+N)2(+H3+O4+P)']:
            if keyq.count("1") !=2:
                return True
        if loss in ["H3PO4",'1(+H2+O)1(+H3+O4+P)','1(+H3+N)1(+H3+O4+P)']:
            if keyq.count("1") ==0:
                return True
df["drop"]=df.apply(drop,axis=1)
df=df[~df['drop'].isin([True])]
df.drop(['drop'], axis = 1,inplace=True)
df.reset_index(drop=True,inplace=True)
df.to_csv("makingms2irtJPTdropnew.csv",index=False)


dict={"A":71.037114, "R":156.101111,"N":114.042927,"D":115.026943,"C":103.009185,"E":129.042593,"Q":128.058578,"G":57.021464,\
      "H":137.058912,"I":113.084064,"L":113.084064,"K":128.094963,"M":131.040485,"F":147.068414,"P":97.052764,"S":87.032028,\
      "T":101.047679,"U":150.95363,"W":186.079313,"Y":163.06332,"V":99.068414, \
      "Phos":79.966331,"Ox":15.994915,"Car":57.021464,"Ace":42.010565,\
      "H20":18.010565,"NH3":17.026549,"H3PO4":97.976897,"proton":1.007276}
#H的质量，氢的原子质量减去一个电子的质量，如果纯粹质子算出来却是1.007100
def PrecurserMzzzbb(row):
    key=row["key"]
    key=list(key)
    phosz=["1"]
    oxz=["2"]
    carz=["3"]
    acez=["4"]
    key=["Phos" if x in phosz else x for x in key]
    key = ["Ox" if x in oxz else x for x in key]
    key = ["Car" if x in carz else x for x in key]
    key = ["Ace" if x in acez else x for x in key]
    a=0
    for i in key:
        a+=dict[i]
    charge=row["PP.Charge"]
    charge=int(charge)
    weight=a+dict["H20"]+charge*dict["proton"]
    mz=weight/charge
    return mz

df['PrecurserMz']=df.apply(PrecurserMzzzbb,axis=1)
    #碎片mz
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


    by=row["FI.FrgType"]
    loss = row["FI.LossType"]
    losstype={"noloss":0, "H2O": 18.010565, "NH3": 17.026549, "H3PO4":97.976897, '1(+H2+O)1(+H3+O4+P)': 115.987462, \
    '1(+H3+N)1(+H3+O4+P)': 115.003446, "2(+H3+O4+P)": 195.953794, '1(+H2+O)2(+H3+O4+P)': 213.964359,'1(+H3+N)2(+H3+O4+P)': 212.980343}
    ficharge = row["FI.Charge"]
    ficharge=int(ficharge)
    #fragnum有几就表示从哪里切的过程有几个氨基酸 左或右
    FrgNum=row["FI.FrgNum"]
    FrgNum=int(FrgNum)
    qienum=0
    pure = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "U", "W", "Y", "V"]
    FrgNumC=FrgNum
    if by=="b":
        keyq=key[:FrgNumC]
        for x in pure:
            qienum+=keyq.count(x)
        while qienum != FrgNum:
            FrgNumC+=1
            keyq = key[:FrgNumC]
            qienum=0
            for x in pure:
                qienum += keyq.count(x)
        if key[FrgNumC] in ["Ox","Phos","Car"]:
            FrgNumC+=1
            keyq = key[:FrgNumC]
        b=0
        for i in keyq:
            b+=dict[i]
        FIweight=b+ficharge*dict["proton"]-losstype[loss]
        fimz=FIweight/ficharge
        # print(fimz)
        return fimz
    if by=="y":
        FrgNumindex=len(key)-FrgNum
        keyq = key[FrgNumindex:]
        qienum=0
        for x in pure:
            qienum+=keyq.count(x)
        while qienum != FrgNum:
            FrgNumindex=FrgNumindex-1
            keyq = key[FrgNumindex:]
            qienum=0
            for x in pure:
                qienum += keyq.count(x)
        if key[FrgNumindex] in ["Ace"]:
            FrgNumindex=FrgNumindex-1
            keyq = key[FrgNumindex:]
        c=0
        for i in keyq:
            c+=dict[i]
        FIweight = c + ficharge * dict["proton"]+dict["H20"] - losstype[loss]
        fimz = FIweight / ficharge
        return fimz
df["FragmentMz"]=df.apply(FragmentMzzzbb,axis=1)
df.to_csv(savename,index=None)