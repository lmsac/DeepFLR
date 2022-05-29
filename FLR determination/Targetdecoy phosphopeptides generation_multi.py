import pandas as pd
import random
from itertools import combinations

def countdecoration(sequence):
    ##example:  input:SDL1K2FJ4NL
    ##          output:00120400
    phos = []
    number = "0123456789"
    l = len(sequence)
    sig=0
    for i in range(l):
        if sequence[i]=='4':
            assert i==0
            sig=1
            phos.append(4)
            continue
        elif sequence[i] in number:
            phos[-1] = int(sequence[i])
            continue

        phos.append(0)
        if sig:
            sig=0
            phos.pop()
    return phos

df=pd.read_table("msms.txt",delimiter="\t")
print("start")
out = pd.DataFrame(columns=["SourceFile","Spectrum","PP.Charge","Peptide","exp_strip_sequence","key"])
out["SourceFile"]=df["Raw file"]
out["Spectrum"]=df["Scan number"]
out["PP.Charge"]=df["Charge"]
out["Peptide"]=df["Modified sequence"]
out=out.drop_duplicates(keep="first")
out.reset_index(drop=True,inplace=True)
out["Peptide"]=out["Peptide"].str.replace("_",'',regex=False)
out["Peptide"]=out["Peptide"].str.replace("(Phospho (STY))",'1',regex=False)
out["Peptide"]=out["Peptide"].str.replace("(Oxidation (M))",'2',regex=False)
out["Peptide"]=out["Peptide"].str.replace("(Acetyl (Protein N-term))",'4',regex=False)
out["Peptide"]=out["Peptide"].str.replace("C",'C3',regex=False)
out=out.loc[~out["Peptide"].str.contains("4M2")]
out=out.loc[~out["Peptide"].str.contains("4S")]
out=out.loc[~out["Peptide"].str.contains("4T")]
out=out.loc[~out["Peptide"].str.contains("4Y")]
out=out.loc[~out["Peptide"].str.contains("4C")]
out["exp_strip_sequence"]=out["Peptide"].str.replace("1",'',regex=False)
out["exp_strip_sequence"]=out["exp_strip_sequence"].str.replace("4",'',regex=False)
out["exp_strip_sequence"]=out["exp_strip_sequence"].str.replace("3",'',regex=False)
out["exp_strip_sequence"]=out["exp_strip_sequence"].str.replace("2",'',regex=False)
out["key"]=out["Peptide"].str.replace("1",'',regex=False)
out=out.loc[out["Peptide"].str.count("1")==2] #
out["phos_num"]=out["Peptide"].apply(lambda x: x.count("1"))
df2=out
df2=df2.loc[(df2["exp_strip_sequence"].str.count("S")+df2["exp_strip_sequence"].str.count("T")+df2["exp_strip_sequence"].str.count("Y"))>df2["phos_num"]]
df2.reset_index(drop=True,inplace=True)
df=df2.copy()
df.to_csv("maxquant_biphos_result_biological_sample.csv",index=False)
# print(df)


#插入的话，插入一个，下一个就要+1
#decoy换文字，怎么考虑index，组合或者用0
for filenumber in range(0,1):
    down=0
    out = pd.DataFrame(columns=["SourceFile","Spectrum","PP.Charge","exp_strip_sequence","Peptide","key"])
    for k in range(0,len(df)):
        count = 0
        sequence=df.loc[k,"key"]
        # print(sequence)
        modified_list = countdecoration(sequence)
        # print("modified_list",modified_list)
        SourceFile=df.loc[k,"SourceFile"]
        Spectrum = df.loc[k, "Spectrum"]
        Charge = df.loc[k, "PP.Charge"]
        Peptide = df.loc[k, "Peptide"]
        exp_strip_sequence = df.loc[k, "exp_strip_sequence"]
        striptrue=list(exp_strip_sequence)
        phosnum=df.loc[k,"phos_num"]
        # print(phosnum)
        y = list(range(len(striptrue)))
        sty_list=[]
        decoyaminolist=[]
        styaminolist=[]
        for x in range(len(modified_list)):
            if modified_list[x]== 2:
                y.remove(x)
            if modified_list[x]== 3:
                y.remove(x)
            if modified_list[x]== 4:
                y.remove(x)
            if striptrue[x] in ["S","T","Y"] :
                sty_list.append(x)
                y.remove(x)
        # print("y",y)
        # print("sty_list",sty_list)
        styaminolist=list(combinations(sty_list, phosnum))
        decoyaminolist = random.choices(y, k=len(styaminolist) * phosnum)
        # print("decoyaminolist",decoyaminolist)
        # print("styaminolist", styaminolist)
        for x in styaminolist:
            modified_list1 = modified_list.copy()
            striptrue1=striptrue.copy()
            for y in x:
                modified_list1[y]=1
            for i in range(len(modified_list1)):
                if modified_list1[i]>=1:
                    striptrue1[i]=striptrue1[i]+str(modified_list1[i])
            sequence1 = ''.join(striptrue1)
            # print(sequence1)
            out.loc[down] = [SourceFile, Spectrum, Charge, exp_strip_sequence, Peptide, sequence1]
            down += 1
        for x in range(len(styaminolist)):
            decoylist_0=list(styaminolist[x])
            for decoynum in range(len(decoylist_0)):
                decoylist=decoylist_0.copy()
                decoylist[decoynum]=decoyaminolist[x*phosnum+decoynum]
                # print("decoylist",decoylist)
                modified_list1 = modified_list.copy()
                striptrue1 = striptrue.copy()
                for y in decoylist:
                    modified_list1[y] = 1
                sty = styaminolist[x][decoynum]
                c = decoyaminolist[x*phosnum+decoynum]
                striptrue1[c], striptrue1[sty] = striptrue1[sty], striptrue1[c]
                for i in range(len(modified_list1)):
                    if modified_list1[i] >= 1:
                        striptrue1[i] = striptrue1[i] + str(modified_list1[i])
                sequence1 = ''.join(striptrue1)
                # print(sequence1)
                out.loc[down] = [SourceFile, Spectrum, Charge, exp_strip_sequence, Peptide, sequence1]
                down += 1
    out.to_csv("PXD3344_biphos_sequence_trial1.csv",index=None)