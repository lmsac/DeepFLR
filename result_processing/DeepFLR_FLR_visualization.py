import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        "--modelresultfile",
        default=None,
        type=str,
        required=True,
        help="inputfile,modelresult from mgfprocess.py ",
    )
parser.add_argument(
        "--sequencefile",
        default=None,
        type=str,
        required=True,
        help="sequencefile from Targetdecoy_phosphopeptides_generation_{mono,multi}.py",
    )
parser.add_argument(
        "--outputfile",
        default="outputfileFLRPSM.csv",
        type=str,
        required=False,
        help="output filename",
    )
args = parser.parse_args()
inputfile=args.modelresultfile
sequencefile=args.sequencefile
outputfileFLRPSM=args.outputfile

def ace(instance):
    if instance[1]=="4":
        instance = list(instance)
        instance.remove("4")
        instance.insert(0, "4")
        instance = ''.join(instance)
    return instance
df=pd.read_csv(inputfile)
df["score"]=df["score"].str.replace("tensor(",'',regex=False)
df["score"]=df["score"].str.replace(".)",'',regex=False)
df["score"]=df["score"].str.replace(")",'',regex=False)
df["score"]=df["score"].astype("float")

df1=pd.read_csv(sequencefile)
df1.columns=["SourceFile","Fspectrum","PP.Charge","exp_strip_sequence","Peptide","key_x"]
df=pd.merge(df,df1,on=["SourceFile","Fspectrum","key_x"])
df=df[["SourceFile","Fspectrum","Peptide","PEP.StrippedSequence","key_x","score"]]
df["striptrue"]=df["Peptide"].str.replace("1",'',regex=False)
df["striptrue"]=df["striptrue"].str.replace("2",'',regex=False)
df["striptrue"]=df["striptrue"].str.replace("3",'',regex=False)
df["striptrue"]=df["striptrue"].str.replace("4",'',regex=False)
df["score"]=df["score"].astype("float")
df=df.sort_values(by='score',ascending=False)
df.reset_index(drop=True,inplace=True)
dfmax=df.drop_duplicates(subset=['SourceFile', 'Fspectrum', 'Peptide'])
dftruemax=dfmax.loc[dfmax["striptrue"]==dfmax["PEP.StrippedSequence"]]
dffalsemax=dfmax.loc[dfmax["striptrue"]!=dfmax["PEP.StrippedSequence"]]
dftrue=df.loc[df["striptrue"]==df["PEP.StrippedSequence"]]
dfdecoy=df.loc[df["striptrue"]!=df["PEP.StrippedSequence"]]
dftruenum=len(dftrue)
dfdecoynum=len(dfdecoy)
print(dftruenum)
print(dfdecoynum)

df=pd.concat([dfmax,dftrue])
df=df.drop_duplicates()
f =lambda x: x.score.iloc[0]-x.score.iloc[1]
dfdelta=df.groupby(["Fspectrum","SourceFile","Peptide"]).apply(f)
print("df",df)
print("dfdelta",dfdelta)
dfdelta=pd.DataFrame(dfdelta).reset_index(drop=False)
print(dfdelta)
dfdelta.columns=['Fspectrum','SourceFile',"Peptide",'deltascore']
zz=pd.merge(dfdelta,dffalsemax,on=['Fspectrum','SourceFile',"Peptide"],how='right')
zz=zz[['Fspectrum','SourceFile',"deltascore","score","key_x","striptrue","PEP.StrippedSequence"]]
print(zz.shape)

ww=pd.merge(dfdelta,dftruemax,on=['Fspectrum','SourceFile',"Peptide"],how='right')
ww=ww[['Fspectrum','SourceFile',"deltascore","score","key_x","striptrue","PEP.StrippedSequence"]]
print(ww.shape)

df=pd.concat([zz,ww])
print(df.shape)

df["deltascore"]=df["deltascore"].astype("float")
out = pd.DataFrame(columns=["cutoff","esti_FLR","PSMs"])
down=0
df["key_x"]=df["key_x"].str.replace("2",'',regex=False)
df["key_x"]=df["key_x"].str.replace("3",'',regex=False)
df["key_x"]=df["key_x"].str.replace("4",'',regex=False)

d=1
dftarget=df
dftarget=dftarget["deltascore"]
dftarget=dftarget.drop_duplicates()
dftarget=dftarget.sort_values(ascending=True)
dftarget = np.array(dftarget) #先将数据框转换为数组
dftarget = dftarget.tolist()
for a in dftarget:
    print(a)
    dfcutoff=df[df["deltascore"]>=a]
    if len(dfcutoff)==0:
        break
    dfcutoffdecoy=dfcutoff.loc[dfcutoff["striptrue"]!=dfcutoff["PEP.StrippedSequence"]]
    decoy_score = ((dftruenum+dfdecoynum)/dfdecoynum)*len(dfcutoffdecoy) / (len(dfcutoff))
    if d <= decoy_score:
        decoy_score = d
    d = min(d, decoy_score)
    PSMs=len(dfcutoff)-len(dfcutoffdecoy)
    out.loc[down] = [a,decoy_score,PSMs]
    down+=1
    if decoy_score == 0:
        break
out.to_csv(outputfileFLRPSM,index=False)