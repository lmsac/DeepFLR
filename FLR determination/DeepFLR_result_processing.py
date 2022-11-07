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
        "--outputresult",
        default="outputresult.csv",
        type=str,
        required=False,
        help="output filename",
    )

parser.add_argument(
        "--outputfileFLRPSM",
        default="outputfileFLRPSM.csv",
        type=str,
        required=False,
        help="output filename",
    )

parser.add_argument(
        "--cutoff",
        default=0,
        type=float,
        required=True,
        help="cutoff for estimated FLR",
    )
parser.add_argument(
        "--MSMSfile",
        default=None,
        type=str,
        required=True,
        help="sequencefile from Targetdecoy_phosphopeptides_generation_{mono,multi}.py",
    )    
args = parser.parse_args()
inputfile=args.modelresultfile
outputfile1=args.outputresult
sequencefile=args.sequencefile
outputfile2=args.outputfileFLRPSM
cutoff=args.cutoff

def ace(instance):
    if instance[1]=="4":
        instance = list(instance)
        instance.remove("4")
        instance.insert(0, "4")
        instance = ''.join(instance)
    return instance
df=pd.read_csv(inputfile)
print(df)
df["score"]=df["score"].str.replace("tensor(",'',regex=False)
df["score"]=df["score"].str.replace(".)",'',regex=False)
df["score"]=df["score"].str.replace(")",'',regex=False)
df["score"]=df["score"].astype("float")


df1=pd.read_csv(sequencefile)
df1.columns=["SourceFile","Fspectrum","PP.Charge","exp_strip_sequence","Peptide","key_x"]
df=pd.merge(df,df1,on=["SourceFile","Fspectrum","key_x","PP.Charge"],how="left")
df=df[["SourceFile","Fspectrum","Peptide","PEP.StrippedSequence","PP.Charge","key_x","score"]]
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

df=pd.concat([dfmax,dftrue])
df=df.drop_duplicates()
f =lambda x: x.score.iloc[0]-x.score.iloc[1]
dfdelta=df.groupby(["Fspectrum","SourceFile","Peptide"]).apply(f)
dfdelta=pd.DataFrame(dfdelta).reset_index(drop=False)
print(dfdelta)
dfdelta.columns=['Fspectrum','SourceFile',"Peptide",'deltascore']
zz=pd.merge(dfdelta,dffalsemax,on=['Fspectrum','SourceFile',"Peptide"],how='right')
zz=zz[['Fspectrum','SourceFile',"PP.Charge","deltascore","score","key_x","striptrue","PEP.StrippedSequence"]]
print(zz.shape)

ww=pd.merge(dfdelta,dftruemax,on=['Fspectrum','SourceFile',"Peptide"],how='right')
ww=ww[['Fspectrum','SourceFile',"deltascore","PP.Charge","score","key_x","striptrue","PEP.StrippedSequence"]]
print(ww.shape)

df=pd.concat([zz,ww])
df=df.drop_duplicates()
print(df.shape)

df=df.loc[df["striptrue"]==df["PEP.StrippedSequence"]]
df["deltascore"]=df["deltascore"].astype("float")
df=df[df["deltascore"]>=cutoff]  #cutoff for 0.01 estimated FLR
print(len(df))
df["key"]=df["key_x"].str.replace("2",'',regex=False)
df["key"]=df["key"].str.replace("3",'',regex=False)
df["key"]=df["key"].str.replace("4",'',regex=False)
df.reset_index(drop=True,inplace=True)
for k in range(0,len(df)):
    sequence=df.loc[k,"key"]
    sequence=list(sequence)
    m = 0
    index_list = []
    for i in range(len(sequence)):
        if sequence[i] == "1":
            i = i - m
            index_list.append(i)
            m += 1
    index_list = list(map(str, index_list))
    index_list = ";".join(index_list)
    df.loc[k,"Index"]=index_list
df = df.drop('Index', axis=1).join(
    df['Index'].str.split(";", expand=True).stack().reset_index(level=1, drop=True).rename('Index'))
df["Index"]=df["Index"].astype("int")
dfmodel=df.copy()
print(len(dfmodel))

dfmodel=dfmodel[['Fspectrum', 'SourceFile', "PP.Charge",'deltascore', 'score', 'key_x', 'striptrue',
       'key', 'Index']]
dfmodel.columns=['Spectrum','SourceFile',"PP.Charge", 'deltascore_model', 'score_model', 'key_model_1234', 'striptrue',
       'key_model_1', 'Index_model']


dfmsms=pd.read_csv("..\data\msms.txt",delimiter="\t")
dfmsms=dfmsms[['Raw file', 'Scan number',  'Sequence',  'Modified sequence',"Charge",
       'id','Peptide ID', 'Mod. peptide ID', 'Evidence ID','Phospho (STY) site IDs']]
dfmsms.columns=['SourceFile', 'Spectrum',  'striptrue',  'Peptide',"PP.Charge",
       'MSMS_ID','Peptide ID', 'Mod. peptide ID', 'Evidence ID','Phospho (STY) site IDs']
dfmsms.drop_duplicates(keep="first",inplace=True)
dfmsms.reset_index(drop=True,inplace=True)
df=pd.merge(dfmodel,dfmsms,on=["SourceFile", 'Spectrum',  "PP.Charge",'striptrue'],how="left")
df=df.dropna(subset=["Phospho (STY) site IDs"])
df.reset_index(drop=True,inplace=True)
df["Phospho (STY) site IDs"]=df["Phospho (STY) site IDs"].astype("str")
df = df.drop('Phospho (STY) site IDs', axis=1).join(
    df['Phospho (STY) site IDs'].str.split(";", expand=True).stack().reset_index(level=1, drop=True).rename('Phossite_IDs_maxq'))

df1=pd.read_table("..\data\Phospho (STY)Sites.txt",delimiter="\t")
df1=df1[['Proteins', 'Positions within proteins', 'Leading proteins', 'Protein','Phospho (STY) Probabilities','Position in peptide','Positions', 'Position','MS/MS IDs', 'Best localization MS/MS ID','Best score scan number',"id"]]
df1.columns=['Proteins', 'Positions within proteins', 'Leading proteins', 'Protein','Phospho (STY) Probabilities','Position in peptide','Positions', 'Position','MS/MS IDs', 'Best localization MS/MS ID','Best score scan number',"Phossite_IDs_maxq"]
df1["Phossite_IDs_maxq"]=df1["Phossite_IDs_maxq"].astype("str")
df=pd.merge(df,df1,on=["Phossite_IDs_maxq"],how="left")
df.drop_duplicates(keep="first",inplace=True)
df.reset_index(drop=True,inplace=True)

for k in range(len(df)):
    position_model=df.loc[k,"Index_model"]
    print(position_model)
    position_imply = df.loc[k, "Position in peptide"]
    print(position_imply)
    position_protein = df.loc[k, "Position"]
    print(position_protein)
    if not np.isnan(position_model):
        position_delta=int(position_model)-int(position_imply)
        position_protein_model=int(position_delta)+int(position_protein)
        df.loc[k,"position_protein_model"]=position_protein_model
        print(position_protein_model)

df["stripPeptide_phosprob"]=df["Phospho (STY) Probabilities"].str.replace("(","",regex=False)
df["stripPeptide_phosprob"]=df["stripPeptide_phosprob"].str.replace(")","",regex=False)
df["stripPeptide_phosprob"]=df["stripPeptide_phosprob"].str.replace(r"(\d+\.*\d*)","")
df=df.loc[df["stripPeptide_phosprob"]==df["striptrue"]]
df=df[['Spectrum', 'SourceFile', 'PP.Charge', 'deltascore_model',
       'score_model', 'key_model_1234', 'striptrue', 'key_model_1',
       'Index_model', 'Peptide', 'MSMS_ID', 'Peptide ID', 'Mod. peptide ID',
       'Evidence ID', 'Proteins',
        'Leading proteins', 'Protein','position_protein_model']]
df=df.drop_duplicates(keep="first")
df.reset_index(drop=True,inplace=True)
def combine(instance):
    x=instance["Protein"]
    y=instance["position_protein_model"]
    return str(x)+"_"+str(int(y))
df["model_proteinsite"]=df.apply(combine,axis=1)
df=df.loc[~df["Protein"].str.contains("REV__")]
df=df.loc[~df["Protein"].str.contains("CON__")]
df.to_csv("Model_proteinsite_Deep_FLR.csv",index=False)
