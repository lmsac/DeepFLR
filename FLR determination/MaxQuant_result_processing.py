import pandas as pd
import re
import numpy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        "--inputfile",
        default=None,
        type=str,
        required=True,
        help="inputfile,searching result from Maxquant(msms.txt)",
    )

parser.add_argument(
        "--inputfile1",
        default=None,
        type=str,
        required=True,
        help="inputfile,searching result from Maxquant(Phospho (STY)Sites.txt)",
    )

parser.add_argument(
        "--outputresult",
        default="outputresult.csv",
        type=str,
        required=False,
        help="output filename, proteinsites identified by MaxQuant with score >=0.99",
    )
args = parser.parse_args()

inputfile=args.inputfile
inputfile1=args.inputfile1
outputfile=args.outputresult

df=pd.read_csv(inputfile,delimiter="\t")
df=df[['Raw file', 'Scan number',  'Sequence',  'Modified sequence',"Charge",
       'id','Peptide ID', 'Mod. peptide ID', 'Evidence ID',"Phospho (STY) Probabilities",'Phospho (STY) site IDs']]
df.columns=['SourceFile', 'Spectrum',  'striptrue',  'Peptide',"PP.Charge",
       'MSMS_ID','Peptide ID', 'Mod. peptide ID', 'Evidence ID',"Score",'Phospho (STY) site IDs']
df=df.dropna(subset=["Phospho (STY) site IDs"])
df=df.drop_duplicates(keep="first")
df.reset_index(drop=True,inplace=True)
df["Peptide"]=df["Peptide"].str.replace("_",'',regex=False)
df["Peptide"]=df["Peptide"].str.replace("(Phospho (STY))",'1',regex=False)
df["Peptide"]=df["Peptide"].str.replace("(Oxidation (M))",'2',regex=False)
df["Peptide"]=df["Peptide"].str.replace("(Acetyl (Protein N-term))",'4',regex=False)
df=df.loc[~df["Peptide"].str.contains("4M2")]
df=df.loc[~df["Peptide"].str.contains("4S")]
df=df.loc[~df["Peptide"].str.contains("4T")]
df=df.loc[~df["Peptide"].str.contains("4Y")]
df=df.loc[~df["Peptide"].str.contains("4C")]
df["Peptide"]=df["Peptide"].str.replace("4",'',regex=False)
df["Peptide"]=df["Peptide"].str.replace("2",'',regex=False)
#不能用去掉4,2的peptide产生model sequence
df["key"]=df["Peptide"].str.replace("1",'',regex=False)
df["phos_num"]=df["Peptide"].apply(lambda x: x.count("1"))
df=df.loc[df["Peptide"].str.count("1")<=2]
df=df.loc[(df["Peptide"].str.count("S")+df["Peptide"].str.count("T")+df["Peptide"].str.count("Y"))>df["phos_num"]] #> ==
df.reset_index(drop=True,inplace=True)
# df.to_csv("del.csv",index=False)
#分数提取,提取括号内的数字，找最大值
for k in range(0,len(df)):
    a=df.loc[k,"Score"]
    phosnum=df.loc[k,"phos_num"]
    b= re.findall(r"(\d+\.*\d*)", a)
    b=[float(i) for i in b]
    b = sorted(b, reverse=True)
    b=b[phosnum-1]
    df.loc[k,"Score"]=b
df["Score"]=df["Score"].astype("float")
df.reset_index(drop=True,inplace=True)
df.to_csv("graph_maxquant.csv",index=False)

df["Score"]=df["Score"].astype("float")
df=df[df["Score"]>=0.99]
df["striptrue"]=df["Peptide"].str.replace("1",'',regex=False)
df.reset_index(drop=True,inplace=True)
print(df)
for k in range(0,len(df)):
    sequence=df.loc[k,"Peptide"]
    sequence=list(sequence)
    m=0
    index_list=[]
    for i in range(len(sequence)):
        if sequence[i]=="1":
            i=i-m
            index_list.append(i)
            m += 1
    index_list = list(map(str, index_list))
    index_list=";".join(index_list)
    df.loc[k,"Index_Maxquant"]=index_list
print("*")

df = df.drop('Index_Maxquant', axis=1).join(
    df['Index_Maxquant'].str.split(";", expand=True).stack().reset_index(level=1, drop=True).rename('Index_Maxquant'))
df["Index_Maxquant"]=df["Index_Maxquant"].astype("int")


df["Phospho (STY) site IDs"]=df["Phospho (STY) site IDs"].astype("str")
df = df.drop('Phospho (STY) site IDs', axis=1).join(
    df['Phospho (STY) site IDs'].str.split(";", expand=True).stack().reset_index(level=1, drop=True).rename('Phossite_IDs_maxq'))
df1=pd.read_table(inputfile1,delimiter="\t")
df1=df1[['Proteins', 'Positions within proteins', 'Leading proteins', 'Protein',
         'Phospho (STY) Probabilities','Position in peptide','Positions', 'Position',
         'MS/MS IDs', 'Best localization MS/MS ID','Best score scan number',"id"]]
df1.columns=['Proteins', 'Positions within proteins', 'Leading proteins', 'Protein',
             'Phospho (STY) Probabilities','Position in peptide','Positions', 'Position',
             'MS/MS IDs', 'Best localization MS/MS ID','Best score scan number',"Phossite_IDs_maxq"]
df1["Phossite_IDs_maxq"]=df1["Phossite_IDs_maxq"].astype("str")
df=pd.merge(df,df1,on=["Phossite_IDs_maxq"],how="left")
df.drop_duplicates(keep="first")



for k in range(len(df)):
    position_model=df.loc[k,"Index_Maxquant"]
    position_imply = df.loc[k, "Position in peptide"]
    position_protein = df.loc[k, "Position"]
    if not np.isnan(position_model):
        position_delta=int(position_model)-int(position_imply)
        position_protein_model=int(position_delta)+int(position_protein)
        df.loc[k,"position_protein_Maxquant"]=position_protein_model
df["stripPeptide_phosprob"]=df["Phospho (STY) Probabilities"].str.replace("(","",regex=False)
df["stripPeptide_phosprob"]=df["stripPeptide_phosprob"].str.replace(")","",regex=False)
df["stripPeptide_phosprob"]=df["stripPeptide_phosprob"].str.replace(r"(\d+\.*\d*)","")
df=df.loc[df["stripPeptide_phosprob"]==df["striptrue"]]
print(df.columns)
df=df[['Spectrum', 'SourceFile', 'PP.Charge', "Score", 'striptrue',
       'Index_Maxquant', 'Peptide', 'MSMS_ID', 'Peptide ID', 'Mod. peptide ID',
       'Evidence ID', 'Proteins',
        'Leading proteins', 'Protein','position_protein_Maxquant']]
df=df.drop_duplicates(keep="first")
df.reset_index(drop=True,inplace=True)
def combine(instance):
    x=instance["Protein"]
    y=instance["position_protein_Maxquant"]
    return str(x)+"_"+str(int(y))
df["Maxquant_proteinsite"]=df.apply(combine,axis=1)
df=df.loc[~df["Protein"].str.contains("REV__")]
df=df.loc[~df["Protein"].str.contains("CON__")]
df.to_csv(outputfile,index=False)
