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
        "--outputresult",
        default="outputresult.csv",
        type=str,
        required=False,
        help="output filename",
    )
args = parser.parse_args()

inputfile=args.inputfile
outputfile=args.outputresult

df=pd.read_table(inputfile,delimiter="\t")
out = pd.DataFrame(columns=["SourceFile","Spectrum","PP.Charge","Peptide","key","Score"])
out["SourceFile"]=df["Raw file"]
out["Spectrum"]=df["Scan number"]
out["PP.Charge"]=df["Charge"]
out["Peptide"]=df["Modified sequence"]
out["Score"]=df["Phospho (STY) Probabilities"]
out=out.drop_duplicates(keep="first")
out.reset_index(drop=True,inplace=True)
out["Peptide"]=out["Peptide"].str.replace("_",'',regex=False)
out["Peptide"]=out["Peptide"].str.replace("(Phospho (STY))",'1',regex=False)
out["Peptide"]=out["Peptide"].str.replace("(Oxidation (M))",'2',regex=False)
out["Peptide"]=out["Peptide"].str.replace("(Acetyl (Protein N-term))",'4',regex=False)
out["Peptide_1234"]=out["Peptide"].str.replace("C",'C3',regex=False)

out=out.loc[~out["Peptide"].str.contains("4M2")]
out=out.loc[~out["Peptide"].str.contains("4S")]
out=out.loc[~out["Peptide"].str.contains("4T")]
out=out.loc[~out["Peptide"].str.contains("4Y")]
out=out.loc[~out["Peptide"].str.contains("4C")]
out["Peptide"]=out["Peptide"].str.replace("4",'',regex=False)
out["Peptide"]=out["Peptide"].str.replace("2",'',regex=False)

out["key"]=out["Peptide"].str.replace("1",'',regex=False)
df=out
df["phos_num"]=df["Peptide"].apply(lambda x: x.count("1"))
df=df.loc[df["Peptide"].str.count("1")<=2]
df=df.loc[(df["Peptide"].str.count("S")+df["Peptide"].str.count("T")+df["Peptide"].str.count("Y"))>df["phos_num"]]
df.reset_index(drop=True,inplace=True)
print(df)


for k in range(0,len(df)):
    a=df.loc[k,"Score"]
    b= re.findall(r"(\d+\.*\d*)", a)
    b=[float(i) for i in b]
    b=max(b)
    df.loc[k,"Score"]=b
df["Score"]=df["Score"].astype("float")
df.reset_index(drop=True,inplace=True)
# df.to_csv("test.csv",index=False)
df.to_csv(outputfile,index=False)

dfcutoff=df[df["Score"]>=0.75] 
print("PSMs:",dfcutoff.shape)
