import pandas as pd
import random
df=pd.read_table("msms.txt",delimiter="\t")
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
out=out.loc[out["Peptide"].str.count("1")==1]
df2=out
df2=df2.loc[(df2["exp_strip_sequence"].str.count("S")+df2["exp_strip_sequence"].str.count("T")+df2["exp_strip_sequence"].str.count("Y"))>1]
df2.reset_index(drop=True,inplace=True)
df=df2
df.to_csv("maxquant_result_biological_sample.csv",index=False)
print(df)


down=0
out = pd.DataFrame(columns=["SourceFile","Spectrum","PP.Charge","exp_strip_sequence","Peptide","key"])
for k in range(0,len(df)):
    count = 0
    sequence=df.loc[k,"key"]
    print(sequence)
    sequence=list(sequence)
    SourceFile=df.loc[k,"SourceFile"]
    Spectrum = df.loc[k, "Spectrum"]
    Charge = df.loc[k, "PP.Charge"]
    Peptide = df.loc[k, "Peptide"]
    exp_strip_sequence = df.loc[k, "exp_strip_sequence"]
    y = list(range(len(sequence)))
    sty_list=[]
    for x in range(len(sequence)):
        if sequence[x] in ["S","T","Y"] :
            sty_list.append(x)
            sequence.insert(x+1,"1")
            sequence1=''.join(sequence)
            print(sequence1)
            out.loc[down]=[SourceFile,Spectrum,Charge,exp_strip_sequence,Peptide,sequence1]
            down+=1
            sequence.remove("1")
            count+=1
            y.remove(x)
        if sequence[x]== "2":
            y.remove(x)
            y.remove(x-1)
        if sequence[x]== "3":
            y.remove(x)
            y.remove(x-1)
        if sequence[x]== "4":
            y.remove(x)
            y.remove(x+1)
        if x==len(sequence)-1:
            print(sty_list)
            stynum = 0
            if count <=len(y):
                b = random.sample(y, count)
                for c in b:
                    sty=int(sty_list[stynum])
                    sequence[c],sequence[sty] = sequence[sty],sequence[c]
                    sequence.insert(c + 1, "1")
                    stynum+=1
                    sequence1 = ''.join(sequence)
                    print(sequence1)
                    out.loc[down]=[SourceFile,Spectrum,Charge,exp_strip_sequence,Peptide,sequence1]
                    down += 1
                    sequence.remove("1")
                    sequence[c],sequence[sty] = sequence[sty],sequence[c]
            else:
                b = y
                for c in b:
                    sty = int(sty_list[stynum])
                    sequence[c],sequence[sty] = sequence[sty],sequence[c]
                    sequence.insert(c + 1, "1")
                    stynum += 1
                    sequence1 = ''.join(sequence)
                    print(sequence1)
                    out.loc[down]=[SourceFile,Spectrum,Charge,exp_strip_sequence,Peptide,sequence1]
                    down += 1
                    sequence.remove("1")
                    sequence[c],sequence[sty] = sequence[sty],sequence[c]
out.to_csv("PXD20770_Biological_sample_maxquant_newmodel_sequence.csv",index=None)