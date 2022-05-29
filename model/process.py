import pandas as pd
import os
a=pd.read_csv("20200126_162312_phos5_U2OS_Fragment Report_20200127_110330.csv") #excel的文件名###除去N乙酰化以及磷酸化大于1的肽段


b = a[~a["PP.PIMID"].str.contains('Acetyl')]
b=b.loc[b["PP.PIMID"].str.count("Phospho")<=1]
b.to_csv("5.csv")
print("finish")
data=pd.read_csv('5.csv')
path=os.getcwd()
path=os.path.join(path,"phosData")
# os.makedirs(path)
for charge,frame in data.groupby("PP.Charge"):
    name="5charge"+str(charge)+".csv"
    filepath=os.path.join(path,name)
    frame.to_csv(filepath,index=None)
    print(name+" saved.")
