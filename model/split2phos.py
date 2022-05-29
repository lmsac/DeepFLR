import pandas as pd
import os
def sumT(lis):
    s=sum(lis)
    if s==0 :
        return True
    else:
        return False
def split2phos_(filename):
    dirpath=os.path.dirname(filename)
    datata = pd.read_json(filename)
    datata['count==0']=datata['decoration'].apply(sumT)
    for i,data in datata.groupby('count==0',as_index=False):
        nname=filename[:-5]+str(i)+".json"
        data.to_json(nname)
split2phos_("phosData/Tprocessed2.json")