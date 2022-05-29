import pandas as pd
import os
def mixchargen(n):#######合并各个charge的json去重

    data2=pd.read_json("phosData/2charge"+str(n)+"processed.json")
    data1=pd.read_json("phosData/1charge"+str(n)+"processed.json")
    data3=pd.read_json("phosData/3charge"+str(n)+"processed.json")
    data4=pd.read_json("phosData/4charge"+str(n)+"processed.json")
    data5 = pd.read_json("phosData/5charge" + str(n) + "processed.json")
    data=pd.concat([data1,data3,data2,data4,data5],ignore_index=True)
    data=data.sort_values(by=['_psequence','psmscore'],ascending=True)
    print(data)
    print("length:",len(data))
    data=data.drop_duplicates(subset='_psequence',keep='last')
    print(data)
    print("afterlength:",len(data))
    data.to_json("phosData/Tcharge"+str(n)+"processed.json")
def mixall(dirpath):#####合并所有路径下的json去重
    dlist=[]
    for item in os.listdir(dirpath):
        if item.endswith("json"):
            itempath=os.path.join(dirpath,item)

            dlist.append(pd.read_json(itempath))
    data = pd.concat(dlist, ignore_index=True)
    data = data.sort_values(by=['_psequence', 'psmscore'], ascending=True)
    print(data)
    print("length:", len(data))
    data = data.drop_duplicates(subset=['_psequence','charge'], keep='last')
    print(data)
    print("afterlength:", len(data))
    data.to_json(dirpath+"/Tprocessed.json")
def simplemixall(dirpath):#####合并所有路径下的json不去重
    dlist=[]
    for item in os.listdir(dirpath):
        if item.endswith("json"):
            itempath=os.path.join(dirpath,item)

            dlist.append(pd.read_json(itempath))
    data = pd.concat(dlist, ignore_index=True)
    print("length:", len(data))
    data.to_json(dirpath+"/Tprocessed.json")
def mixjsons(n,jsonslist):#######合并jsonslist里面的jsonname并去重
    dlist=[]
    for jsonname in jsonslist:

        dlist.append(pd.read_json(jsonname))
    data = pd.concat(dlist, ignore_index=True)
    data = data.sort_values(by=['_psequence', 'psmscore'], ascending=True)
    print(data)
    print("length:", len(data))
    data = data.drop_duplicates(subset=['_psequence','charge'], keep='last')
    print(data)
    print("afterlength:", len(data))
    data.to_json("AllphosT/Tprocessed"+str(n)+".json")
def simplemixjsons(n,jsonslist):#######合并jsonslist里面的jsonname*****不去重#######
    dlist=[]
    for jsonname in jsonslist:

        dlist.append(pd.read_json(jsonname))
    data = pd.concat(dlist, ignore_index=True)
    data = data.sort_values(by=['_psequence', 'psmscore'], ascending=True)
    print(data)
    print("length:", len(data))
    data = data.drop_duplicates(subset=['_psequence','charge'], keep='last')
    print(data)
    print("afterlength:", len(data))
    data.to_json("phosData/Tprocessed"+str(n)+".json")
def split2charge(jsonfile):##########按charge分开json文件
    data=pd.read_json(jsonfile)
    for charge , frame in data.groupby("charge"):
        print(charge)
        frame.to_json(jsonfile[0:-5]+str(charge)+".json")
def split2phos(jsonfile):##########按mono/multi phospho分开json文件
    data=pd.read_json(jsonfile)
    multiphos=pd.DataFrame(columns=data.columns)
    for pnumber , frame in data.groupby("pnumber"):
        print(pnumber)
        if pnumber<2:

            frame.to_json(jsonfile[0:-5]+"pnumber"+str(pnumber)+".json")
        else:
            multiphos = pd.concat([multiphos,frame], ignore_index=True)
    multiphos.to_json(jsonfile[0:-5]+"pnumbermulti.json")
mixjsons("combinemann",["AllphosT/Tprocessed2.json","Task53_fintune_training_datasetp.json"])
