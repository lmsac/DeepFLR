import pandas as pd
from pyteomics import mgf
import re
import os
import numpy as np
############先读取转换为碎片形式的结果
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        "--inputfile",
        default=None,
        # default="./output_train/train_pred_gold_ner.json",
        type=str,
        # required=True,
        help="input file",
    )
parser.add_argument(
        "--outputfile",
        default="decoyoutputscore.csv",
        # default="./output_train/train_pred_gold_ner.json",
        type=str,
        # required=True,
        help="output filename",
    )
parser.add_argument(
        "--ppm",
        default=25,

        type=int,
        # required=True,
        help="ppm for matching",
    )

parser.add_argument(
        "--mgfdatafold",
        default=None,
        # default="./output_train/train_pred_gold_ner.json",
        type=str,
        # required=True,
        help="mgfdatafold ,every mgf in it",
    )
parser.add_argument(
    "--do_mgfprocess", action="store_true", help="first time do mgfprocess to get json file"
)
parser.add_argument(
    "--do_scoreprediction", action="store_true", help="do score prediction"
)
parser.add_argument(
    "--save_score", action="store_true", help="do score prediction"
)
args = parser.parse_args()
mgfdatafold=args.mgfdatafold

################创建mgfdataframe
if args.do_mgfprocess:

    jsonfold=os.path.join(mgfdatafold,"json")
    if os.path.exists(jsonfold):
        0
    else:
        os.mkdir(jsonfold)

    for filename in os.listdir(mgfdatafold):
        if filename.endswith("mgf"):
            mgfname=os.path.join(mgfdatafold,filename)
            print(mgfname)
            mgfdata=pd.DataFrame(columns=["SourceFile","Spectrum","key","intensity","mz"])
            namere = re.compile(r'File:"(.*?)"', re.S)  #最小匹配
            cNre = re.compile(r'controllerNumber=(\d+)', re.S)
            scanre=re.compile(r'scan=(.*?)"', re.S)
            down=0
            for k,spectrum in enumerate(mgf.read(mgfname)):

                params = spectrum.get('params')

                title = params.get('title')
                sourcefile=namere.findall(title)[0]
                cNumber=cNre.findall(title)[0]
                scan=scanre.findall(title)[0]
                Spectrum=str(scan)
                ###
                # print(Spectrum)
                intensity=list(spectrum.get('intensity array'))
                mz=list(spectrum.get('m/z array'))
                if len(intensity)!=len(mz):
                    print(k,mgfname)
                mgfdata.loc[down]=[sourcefile,Spectrum,"",intensity,mz]
                down+=1
            mgfdata.to_json(os.path.join(jsonfold,filename[0:-3]+"json"))

    datalis=[]
    for root,dirs,files in os.walk(jsonfold):
        for filename in files:

            datalis.append(pd.read_json(os.path.join(jsonfold,filename)))

    outdata=pd.concat(datalis,ignore_index=True)
    outdata.to_json(os.path.join(mgfdatafold,"SStruespectrum.json"))
# #################创建完毕之后merge
if args.do_scoreprediction:
    monomzname = args.inputfile
    finalname = args.outputfile
    df=pd.read_csv(monomzname)
    mgfdf=pd.read_json(os.path.join(mgfdatafold,"SStruespectrum.json"))
    try:
        df['Spectrum']=df['Spectrum'].apply(lambda x: x.split(":")[-1])
    except:
        0
    df['Spectrum']=df['Spectrum'].apply(lambda x: str(x))
    mgfdf['Spectrum']=mgfdf['Spectrum'].apply(lambda x: str(x))

    if mgfdf['SourceFile'][0].endswith("raw"):

        mgfdf["SourceFile"]=mgfdf['SourceFile'].apply(lambda x: x.split(".")[0])
    df['SourceFile']=df['SourceFile'].apply(lambda x: str(x))
    print("df:",df)
    print("mgfdf:",mgfdf)
    # df.to_csv("FDRmodelSS.csv")
    # mgfdf.to_json("SStruespectrum.json")
    # import ipdb
    # ipdb.set_trace()
    mgfdf['intensity']=mgfdf['intensity'].apply(lambda x: np.array(x)/max(x,default=1))

    print("read data ok")
    total=pd.merge(df,mgfdf,on=["Spectrum","SourceFile"],how="left")
    print(total)
    print("merge ok")
    # total.to_csv("total.csv")
    def putTintensity(row):
        mz=row["FragmentMz"]
        ppm=1/1000000

        mzlist=row['mz']
        i=(np.abs(np.array(mzlist)-mz)).argmin()
        if  abs(mzlist[i]-mz)<mz*args.ppm*ppm:
            return row['intensity'][i]
        else :
            return 0 #返回和mz最接近的mzlist里的值
    total['Tintensity']=total.apply(putTintensity,axis=1)
    # print("saving modelTT.........")
    # total.to_csv("modelTT.csv",index=False)
    ##############对每一个key匹配计算
    from torch.nn import CosineSimilarity
    import torch
    cos = CosineSimilarity(dim=0)
    # total=pd.read_json("modelTT.json")
    print("read data ok")
    kkdict={}
    for kk, kframe in total.groupby(["Spectrum","SourceFile"],as_index=False):
        diclist=list(set(list(kframe["FragmentMz"])))
        tq = [0] * len(diclist)
        for ind in kframe.index:
            zind = diclist.index(kframe.loc[ind, "FragmentMz"])
            tq[zind] = kframe.loc[ind, 'Tintensity']
        for iii,iframe in kframe.groupby("iii",as_index=False):
            modelq = [0] * len(diclist)

            for ind in iframe.index:
                zind=diclist.index(iframe.loc[ind,"FragmentMz"])

                modelq[zind] = iframe.loc[ind, 'FI.Intensity']
                sim = cos(torch.sqrt(torch.Tensor(modelq)), torch.sqrt(torch.Tensor(tq)))
                kkdict[iii] = sim
    total["score"]=total["iii"].apply(lambda x:kkdict[x])
    if args.save_score:
        total.to_csv("modelTTscore.csv",index=False)

    total=total.drop_duplicates(subset=["iii"],keep="first")
    print("saving......")
    total.to_csv(finalname,index=False)

    # total["score"]=total["iii"].apply(lambda x:kkdict[x])
    #
    # total=total.drop_duplicates(subset=["iii"],keep="first")
    # total.to_csv("finaldata.csv",index=False)
    # total=pd.read_csv("finaldata.csv")
    # total=total.drop_duplicates(subset=["iii"],keep="first")
    # total.to_csv("finaldata.csv",index=False)
    print(total)
