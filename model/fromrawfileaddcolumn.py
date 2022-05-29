import pandas as pd
# def addcolumntoresult(resultfilename, rawfilename, columnname="IonMobility"):
#         rawdata = pd.read_csv(rawfilename)
#         resultdata = pd.read_csv(resultfilename)
#         assert columnname in rawdata.columns, "columnname not found in rawdata"
#         if columnname in resultdata.columns:
#             print("warning: columnname already in result, will overwrite result data")


def addionmobility(resultfilename, rawfilename, columnname="IonMobility"):
    rawdata = pd.read_csv(rawfilename)
    resultdata = pd.read_csv(resultfilename)
    assert columnname in rawdata.columns, "columnname not found in rawdata"
    if columnname in resultdata.columns:
        print("warning: columnname already in result, will overwrite result data")
    resultdata[columnname]=[-65]*len(resultdata)
    _resultdata=resultdata.copy()
    _resultdata[columnname] = [-45] * len(resultdata)
    out=pd.concat([resultdata,_resultdata],axis=0)
    print(len(resultdata))
    print(len(out))
    out.to_csv("modeldata_ionmobility.csv",index=False)
    return
if __name__=="__main__":
    addionmobility("modelirtdatamonomz(1).csv","zhaodan_only_phos_DDA_exp(1).csv")