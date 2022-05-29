import numpy as np
import pandas as pd

import os

import re

#############



def countdecoration(presequence):
    ##example:  input:SDL1K2FJ4NL
    ##          output:00120400
    phos = []
    number = "0123456789"
    l = len(presequence)
    sig=0
    for i in range(l):
        if presequence[i]=='4':
            assert i==0
            sig=1
            phos.append(4)
            continue
        elif presequence[i] in number:
            phos[-1] = int(presequence[i])
            continue

        phos.append(0)
        if sig:
            sig=0
            phos.pop()
    return phos
def report2matrix(filename,outputname=None):
    fields = "sequence,decoration,charge,ions,psmscore,_psequence,irt,pnumber".split(',')
    fieldslen = range(len(fields))
    outputdataframe = pd.DataFrame(columns=fields)

    print(outputdataframe)
    if isinstance(filename,str):
        file = filename
        path0 = os.getcwd()
        filepath = os.path.join(path0, file)
        testdata = pd.read_csv(filepath)
    elif isinstance(filename,pd.DataFrame):
        testdata=filename
        file="rawdata.csv"
    else:
        raise Exception

    down = 0
    decorationdict={}
    decorationdict['[Phospho (STY)]']=1
    decorationdict['[Oxidation (M)]']=2
    decorationdict['[Carbamidomethyl (C)]'] = 3
    decorationdict['[Acetyl (Protein N-term)]'] = 4
    lossdict = {"noloss": '', "H2O": 'o', "NH3": 'n', "H3PO4": 'p', '1(+H2+O)1(+H3+O4+P)': 'op',
                '1(+H3+N)1(+H3+O4+P)': 'np',"2(+H3+O4+P)": '2p', '1(+H2+O)2(+H3+O4+P)': 'o2p',
                '1(+H3+N)2(+H3+O4+P)': 'n2p'}
    for charge, chargedata in testdata.groupby("PP.Charge"):
        for psequence, iframe in chargedata.groupby("PP.PIMID"):
            psmscore, frame = list(iframe.groupby("PSM.Score"))[-1]
            # for psmscore,frame in iframe.groupby("PSM.Score"):

            _psequence = psequence[1:-1]
            sequence = list(frame['PEP.StrippedSequence'])[0]

            #####decoration

            match = re.findall(r'\[.*?\]', _psequence)
            for _ in match:
                _psequence = _psequence.replace(_, str(decorationdict[_]))

            print(_psequence)

            decoration = countdecoration(_psequence)
            length = len(decoration)
            #######decoration///
            # charge = list(frame['PP.Charge'])[0]
            ###########ions
            ionname = "b1,bn1,bo1,b2,bn2,bo2,y1,yn1,yo1,y2,yn2,yo2,bp1,bnp1,bop1,bp2,bnp2,bop2,yp1,ynp1,yop1,yp2,ynp2,yop2," \
                      "b2p1,bn2p1,bo2p1,b2p2,bn2p2,bo2p2,y2p1,yn2p1,yo2p1,y2p2,yn2p2,yo2p2".split(
                ',')
            ions = np.zeros((length - 1, len(ionname)))
            pnumber=decoration.count(1)
            for index, row in frame.iterrows():
                try:
                    ion = row['FI.FrgType'] + lossdict[row['FI.LossType']] + str(row['FI.Charge'])


                    j = ionname.index(ion)

                except:
                    continue
                i = row['FI.FrgNum'] - 1 if row['FI.FrgType'] == 'b' else length - row['FI.FrgNum'] - 1
                # print(i,j)
                # print("len:",length)
                ions[i, j] = row['FI.Intensity']
            irt=list(frame["PP.iRTEmpirical"])[0]
            #############ions///
            outputdataframe.loc[down] = [sequence, decoration, charge, ions, psmscore, _psequence,irt,pnumber]
            down += 1
    if outputname:
        print("savingfilename:" + outputname)
        outputdataframe.to_json(outputname)
    else:
        print("savingfilename:",file[0:-4] + "processed.json")
        outputdataframe.to_json(file[0:-4] + "processed.json")
    # decorationvocab.save("decorationvocab")
    # print(decorationvocab)
    print("finish")
    # print(outputdataframe)

##############################################
def resultreport2matrix(filename,name=None):
    fields = "sequence,decoration,charge,ions,_psequence,irt".split(',')
    fieldslen = range(len(fields))
    outputdataframe = pd.DataFrame(columns=fields)

    print(outputdataframe)
    if isinstance(filename,str):
        file = filename
        path0 = os.getcwd()
        filepath = os.path.join(path0, file)
        testdata = pd.read_csv(filepath)
    elif isinstance(filename,pd.DataFrame):
        testdata=filename
        if name:
            file=name
        else:
            file="rawdata.csv"
    else:
        raise Exception
    down = 0
    decorationdict={}
    decorationdict['[Phospho (STY)]']=1
    decorationdict['[Oxidation (M)]']=2
    decorationdict['[Carbamidomethyl (C)]'] = 3
    decorationdict['[Acetyl (Protein N-term)]'] = 4
    lossdict = {"noloss": '', "H2O": 'o', "NH3": 'n', "H3PO4": 'p', '1(+H2+O)1(+H3+O4+P)': 'op',
                '1(+H3+N)1(+H3+O4+P)': 'np',"2(+H3+O4+P)": '2p', '1(+H2+O)2(+H3+O4+P)': 'o2p',
                '1(+H3+N)2(+H3+O4+P)': 'n2p'}
    for charge, chargedata in testdata.groupby("PrecursorCharge"):
        for psequence, iframe in chargedata.groupby("ModifiedPeptide"):
            frame=iframe
            # for psmscore,frame in iframe.groupby("PSM.Score"):

            _psequence = psequence[1:-1]
            sequence = list(frame['StrippedPeptide'])[0]

            #####decoration

            match = re.findall(r'\[.*?\]', _psequence)
            for _ in match:
                _psequence = _psequence.replace(_, str(decorationdict[_]))

            print(_psequence)

            decoration = countdecoration(_psequence)
            length = len(decoration)
            #######decoration///
            # charge = list(frame['PP.Charge'])[0]
            ###########ions
            ionname = "b1,bn1,bo1,b2,bn2,bo2,y1,yn1,yo1,y2,yn2,yo2,bp1,bnp1,bop1,bp2,bnp2,bop2,yp1,ynp1,yop1,yp2,ynp2,yop2," \
                      "b2p1,bn2p1,bo2p1,b2p2,bn2p2,bo2p2,y2p1,yn2p1,yo2p1,y2p2,yn2p2,yo2p2".split(
                ',')
            ions = np.zeros((length - 1, len(ionname)))

            for index, row in frame.iterrows():
                try:
                    ion = row['FragmentType'] + lossdict[row['FragmentLossType']] + str(row['FragmentCharge'])


                    j = ionname.index(ion)
                except:
                    continue
                i = row['FragmentNumber'] - 1 if row['FragmentType'] == 'b' else length - row['FragmentNumber'] - 1
                # print(i,j)
                # print("len:",length)
                ions[i, j] = row['RelativeIntensity']
            irt=list(frame["iRT"])[0]
            #############ions///
            outputdataframe.loc[down] = [sequence, decoration, charge, ions, _psequence,irt]
            down += 1
    outputdataframe.to_json(file[0:-4] + "processed.json")
    # decorationvocab.save("decorationvocab")
    # print(decorationvocab)
    print("finish")
    # print(outputdataframe)

##############################################
def report2irtmatrix(filename,name=None):#####因为preprocess需要ions故这里产生的数据ions都用1代替，是无意义的


    if isinstance(filename,str):
        file = filename
        path0 = os.getcwd()
        filepath = os.path.join(path0, file)
        testdata = pd.read_csv(filepath)
    elif isinstance(filename,pd.DataFrame):
        testdata=filename
        if name:
            file=name
        else:
            file="rawdata.csv"
    else:
        raise Exception
    down = 0
    decorationdict={}
    decorationdict['[Phospho (STY)]']=1
    decorationdict['[Oxidation (M)]']=2
    decorationdict['[Carbamidomethyl (C)]'] = 3
    decorationdict['[Acetyl (Protein N-term)]'] = 4
    outputdata=testdata.drop_duplicates(subset=["PP.iRTEmpirical"])
    zz = outputdata.loc[:, ["R.FileName", "PP.PIMID", 'PEP.StrippedSequence', "PP.Charge", "PP.iRTEmpirical", "PP.EmpiricalRT"]]

    newcolumns = ["filename", "PP.PIMID", "sequence", 'charge', 'irt0', 'rt0']
    zz.columns = newcolumns
    zz['decoration'] = ''

    def rowdecoration(row):
        psequence = row["PP.PIMID"]
        _psequence = psequence[1:-1]

        #####decoration

        match = re.findall(r'\[.*?\]', _psequence)
        for _ in match:
            _psequence = _psequence.replace(_, str(decorationdict[_]))
        decoration = countdecoration(_psequence)

        #######decoration///
        # charge = list(frame['PP.Charge'])[0]
        return decoration

    zz['decoration'] = zz.apply(lambda row: rowdecoration(row), axis=1)
    zz['ions']=[1]*len(zz)
    zz.to_json(file[0:-4]+ "processed_onlyirt.json")
    print("finish")
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--DDAfile",
        default=None,
        # default="./output_train/train_pred_gold_ner.json",
        type=str,
        required=True,
        help="DDA raw file",
    )
    parser.add_argument(
        "--outputfile",
        default=None,
        # default="./output_train/train_pred_gold_ner.json",
        type=str,
        required=False,
        help="output filename",
    )
    parser.add_argument(
        "--do_ms2",
        action="store_true", help="do ms2",

    )
    parser.add_argument(
        "--do_irt",
        action="store_true", help="do irt",

    )
    args = parser.parse_args()
    if args.do_ms2:
        if not args.outputfile:
            report2matrix(args.DDAfile,args.DDAfile[:-4]+".json")
        else:
            report2matrix(args.DDAfile,args.outputfile)
    if args.do_irt:
        report2irtmatrix(args.DDAfile)
if __name__=='__main__':
    main()
    # report2irtmatrix(target)