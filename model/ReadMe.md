#Readme
##1. Train
Training data:/remote-home/yxwang/Finalcut/AllphosT/Tprocessed2.json.
###1 Train MS/MS
1.python matrixwithdict.py (func:report2matrix(path)) to get target json file.
python matrixwithdict.py \
--do_ms2 \
--DDAfile fintune/PXD14525_fintune.csv \
--outputfile fintune/PXD14525_fintune.json

2.use funcs in mix_jsons.py to mix or split json files

3.set attributes in utils.py.

4 python train.py 


##2. Test
1.python matrixwithdict.py (func:report2irtmatrix(path)) to get target json file.
python matrixwithdict.py \
--do_ms2 \
--DDAfile fintune/mouse_ear_cress_fintune/mouse_ear_cress_fintune_test_dataset.csv \
--outputfile msmstest/human_monophos.json

2.python test.py or testirt.py

##3 spectral graph to sequence decoy FDR
python replace.py \
--do_decoy \
--no_target \
--inputfile mann_exchange_model_sequence_1_task1.csv 

outputfile is xxx+modelmonomz.csv
If having the --only_combine do only combinerawresult.py
to get final score ,do the following, inputfile is the outputfile above 

python mgfprocess.py \
--do_mgfprocess \
--do_scoreprediction \
--inputfile mann_exchange_model_sequence_1_task1modeldatamonomz.csv \
--outputfile mann_task1.csv \
--mgfdatafold mann_mgf

if don't need mgfprocess 

python mgfprocess.py \
--do_scoreprediction \
--inputfile mann_exchange_model_sequence_1_task1modelmonomz.csv \
--outputfile mann_task1.csv \
--mgfdatafold mann_mgf 

the default outputfile is decoyoutputscore

##4 pdeep2test
python matrixwithdict.py \
    --do_ms2 \
    --DDAfile zzz.csv \
    --outputfile yyyy.json

The outputfile is the DDAprocessedfile below

python pdeep2test.py \
    --pdeep2jsonfile xxxx.json \
    --DDAprocessedfile yyyy.json 

output score filename is pdeep2testresult.csv

##5 CosScoring for models
python matrixwithdict.py \
    --do_ms2 \
    --DDAfile zzz1.csv \
    --outputfile yyyy1.json

python matrixwithdict.py \
    --do_ms2 \
    --DDAfile zzz2.csv \
    --outputfile yyyy2.json

The outputfile is zzz1.csv is the data predicted from model and zzz2.csv is real data. ALL in spectromine style.

python CosScoring.py \
    --scoringfile yyyy1.json \ 
    --DDAprocessedfile yyyy2.json 

output score filename is CosScoringtestresult.csv