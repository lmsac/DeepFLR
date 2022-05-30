
# Readme
##  Train
Training data path: `Tprocessed2.json`.
### Train MS/MS
1. Use `matrixwithdict.py` (func:report2matrix(path)) to get target json file.

`python matrixwithdict.py 
--do_ms2 
--DDAfile input.csv 
--outputfile output.json`

2. Use funcs in `mix_jsons.py` to mix or split json files

3. Set hyperparameters in `utils.py`.

4. change the fpaths in `train.py` to training dataset path and `python train.py`


## Test
1. Use `matrixwithdict.py` (func:report2irtmatrix(path)) to get target json file.

`python matrixwithdict.py 
--do_ms2 
--DDAfile fintune/mouse_ear_cress_fintune/mouse_ear_cress_fintune_test_dataset.csv 
--outputfile msmstest/human_monophos.json`

2. `python test.py` or `testirt.py`

## Spectral graph to sequence decoy FDR
`python replace.py 
--do_decoy 
--no_target 
--inputfile input.csv` 

outputfile is `input`+`modelmonomz.csv`
If having the `--only_combine` we do only `combinerawresult.py`
to get final score ,do the following, inputfile is the outputfile above 

`python mgfprocess.py 
--do_mgfprocess 
--do_scoreprediction 
--inputfile mann_exchange_model_sequence_1_task1modeldatamonomz.csv 
--outputfile mann_task1.csv 
--mgfdatafold mann_mgf`

if don't need mgfprocess 

`python mgfprocess.py 
--do_scoreprediction 
--inputfile mann_exchange_model_sequence_1_task1modelmonomz.csv 
--outputfile mann_task1.csv 
--mgfdatafold mann_mgf` 

the default outputfile is `decoyoutputscore.csv`

## pdeep2test
`python matrixwithdict.py 
    --do_ms2 
    --DDAfile zzz.csv 
    --outputfile yyyy.json`

The outputfile is the DDAprocessedfile below

`python pdeep2test.py 
    --pdeep2jsonfile xxxx.json 
    --DDAprocessedfile yyyy.json` 

output score filename is `pdeep2testresult.csv`

## CosScoring for models
`python matrixwithdict.py 
    --do_ms2 
    --DDAfile zzz1.csv 
    --outputfile yyyy1.json`

`python matrixwithdict.py 
    --do_ms2 
    --DDAfile zzz2.csv 
    --outputfile yyyy2.json`

The outputfile is `zzz1.csv` is the data predicted from model and `zzz2.csv` is real data. ALL in spectromine style.

`python CosScoring.py 
    --scoringfile yyyy1.json 
    --DDAprocessedfile yyyy2.json` 

output score filename is `CosScoringtestresult.csv`