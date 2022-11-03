# Readme
## Environments
Main packages are as follows.\
FastNLP                   0.6.0 \
pytorch                   1.8.1 \
transformers              4.12.5 

## 1. Train
Training data has been deposited to ProteomeXchange via the iProX partner repository with the dataset identifier PXD037580.

### 1.1 Train MS/MS Prediction
1. If you want to train your own model or finetune the model, you need `python matrixwithdict.py` to get the json file in the right format for training. Here the `finetunefile.csv` is in the form of spectromine. See the demo `finetune_demo.csv` in `demo_data`.

`python matrixwithdict.py 
--do_ms2 
--DDAfile finetunefile.csv 
--outputfile finetunefile.json`

2. You can set hyperparameters in `utils.py`.

3. To set the training and validation dataset, you can set the filename in `train.py `. Normally we split the total dataset to get training and validation datse. And then 
`python train.py `

## 2 Spectra Prediction for Decoys and Targets
Suppose we have sequences including decoy sequences in `input.csv`.
First we predict spectra:

`python replace.py 
--do_decoy 
--no_target 
--inputfile input.csv `

By default outputfile is `inputmodelmonomz.csv`.

## 3 Cosine Similarity Calculation
To get the cosine similarity between the predicted spectrum and the  experiemntal spectrum,do the following. The inputfile is`inputmodelmonomz.csv` ,the default outputfile is `decoyoutputscore.csv`. `mgffoldlocation` is the datafolder location of the 
mgf files. For the first time we add `--do_mgfprocess` to transform mgf to json files.

`python mgfprocess.py 
--do_mgfprocess 
--do_scoreprediction 
--inputfile inputmodelmonomz.csv 
--outputfile outputscore.csv 
--mgfdatafold mgffoldlocation`

If not the first time or you have the json files, you don't need `--do_mgfprocess` like below.

`python mgfprocess.py 
--do_scoreprediction 
--inputfile inputmodelmonomz.csv 
--outputfile outputscore.csv 
--mgfdatafold mgf`