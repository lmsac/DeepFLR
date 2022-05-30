# Introduction
Codes for DeepFLR([paper]()).
# Requirements
Model construction was performed using python (3.8.3, Anaconda distribution version 5.3.1, https://www.anaconda.com/) with the following packages: FastNLP (0.6.0), pytorch (1.8.1) and transformers (4.12.5). 

Data analysis for FLR estimation was performed using python (3.8.3) with the following packages: pandas (1.0.5) and numpy (1.18.5). 
# Usage
## Model training
For model training, see ReadMe.md in `model` folder.

## Data analysis
To procecss DeepFLR result 
`python DeepFLR_result_processing.py --inputfile input.csv --maxquantfile maxquant.csv --outputfile output.csv`





