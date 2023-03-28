# DeepFLR
DeepFLR is a deep learning-based framework that accurately predicts phosphopeptides spectra and effectively controls false localization rates in phosphoproteomics.

# Requirements
Model construction was performed using python (3.8.3, Anaconda distribution version 5.3.1, https://www.anaconda.com/) with the following packages: FastNLP (0.6.0), pytorch (1.8.1), transformers (4.12.5), bidict (0.22.0) and pyteomics (4.5.5).

Data analysis for FLR estimation was performed using python (3.8.3) with the following packages: pandas (1.0.5) and numpy (1.18.5).

Users can install packages using the command provided in the User Guide, or use the “pip install -r requirements.txt” command to install all the required packages.


# Tutorial
Tutorials are avaliable in [User guide.docx](https://github.com/yuz2011/DeepFLR/blob/main/User%20guide.docx).
# Model
Users can either use the model parameters used in DeepFLR or finetune the model or retrain a model following the  [User guide.docx](https://github.com/yuz2011/DeepFLR/blob/main/User%20guide.docx). The Model parameters "best__2deepchargeModelms2_bert_mediancos_2021-09-20-01-17-50-729399" (329 MB) can be downloaded from [Model](https://github.com/yuz2011/DeepFLR/releases/tag/Model) and put it in `phosT` folder.
# Publications
Waiting for publication...([paper]())


# License
DeepFLR is distributed under a BSD license. See the LICENSE file for details.

# Contacts
Please report any problems directly to the github issue tracker. Also, you can send feedback to liang_qiao@fudan.edu.cn.
