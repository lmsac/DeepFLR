import fastNLP
import torch
maxlength=60
maxlength_hela1charge3=49
maxlength_hela2charge3=44
maxlength_phos2charge2=33
maxlength_phos2charge3=46
acid_size=22
finetune_epoch=10
N_epochs=100
vocab_save=False
vocab=fastNLP.Vocabulary().load("phosT/phosT_vocab")
BATCH_SIZE=128
device="cuda:0" if torch.cuda.is_available () else "cpu"
num_col=36
seed=101
dropout=0.2
lr=2e-5
embed_size=512
nhead=8
num_layers=6
warmupsteps=3000