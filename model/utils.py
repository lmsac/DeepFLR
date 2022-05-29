import fastNLP
import wandb
maxlength=60
maxlength_hela1charge3=49#最长序列长度
maxlength_hela2charge3=44
maxlength_phos2charge2=33
maxlength_phos2charge3=46
acid_size=22#氨基酸种类数(包含unk和pad)
finetune_epoch=10
N_epochs=100
vocab_save=False
vocab=fastNLP.Vocabulary().load("phosT/phosT_vocab")
BATCH_SIZE=128
device="cuda:0"
num_col=36
seed=101
dropout=0.2
lr=2e-5
embed_size=512
nhead=8
num_layers=6
warmupsteps=3000


class WandbCallback(fastNLP.Callback):
    r"""
    
    """
    def __init__(self,project,name,config:dict):
        r"""
        
        :param str name: project名字
        :param dict config: 模型超参
        :
        """
        super().__init__()
        self.project = project
        self.name=name
        self.config=config
    def on_train_begin(self):
        wandb.init(
      # Set entity to specify your username or team name
      # ex: entity="carey",
      # Set the project where this run will be logged
      project=self.project,
      name=self.name, 
      # Track hyperparameters and run metadata
      config=self.config)
    def on_train_end(self):
        wandb.finish()
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        wandb.log(eval_result)
    def on_backward_begin(self,loss):
        wandb.log({"loss":loss})