from model import *
from model import _2deepdiaModelms2
from fastNLP import Tester
from preprocess import *
from model import _2deepchargeModelms2_all
from Bertmodel import _2deepchargeModelms2_bert,_2deepchargeModelms2_roberta
import fastNLP
from transformers import BertConfig,RobertaConfig
####################Const
from utils import *
####################
set_seed(seed)
import argparse
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument(
    "--inputfile",
    default="./output_train/train_pred_gold_ner.json",
    type=str,
    required=True,
    help="DDA processed file,if using no_target ,use zz csv file",
)
parser.add_argument(
    "--outputfile",
    default="repsequence.json",
    type=str,
    # required=True,
    help="output filename",
)
parser.add_argument(
    "--do_decoy", action="store_true", help="if ReadMe ##5, do_decoy"
)
parser.add_argument(
    "--concat", action="store_true", help="use concat instead of merge"
)
parser.add_argument(
    "--only_combine", action="store_true", help="do combinerawresultonly.py"
)
parser.add_argument(
    "--no_target", action="store_true", help="using zz csv file ,no_target, using processed json file ,don't use"
)
parser.add_argument(
    "--cpu", action="store_true", help="if ,device=cpu"
)
parser.add_argument(
    "--DDArawfile",
    default=None,
    # default="./output_train/train_pred_gold_ner.json",
    type=str,
    # required=True,
    help="DDA raw filename",
)
args = parser.parse_args()

processeddata=args.inputfile
save_path=args.outputfile
class CossimilarityMetricforreplace(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self, pred=None, target=None, seq_len=None,num_col=None,sequence=None,charge=None,decoration=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len,sequence=sequence,charge=charge,decoration=decoration)
        self.bestcos=0
        self.total = 0
        self.cos = 0
        self.listcos=[]
        self.nj=0
        self.repsequence=pd.DataFrame(columns=['repsequence','charge','decoration','ms2'])
        self.numcol=num_col

    def evaluate(self, pred, target, seq_len=None,sequence=None,charge=None,decoration=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        N=pred.size(0)
        L=pred.size(1)
        # if seq_len is not None and target.dim() > 1:
        #     max_len = target.size(1)
        #     masks = seq_len_to_mask(seq_len=(seq_len-1)*int(self.num_col))
        # else:



        cos=CosineSimilarity(dim=1)
        s = torch.sum(cos(pred, target)).item()
        self.cos += s
        self.total += pred.size(0)
        self.bestcos = max(self.bestcos, torch.max(cos(pred, target)).item())
        self.listcos += cos(pred, target).reshape(N, ).cpu().numpy().tolist()
        for i in range(N):
            il = seq_len[i]
            isequence=sequence[i][:il]
            icharge=charge[i]
            idecoration=decoration[i][:il]

            ims2=pred[i].reshape((-1,self.numcol))[:il-1,:self.numcol]

            self.repsequence.loc[self.nj] = [isequence.cpu().numpy().tolist(),
                                         icharge.cpu().numpy().tolist(),
                                         idecoration.cpu().numpy().tolist(),

                                         ims2.cpu().numpy().tolist()]
            self.nj+=1


    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        data=pd.Series(self.listcos)
        data.to_csv("Cossimilaritylist.csv",index=False)
        self.repsequence.to_json(save_path)
        evaluate_result = {'mediancos':round(np.median(self.listcos),6),
                            'total number':self.nj,
                           'coss': round(float(self.cos) / (self.total + 1e-12), 6),
                           'bestcoss':round(self.bestcos, 6),
                           }
        if reset:
            self.cos = 0
            self.total = 0
            self.bestcos=0
            self.listcos=[]
        return evaluate_result

################data
##########################
if args.only_combine:
    # subprocess.call(
    #         "python combinerawresult.py --inputfile {} --repsequencefile {} ".format(processeddata,save_path)

    #         ,
    #         shell=True,
    #     )
        os.system(
            "python fromdiskcombine.py --inputfile {} --repsequencefile {} ".format(processeddata,save_path)


        )
##################
else:
    from datapath import *
    fpath=processeddata
    if args.no_target:
        databundle=PPeptidePipe_notarget(vocab=vocab).process_from_file(paths=fpath)
    else:
        databundle = PPeptidePipe(vocab=vocab).process_from_file(paths=fpath)
    totaldata=databundle.get_dataset("train")
    print(totaldata)
    ###########model

    config=BertConfig.from_pretrained("bert-base-uncased")
    bestmodelpath="phosT/best__2deepchargeModelms2_bert_mediancos_2021-09-20-01-17-50-729399"###########一直用的bestmodel,pretrainbert
    deepms2=_2deepchargeModelms2_bert(config)
    bestmodel=torch.load(bestmodelpath).state_dict()
    deepms2.load_state_dict(bestmodel)

    ###########Trainer



    from fastNLP import Const
    metrics=CossimilarityMetricforreplace(pred=Const.OUTPUT,target=Const.TARGET,seq_len='seq_len',
                                    num_col=num_col,sequence='sequence',charge="charge",decoration="decoration")
    from fastNLP import MSELoss
    loss=MSELoss(pred=Const.OUTPUT,target=Const.TARGET)

    ############tester

    
    pptester=Tester(model=deepms2,device=device,data=totaldata,
                    loss=loss,metrics=metrics,
                    batch_size=BATCH_SIZE)
                    
    from timeit import default_timer as timer
    train_time_start = timer()
    pptester.test()
    
    train_time_end = timer()
    def print_train_time(start: float, end: float, device: torch.device = None):
        """Prints difference between start and end time.

        Args:
            start (float): Start time of computation (preferred in timeit format). 
            end (float): End time of computation.
            device ([type], optional): Device that compute is running on. Defaults to None.

        Returns:
            float: time between start and end in seconds (higher is longer).
        """
        total_time = end - start
        print(f"Train time on {device}: {total_time:.3f} seconds")
        return total_time
    total_train_time_model_2 = print_train_time(start=train_time_start,
                                           end=train_time_end,
                                           device=device)
    if args.do_decoy:
        if args.concat:
            print("####################using concat instead of merge ########################")
            subprocess.call(
            "python firstdropandcombine_concat.py --inputfile {} --repsequencefile {} ".format(processeddata,save_path)

            ,
            shell=True,
        )
        else:
            subprocess.call(
            "python firstdropandcombine.py --inputfile {} --repsequencefile {} ".format(processeddata,save_path)

            ,
            shell=True,
        )



