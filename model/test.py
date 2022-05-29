from model import *
from model import _2deepdiaModelms2
from fastNLP import Tester
from model import _2deepchargeModelms2
from model import _2deepchargeModelms2_all
import fastNLP
from Bertmodel import _2deepchargeModelms2_bert,_2deepchargeModelms2_roberta
import fastNLP
from transformers import BertConfig,RobertaConfig
####################Const
from utils import *
####################
set_seed(seed)
################data

from datapath import *
fold=False
if fold==True:
    datafold="fintune/DeepFLR/mono_multiphos"
    for data in os.listdir(datafold):
        if data.endswith("json"):
            testdata=os.path.join(datafold,data)
        else:
            continue
        print("testdata: "+testdata)
        fpath=testdata
        databundle=PPeptidePipe(vocab=vocab).process_from_file(paths=fpath)
        totaldata=databundle.get_dataset("train")
        # traindata,devdata=totaldata.split(0.10)
        # devphosdata=devdata.drop(lambda ins:ins["decoration"].count(1)<1,inplace=False)
        # devphos1data=devphosdata.drop(lambda ins:ins["decoration"].count(1)>1,inplace=False)
        # devphosmultidata=devphosdata.drop(lambda ins:ins["decoration"].count(1)<2,inplace=False)
        # print(devdata)
        ###########model
        config=BertConfig.from_pretrained("bert-base-uncased")
        bestmodelpath="/remote-home/yxwang/Finalcut/checkpoints/pretrainedbertbaseconfig_trainall/best__2deepchargeModelms2_bert_mediancos_2021-09-20-01-17-50-729399"###########一直用的bestmodel,pretrainbert
        # bestmodelpath="/remote-home/yxwang/Finalcut/checkpoints/bert-base-uncased/pretrained_trainall_ss/furthermann/2021-09-30-13-31-45-442035/epoch-9_step-4275_mediancos-0.955552.pt"###mannfurther
        # bestmodelpath="m /remote-home/yxwang/Finalcut/checkpoints/bert-base-uncased/pretrained_trainall_ss/furthermann_all/best__2deepchargeModelms2_bert_mediancos_2021-10-09-10-58-15-235402"#####这行改
        deepms2=_2deepchargeModelms2_bert(config)
        bestmodel=torch.load(bestmodelpath).state_dict()
        deepms2.load_state_dict(bestmodel)

        ###########Trainer



        from fastNLP import Const
        metrics=CossimilarityMetricfortest(savename=testdata,pred=Const.OUTPUT,target=Const.TARGET,seq_len='seq_len',
                                        num_col=num_col,sequence='sequence',charge="charge",decoration="decoration")
        from fastNLP import MSELoss
        loss=MSELoss(pred=Const.OUTPUT,target=Const.TARGET)

        ############tester


        pptester=Tester(model=deepms2,device=device,data=totaldata,
                        loss=loss,metrics=metrics,
                        batch_size=BATCH_SIZE)
        pptester.test()


else:
    fpath="/remote-home/yxwang/Finalcut/fintune/DeepFLR/fintune/ear_cress_fintune/Model_ear_mouse_cress_fintune_test_dataset.json"
    databundle=PPeptidePipe(vocab=vocab).process_from_file(paths=fpath)
    totaldata=databundle.get_dataset("train")
    traindata,devdata=totaldata.split(0.10)
    # devphosdata=devdata.drop(lambda ins:ins["decoration"].count(1)<1,inplace=False)
    # devphos1data=devphosdata.drop(lambda ins:ins["decoration"].count(1)>1,inplace=False)
    # devphosmultidata=devphosdata.drop(lambda ins:ins["decoration"].count(1)<2,inplace=False)
    print(devdata)
    ###########model
    config=BertConfig.from_pretrained("bert-base-uncased")
    bestmodelpath="/remote-home/yxwang/Finalcut/checkpoints/pretrainedbertbaseconfig_trainall/best__2deepchargeModelms2_bert_mediancos_2021-09-20-01-17-50-729399"###########一直用的bestmodel,pretrainbert
    # bestmodelpath="/remote-home/yxwang/Finalcut/checkpoints/bert-base-uncased/pretrained_trainall_ss/furthermann/2021-09-30-13-31-45-442035/epoch-9_step-4275_mediancos-0.955552.pt"###mannfurther
    bestmodelpath="/remote-home/yxwang/Finalcut/fintune/DeepFLR/fintune/ear_cress_fintune/Model_ear_mouse_cress_fintune_training_dataset/checkpoints/best__2deepchargeModelms2_bert_mediancos_2021-12-30-11-43-31-127318"#####这行改
    # bestmodelpath="/remote-home/yxwang/Finalcut/fintune/MSMS——fintune/mouse_fintune_2/mouse_fintune_training_datase/checkpoints/best__2deepchargeModelms2_bert_mediancos_2021-12-05-02-56-34-274010"
    deepms2=_2deepchargeModelms2_bert(config)
    bestmodel=torch.load(bestmodelpath).state_dict()
    deepms2.load_state_dict(bestmodel)

    ###########Trainer



    from fastNLP import Const
    metrics=CossimilarityMetricfortest(savename=fpath[:-5],pred=Const.OUTPUT,target=Const.TARGET,seq_len='seq_len',
                                    num_col=num_col,sequence='sequence',charge="charge",decoration="decoration")
    from fastNLP import MSELoss
    loss=MSELoss(pred=Const.OUTPUT,target=Const.TARGET)

    ############tester


    pptester=Tester(model=deepms2,device=device,data=totaldata,
                    loss=loss,metrics=metrics,
                    batch_size=1)
    pptester.test()