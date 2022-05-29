from transformers.models.bert.modeling_bert import (BertEmbeddings, BertModel,
BertForSequenceClassification,BertPreTrainedModel,BertEncoder,BertPooler)
from transformers.models.roberta.modeling_roberta import RobertaModel,RobertaEncoder
import torch.nn as nn
import torch
import torch.nn.functional as F
from fastNLP.core.metrics import MetricBase,seq_len_to_mask
class Acid_BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.charge_embedding=nn.Embedding(10,config.hidden_size,padding_idx=0)
        self.a_embedding=nn.Embedding(30,config.hidden_size,padding_idx=0)
        self.phos_embedding=nn.Embedding(10,config.hidden_size)#修饰三种加上padding###全部调大了
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))


    def forward(
        self, peptide_tokens=None, position_ids=None,decoration=None,charge=None, inputs_embeds=None, past_key_values_length=0
    ):
        N=peptide_tokens.size(0)
        L=peptide_tokens.size(1)
        sequence=peptide_tokens
        charge_embed=self.charge_embedding(charge.unsqueeze(1).expand(N,L))

        assert sequence.size(0) == decoration.size(0)
        phos_embed = self.phos_embedding(decoration)


        if peptide_tokens is not None:
            input_shape = peptide_tokens.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        

        inputs_embeds = self.a_embedding(peptide_tokens)


        embeddings = inputs_embeds+phos_embed+charge_embed
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
class _2deepchargeModelms2_bert(BertModel):#input:sequence:N*L(N:batch)##########bertmodel
    #####输入是cls A sep B sep C ....所以长度为2L，最后取所有的sep出来预测因为pretrain的原因只能写死在这里面numcol
    def __init__(self,config):
        super().__init__(config)
        self.config = config

        self.embeddings = Acid_BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation=nn.GELU()
        self.num_col=36
        self.mslinear=nn.Linear(config.hidden_size,36)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self,input_ids,peptide_tokens,peptide_length,charge,decoration,decoration_ids,pnumber,return_dict=None,head_mask=None):
        #input:input_ids:N*2L(N:batch)
        # length:N*1(N:batch)
        # phos:N*2L(N:batch)
        #charge:N*1
        
        # ninput=self.dropout(ninput)
        key_padding_mask=seq_len_to_mask(peptide_length*2)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device 
        past_key_values_length = 0
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(key_padding_mask, input_shape, device)
        embedding_output = self.embeddings(peptide_tokens=input_ids, 
        position_ids=None,
        decoration=decoration_ids,charge=charge, inputs_embeds=None, past_key_values_length=0

        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,

            use_cache=False,

            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]#####N*2L*E
        E=sequence_output.size(-1)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None####cls output可用来预测irt的
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1

        output=sequence_output[:,2:-1:2,:]
        assert output.size(1)==int(seq_length/2-1)
        
        outputms=self.dropout(self.mslinear(output))#N*(L-1)*num_col

        outputms=self.activation(outputms)

        outputms=outputms.reshape(batch_size,-1)#N*((L-1)*num_col)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(outputms)
        # print(torch.sum(outputms))
        return {'pred':outputms,'sequence':peptide_tokens,'charge':charge,"decoration":decoration,"seq_len":peptide_length,
                'pnumber':pnumber}
class _2deepchargeModelms2_bert_irt(BertModel):#input:sequence:N*L(N:batch)##########bertmodel
    #####输入是cls A sep B sep C ....所以长度为2L，最后取所有的sep出来预测因为pretrain的原因只能写死在这里面numcol
    def __init__(self,config):
        super().__init__(config)
        self.config = config

        self.embeddings = Acid_BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation=nn.GELU()
        self.num_col=36
        self.mslinear=nn.Linear(config.hidden_size,36)
        self.pooler = BertPooler(config)
        self.irtlinear=nn.Linear(config.hidden_size,1)
        self.init_weights()

    def forward(self,input_ids,peptide_tokens,peptide_length,charge,decoration,decoration_ids,pnumber,return_dict=None,head_mask=None):
        #input:input_ids:N*2L(N:batch)
        # length:N*1(N:batch)
        # phos:N*2L(N:batch)
        #charge:N*1
        
        # ninput=self.dropout(ninput)
        key_padding_mask=seq_len_to_mask(peptide_length*2)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device 
        past_key_values_length = 0
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(key_padding_mask, input_shape, device)
        embedding_output = self.embeddings(peptide_tokens=input_ids, 
        position_ids=None,
        decoration=decoration_ids,charge=charge, inputs_embeds=None, past_key_values_length=0

        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,

            use_cache=False,

            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]#####N*2L*E
        E=sequence_output.size(-1)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None####cls output可用来预测irt的
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1
        predirt=self.irtlinear(pooled_output).squeeze()##
        output=sequence_output[:,2:-1:2,:]
        assert output.size(1)==int(seq_length/2-1)
        
        outputms=self.dropout(self.mslinear(output))#N*(L-1)*num_col

        outputms=self.activation(outputms)

        outputms=outputms.reshape(batch_size,-1)#N*((L-1)*num_col)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(outputms)
        # print(torch.sum(outputms))
        return {'pred':outputms,'sequence':peptide_tokens,'charge':charge,"decoration":decoration,"seq_len":peptide_length,
                'pnumber':pnumber,"predirt":predirt}
class _2deepchargeModelms2_roberta(RobertaModel):#input:sequence:N*L(N:batch)##########bertmodel
    #####输入是cls A sep B sep C ....所以长度为2L，最后取所有的sep出来预测因为pretrain的原因只能写死在这里面numcol
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self,config):
        super().__init__(config)
        self.config = config

        self.embeddings = Acid_BertEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation=nn.GELU()
        self.num_col=36
        self.mslinear=nn.Linear(config.hidden_size,36)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self,input_ids,peptide_tokens,peptide_length,charge,decoration,decoration_ids,pnumber,return_dict=None,head_mask=None):
        #input:input_ids:N*2L(N:batch)
        # length:N*1(N:batch)
        # phos:N*2L(N:batch)
        #charge:N*1
        
        # ninput=self.dropout(ninput)
        key_padding_mask=seq_len_to_mask(peptide_length*2)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device 
        past_key_values_length = 0
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(key_padding_mask, input_shape, device)
        embedding_output = self.embeddings(peptide_tokens=input_ids, 
        position_ids=None,
        decoration=decoration_ids,charge=charge, inputs_embeds=None, past_key_values_length=0

        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,

            use_cache=False,

            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]#####N*2L*E
        E=sequence_output.size(-1)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None####cls output可用来预测irt的
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1
        
        output=sequence_output[:,2:-1:2,:]
        assert output.size(1)==int(seq_length/2-1)
        
        outputms=self.dropout(self.mslinear(output))#N*(L-1)*num_col

        outputms=self.activation(outputms)

        outputms=outputms.reshape(batch_size,-1)#N*((L-1)*num_col)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(outputms)
        # print(torch.sum(outputms))
        return {'pred':outputms,'sequence':peptide_tokens,'charge':charge,"decoration":decoration,"seq_len":peptide_length,
                'pnumber':pnumber}
class _2deepchargeModelms2_bert_ss(BertModel):#input:sequence:N*L(N:batch)##########bertmodel
    #####输入是cls A sep B sep C ....所以长度为2L，最后取所有的sep出来预测因为pretrain的原因只能写死在这里面numcol
    def __init__(self,config):
        super().__init__(config)
        self.config = config

        self.embeddings = Acid_BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation=nn.GELU()
        self.num_col=36
        self.sslinear=nn.Linear(config.hidden_size,config.hidden_size)
        self.mslinear=nn.Linear(config.hidden_size,36)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self,input_ids,peptide_tokens,peptide_length,charge,decoration,decoration_ids,pnumber,return_dict=None,head_mask=None):
        #input:input_ids:N*2L(N:batch)
        # length:N*1(N:batch)
        # phos:N*2L(N:batch)
        #charge:N*1
        
        # ninput=self.dropout(ninput)
        key_padding_mask=seq_len_to_mask(peptide_length*2)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device 
        past_key_values_length = 0
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(key_padding_mask, input_shape, device)
        embedding_output = self.embeddings(peptide_tokens=input_ids, 
        position_ids=None,
        decoration=decoration_ids,charge=charge, inputs_embeds=None, past_key_values_length=0

        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,

            use_cache=False,

            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]#####N*2L*E
        E=sequence_output.size(-1)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None####cls output可用来预测irt的
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1
        
        output=sequence_output[:,2:-1:2,:]
        assert output.size(1)==int(seq_length/2-1)
        outputss=self.sslinear(output)
        outputms=self.dropout(self.mslinear(outputss))#N*(L-1)*num_col

        outputms=self.activation(outputms)

        outputms=outputms.reshape(batch_size,-1)#N*((L-1)*num_col)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(outputms)
        # print(torch.sum(outputms))
        return {'pred':outputms,'sequence':peptide_tokens,'charge':charge,"decoration":decoration,"seq_len":peptide_length,
                'pnumber':pnumber}
class _2deepchargeModelms2_bert_ss_contrast(BertModel):#input:sequence:N*L(N:batch)##########bertmodel
    #####输入是cls A sep B sep C ....所以长度为2L，最后取所有的sep出来预测因为pretrain的原因只能写死在这里面numcol
    def __init__(self,config):
        super().__init__(config)
        self.config = config

        self.embeddings = Acid_BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation=nn.GELU()
        self.num_col=36
        self.sslinear=nn.Linear(config.hidden_size,config.hidden_size)
        self.mslinear=nn.Linear(config.hidden_size,36)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self,input_ids,peptide_tokens,peptide_length,charge,decoration,decoration_ids,pnumber,return_dict=None,head_mask=None):
        #input:input_ids:N*2L(N:batch)
        # length:N*1(N:batch)
        # phos:N*2L(N:batch)
        #charge:N*1
        
        # ninput=self.dropout(ninput)
        key_padding_mask=seq_len_to_mask(peptide_length*2)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device 
        past_key_values_length = 0
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(key_padding_mask, input_shape, device)
        embedding_output = self.embeddings(peptide_tokens=input_ids, 
        position_ids=None,
        decoration=decoration_ids,charge=charge, inputs_embeds=None, past_key_values_length=0

        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,

            use_cache=False,

            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]#####N*2L*E
        E=sequence_output.size(-1)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None####cls output可用来预测irt的
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1
        
        output=sequence_output[:,2:-1:2,:]
        assert output.size(1)==int(seq_length/2-1)
        ############no_grad_no_linear_contrast, simsiam
        encoder_outputs_ss = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,

            use_cache=False,

            return_dict=return_dict,
        )
        sequence_output_ss=encoder_outputs_ss[0]
        output_nograd=sequence_output_ss[:,2:-1:2,:].detach()
        outputss=self.sslinear(output)
        ####对比outputss与output_nograd.size:B*(L-1)*E
        contrastmask=seq_len_to_mask(seq_len=(peptide_length - 1) *E )
        outputss_masked=outputss.reshape(batch_size,-1).masked_fill(contrastmask.eq(False), 0)
        output_nograd_masked=output_nograd.reshape(batch_size,-1).masked_fill(contrastmask.eq(False), 0)###B*((L-1)*E)
        p=F.normalize(outputss_masked,p=2,dim=1)
        z=F.normalize(output_nograd_masked,p=2,dim=1)
        sscontrastloss=-(p*z).sum(dim=1).mean()
        
        
        outputms=self.dropout(self.mslinear(outputss))#N*(L-1)*num_col

        outputms=self.activation(outputms)

        outputms=outputms.reshape(batch_size,-1)#N*((L-1)*num_col)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(outputms)
        # print(torch.sum(outputms))
        return {"sscontrastloss":sscontrastloss,'pred':outputms,'sequence':peptide_tokens,'charge':charge,"decoration":decoration,"seq_len":peptide_length,
                'pnumber':pnumber}


class _2deepchargeModelms2_roberta_ss(RobertaModel):#input:sequence:N*L(N:batch)##########bertmodel
    #####输入是cls A sep B sep C ....所以长度为2L，最后取所有的sep出来预测因为pretrain的原因只能写死在这里面numcol
    def __init__(self,config):
        super().__init__(config)
        self.config = config

        self.embeddings = Acid_BertEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation=nn.GELU()
        self.num_col=36
        self.sslinear=nn.Linear(config.hidden_size,config.hidden_size)
        self.mslinear=nn.Linear(config.hidden_size,36)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self,input_ids,peptide_tokens,peptide_length,charge,decoration,decoration_ids,pnumber,return_dict=None,head_mask=None):
        #input:input_ids:N*2L(N:batch)
        # length:N*1(N:batch)
        # phos:N*2L(N:batch)
        #charge:N*1
        
        # ninput=self.dropout(ninput)
        key_padding_mask=seq_len_to_mask(peptide_length*2)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device 
        past_key_values_length = 0
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(key_padding_mask, input_shape, device)
        embedding_output = self.embeddings(peptide_tokens=input_ids, 
        position_ids=None,
        decoration=decoration_ids,charge=charge, inputs_embeds=None, past_key_values_length=0

        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,

            use_cache=False,

            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]#####N*2L*E
        E=sequence_output.size(-1)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None####cls output可用来预测irt的
        # outputmean=output.mean(dim=0)#N*E
        # outputrt=self.activation(self.rtlinear(outputmean))#N*1
        
        output=sequence_output[:,2:-1:2,:]
        assert output.size(1)==int(seq_length/2-1)
        outputss=self.sslinear(output)
        outputms=self.dropout(self.mslinear(outputss))#N*(L-1)*num_col

        outputms=self.activation(outputms)

        outputms=outputms.reshape(batch_size,-1)#N*((L-1)*num_col)

        masks = seq_len_to_mask(seq_len=(peptide_length - 1) * self.num_col)##加上mask
        outputms=outputms.masked_fill(masks.eq(False), 0)

        # print(outputms)
        # print(torch.sum(outputms))
        return {'pred':outputms,'sequence':peptide_tokens,'charge':charge,"decoration":decoration,"seq_len":peptide_length,
                'pnumber':pnumber}