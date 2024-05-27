import torch
import torch.nn as nn

class Logistic_OVO(nn.Module):
    def __init__(self,config):
        super(Logistic_OVO,self).__init__()
        self.config = config

        cls_num = len(config.cls_list)
        self.bk = nn.ModuleList([nn.Linear(config.input_dim,1) for _ in range(cls_num*(cls_num-1)//2)])
    def forward(self, x):
        pred_label=[]
        cls_num = len(self.config.cls_list)
        dual_cls_list= torch.tensor(self.config.dual_cls_list)
        for id,classifier in enumerate(self.bk):
            binary_label=torch.round(torch.sigmoid(classifier(x).squeeze())).long() 
            pred_label.append( dual_cls_list[id][binary_label] ) 
        pred_label=torch.stack(pred_label,1) # bs,45
        
        cls_vote=torch.zeros(pred_label.shape[0],cls_num).to(self.config.device) 
        for i in range(cls_num):
            cls_vote[:,i]=torch.sum(pred_label==i,1)
        
        return cls_vote.int()

class SoftmaxRegression(nn.Module):
    def __init__(self,config):
        super(SoftmaxRegression,self).__init__()
        assert 'hidden_units' in config, 'hidden_units is not in config'

        self.config = config

        input_dim = config.input_dim
        hidden_layer_num = len(config.hidden_units)
        cls_num = len(config.cls_list)
        units=[input_dim]+config.hidden_units
        
        module_list=[]
        for i in range(hidden_layer_num): 
            module_list.append(nn.Linear(units[i],units[i+1]))
            module_list.append(config.activate_func)
        
        self.bk=nn.Sequential(
            *module_list,
            nn.Dropout(p=config.dropout_rate) if config.dropout_rate>0 or 'dropout_rate' in config else nn.Identity(),
            nn.Linear(units[-1],cls_num)
        )

    def forward(self,x):
        return self.bk(x)

class RNN(nn.Module):
    def __init__(self,config):
        super(RNN,self).__init__()
        self.bk = nn.RNN()
