import torch,os,json
import numpy as np
import pandas as pd
from tqdm import tqdm
from asset import *

def update_config(config):
    temp = {}       
    with open(config.config_file, 'r') as f:
        temp = json.load(f)  
    for key, value in temp.items():
        setattr(config, key, value) 
    return config

def check_config(config):
    assert config.vectorizer.lower() in preprocess_set, f'config error, {config.vectorizer} not supported! Supplant it with somewhat in `asset.py` preprocess_set'
    assert config.model.lower() in model_set, f'config error, {config.model} not supported! Supplant it with somewhat in `asset.py` model_set'
    assert config.loss_func.lower() in loss_set, f'config error, {config.loss_func} not supported! Supplant it with somewhat in `asset.py` loss_set'
    assert config.optimizer.lower() in optim_set, f'config error, {config.optimizer} not supported! Supplant it with somewhat in `asset.py` optim_set'
    assert config.activate_func.lower() in activate_set, f'config error, {config.activate_func} not supported! Supplant it with somewhat in `asset.py` activate_set'

def time_trans(time):
    h=time//3600
    min=(time-h*3600)//60
    s=time-h*3600-min*60
    return "{:.0f} h {:.0f} mins {:.1f}s".format(h,min,s)

def check_grad(model,input_shape:tuple):
    '''
    input_shape: (bs=1,channel,h,w)
    '''
    x=torch.rand(input_shape)
    for name,layer in model.named_children():
        x = layer(x)
        layer_std = x.std()
        print('{}, std:{:.4f}'.format(name,layer_std))
        if torch.isnan(layer_std): 
            print("output is nan in {}\nLayer Detail: {}".format(name,layer))
            break

def cls_stoi(cls_list):
    cls_set = np.array(list(set(cls_list)))
    mapped_cls = np.zeros_like(cls_set)
    for idx,cls in enumerate(cls_set):
        mapped_cls[np.where(cls==cls_set)[0][0]]=idx
    return pd.Series(mapped_cls)

def split_data(config,train_val_data):
    if ':' in config.train_plan: # divide val set from train set
        train_num, val_num = list(map(int, config.train_plan.split(':')))
        assert train_num+val_num==10, 'train_plan error, the number between `:` should sum to 10'
        total_chunk = 10
    else:
        try:
            fold = int(config.train_plan.split('-')[0])
            train_num, val_num = [], []
            for val_idx in range(fold):
                val_num.append(val_idx)
                train_num.append([slice(0,val_idx),slice(val_idx+1,fold)])
            total_chunk = fold
        except:
            raise ValueError(f'train_plan error, correct form is `k-fold` or `a:b`')
    
    train_val_num = len(train_val_data)
    config.cls_list = list(set(train_val_data['Category']))
    rows_per_slice,remaining_rows = divmod(train_val_num, total_chunk)

    slices = []
    start = 0
    for i in range(total_chunk):
        end = start + rows_per_slice
        if i == total_chunk-1 and remaining_rows > 0:
            end += remaining_rows
        slices.append(slice(start, end))
        start = end

    train_val_sections = [train_val_data.iloc[slc] for slc in slices]
    return train_num, val_num, train_val_sections

def discretize_logit(config, logit, test=True):
    if config.model.startswith('logistic'):
        if config.loss_func != 'bce':
            logit = 5*torch.sigmoid(logit)
        elif test:
            logit = torch.sigmoid(logit) if logit.dtype != torch.int else logit

    elif config.model.startswith('softmax'):
        if config.loss_func != 'ce':
            logit = 5*torch.softmax(logit,-1)
        elif test:
            logit = torch.softmax(logit,-1)
    return logit

def cal_acc(config,dataloader,model):
    model.eval()
    iter_num=len(dataloader)
    top1_iter_acc=torch.zeros(iter_num)
    top5_iter_acc=torch.zeros(iter_num)

    with torch.no_grad():
        for iter,(datas,labels) in tqdm(enumerate(dataloader),desc='Eval'):
            datas,labels = datas.to(config.device), labels.to(config.device)
            logit = model(datas)  # 16,600
            logit = discretize_logit(config,logit) 
            if logit.shape[-1]==1:
                predict_labels = torch.round(logit)
            else:
                predict_labels=torch.argsort(logit,dim=1,descending=True)
            if labels.dim() > 1:
                labels = torch.where(labels==1)[1]
            top1_iter_acc[iter]=sum(predict_labels[:,0]==labels)/len(labels)
            top5_iter_acc[iter]=sum(list(
                                    map(lambda true_label,pred_label: true_label in pred_label,
                                        labels,predict_labels[:,:5]
                                        )))/len(labels)
    top1_acc=top1_iter_acc.mean().item()
    top5_acc=top5_iter_acc.mean().item()
    return top1_acc,top5_acc

def predict(config, dataloader, model):
    model.eval()
    predict_labels, data_idxs = [], []
    with torch.no_grad():
        for datas, data_idx in tqdm(dataloader,desc='Predict'):
            datas = datas.to(config.device)
            logit = model(datas)  # 16,600
            predict_labels.extend(torch.argmax(logit,dim=1).detach().cpu().tolist())
            data_idxs.extend(data_idx.tolist())
    res = pd.DataFrame({'Id':data_idxs,'Category':predict_labels})
    res['Category'] = res['Category'].apply(lambda x:config.cls_list[x])
    res.to_csv(os.path.join(config.save_dir,'submission.csv'),index=False)


