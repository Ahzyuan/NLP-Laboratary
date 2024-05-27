import argparse,os,json,torch,sys,math
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.distributed as dist
from time import time
from asset import *
from test import test
from utils import update_config,check_config,discretize_logit,cal_acc,time_trans,split_data
from data_loader import universed_loader
from torch.utils.tensorboard import SummaryWriter

def init_param(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def init_model(config):
    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True
    gpu_num = torch.cuda.device_count()
    if gpu_num<2: # disable ddp
        config.ddp=0

    activate_type = config.activate_func.lower()
    config.activate_func = activate_set[activate_type]()

    if config.ddp: # enable ddp
        local_rank=int(os.environ['LOCAL_RANK'])
        backend='nccl' if sys.platform=='linux' else 'gloo'
        dist.init_process_group(backend,init_method='env://')
        model = nn.parallel.DistributedDataParallel(model_set[config.model.lower()](config),
                                                    device_ids=[local_rank],
                                                    output_device=local_rank)
        config.local_rank=local_rank
    else: 
        model = model_set[config.model.lower()](config)
        if gpu_num == 1:
            model.cuda() 

        config.local_rank=0

    model.apply(init_param)
    config.activate_func = activate_type

    return model

def init_logger(config,fold=None):
    dir_name = '-'.join([f'{config.task}',
                         f'{config.vectorizer}' if config.vectorizer != 'ngram' else f'{config.n}{config.vectorizer[1:]}',
                         f'{config.model}',
                         f'{config.train_plan}' if ':' not in config.train_plan else '{}'.format(''.join(config.train_plan.split(':'))),
                         f'lr({int(math.log10(config.lr))})',
                         f'dout({config.dropout_rate})' if 'dropout_rate' in config else '', 
                         f'{config.loss_func}',
                         f'{config.optimizer}',
                         f'{config.activate_func}'])
    suffix = os.path.join(dir_name,f'fold_{fold}') if fold is not None else dir_name
    config.save_dir=os.path.join(config.record_dir,suffix)
    os.makedirs(config.save_dir,exist_ok=True)

    config.weight_save_path=os.path.join(config.save_dir, 'best.pth')
    config.task_train_log_path=os.path.join(config.save_dir,'train_log.txt')

def init_optim(config, model):
    params = model.module.parameters() if config.ddp else model.parameters()
    try:
        optimizer = optim_set[config.optimizer.lower()](params, 
                                                        lr=config.lr, 
                                                        weight_decay=config.weight_decay)
    except:
        optimizer = optim_set[config.optimizer.lower()](params, 
                                                        lr=config.lr, 
                                                        weight_decay=config.weight_decay,
                                                        momentum = config.momentum)
    
    if config.model.lower() != 'logistic':
        return optimizer
    else:
        config.model = 'sub_logistic'
        optimizers = [init_optim(config, net) for net in model.bk]
        config.model = 'logistic'
        return optimizers

def train_step(config, model, dataloader, optimizer, criterion):
    model.train()
    iters_num=len(dataloader.train_loader)
    loss_collect=torch.zeros(iters_num)
    
    for batch,(datas,labels) in enumerate(dataloader.train_loader):
        datas,labels = datas.to(config.device), labels.to(config.device)
        optimizer.zero_grad()
        logit = model(datas).squeeze(-1)
        logit = discretize_logit(config,logit,test=False) 
        loss = criterion(logit,labels)
        loss.backward()
        optimizer.step()
        
        loss_collect[batch]=loss
    return loss_collect.mean().item()

def train(config, model, dataloader, fold=None):
    init_logger(config,fold)

    optimizer = init_optim(config, model)
    criterion = loss_set[config.loss_func.lower()]()
    
    writer=SummaryWriter(config.save_dir)
    
    print("\033[0;33;40mtraining...\033[0m")
    time_start=time()
    time_collect=torch.zeros(config.epoch)
    for i in range(config.epoch):
        epoch_start_time=time()
        
        if config.model.lower() != 'logistic':
            epoch_loss=train_step(config, model, dataloader, optimizer, criterion)
        else:
            epoch_loss = []
            dataloaders = dataloader.dataloaders # binary train & val dataloader
            for net_idx,net in enumerate(model.bk): # train each classifier
                net_epoch_loss=train_step(config, net, 
                                          dataloaders[net_idx], optimizer[net_idx], 
                                          criterion)
                epoch_loss.append(net_epoch_loss)
            epoch_loss = sum(epoch_loss)/len(epoch_loss)

        top1_acc,top5_acc=cal_acc(config, 
                                  dataloader.val_loader if config.model.lower() != 'logistic' else dataloader.origin_val_loader.val_loader, 
                                  model)
        time_end=time()
        time_collect[i]=time_end-epoch_start_time
        
        avg_epoch_time=time_collect.sum()/(i+1)
        writer.add_scalar('{}/epoch_loss'.format(config.dataset),epoch_loss,i)
        writer.add_scalar('{}/top1_acc'.format(config.dataset), top1_acc, i)
        writer.add_scalar('{}/top5_acc'.format(config.dataset), top5_acc, i)
        if i==0:
            best_epoch=i
            best_loss=epoch_loss
            best_acc=top1_acc
            best_acc5=top5_acc
        else:
            if top1_acc>=best_acc:
                best_epoch=i
                best_loss=epoch_loss
                best_acc=top1_acc
                best_acc5=top5_acc

                torch.save(model.state_dict(), config.weight_save_path)
        
        log_context="epoch: {}, avg_loss: {:.3f}, Top 1_acc: {:.3f}, Top 5_acc: {:.3f}\n".format(i+1,epoch_loss,top1_acc,top5_acc)
        log_context+="Average {:.1f} s/epoch | ".format(avg_epoch_time)
        log_context+="Spend: {}\n".format(time_trans(time_end-time_start))
        log_context+='Best: epoch_{}, loss_{:.3f}, Top 1_acc: {:.3f}, Top 5_acc: {:.3f}\n'.format(best_epoch+1,best_loss,best_acc,best_acc5)
        log_context+='-'*40+'\n'
        
        with open(config.task_train_log_path,'a',encoding='utf-8') as log_writer:
            print('\033[0;33;40m'+'-'*40+'\033[0m')
            log_writer.writelines(log_context)
            print("\033[0;33;40m{}\033[0m".format(log_context))
    writer.close()
    
    return best_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config_file', type=str, default=r'E:\AIL\project\NLP-Laboratary\Config\pick_optim\sgd.json')#f'{sys.path[0]}/Config/softmax_pattern.json')   
    parser.add_argument('-r','--record_dir', type=str, default=r'E:\AIL\project\NLP-Laboratary\Results\pick_optim')#f'{sys.path[0]}/Results')
    config = parser.parse_args()
    
    config = update_config(config)
    config.task = os.path.split(config.dataset)[-1]
    config.dataset = os.path.join(sys.path[0],config.dataset)
    check_config(config)

    vectorizer, train_val_data, _ = preprocess_set[config.vectorizer](config)
    train_num, val_num, train_val_sections = split_data(config, train_val_data)
    
    if isinstance(val_num,list): # k-fold mode
        fold_acc = []
        for fold_idx, val_idx in enumerate(val_num):
            config.seed += val_idx

            val_data = train_val_sections[val_idx].copy()
            train_data = pd.concat(train_val_sections[train_num[fold_idx][0]].copy() + \
                                   train_val_sections[train_num[fold_idx][1]].copy())
            
            dataloader = universed_loader(config, vectorizer, train_data, val_data)
            
            model = init_model(config)

            acc = train(config, model, dataloader, fold_idx)
            
            fold_acc.append(acc)
            if config.ddp:
                dist.destroy_process_group()
            del model
        
        max_fold_idx = np.argmax(fold_acc)
        best_weight_dir = os.path.join(os.path.dirname(config.save_dir),
                                       f'fold_{max_fold_idx}')
        new_dir_name = os.path.join(os.path.dirname(best_weight_dir),
                                    f'(best)fold_{max_fold_idx}')
        os.renames(best_weight_dir,
                   new_dir_name)
        config.best_weight_path = os.path.join(new_dir_name,'best.pth')
        config.save_dir = os.path.dirname(new_dir_name)
    else:
        train_data = pd.concat(train_val_sections[:train_num])
        val_data = pd.concat(train_val_sections[train_num:])

        dataloader = universed_loader(config, vectorizer, train_data, val_data)

        model = init_model(config)

        train(config, model, dataloader)
        config.best_weight_path = config.weight_save_path
    
    with open(os.path.join(config.save_dir, 'config.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=4)
    
    test(config)

