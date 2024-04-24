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

def init_logger(config,fold=None,net_idx=None):
    dir_name = '-'.join([f'{config.task}',
                         f'{config.vectorizer}',
                         f'{config.model}',
                         f'lr({int(math.log10(config.lr))})',
                         f'{config.loss_func}',
                         f'{config.optimizer}'])
    suffix = os.path.join(dir_name,f'fold_{fold}') if fold is not None else dir_name
    suffix = os.path.join(suffix,'cls_{}-{}'.format(*config.dual_cls_list[net_idx])) if net_idx is not None else suffix
    config.save_dir=os.path.join(config.record_dir,suffix)
    os.makedirs(config.save_dir,exist_ok=True)

    config.weight_save_path=os.path.join(config.save_dir, 'best.pth')
    config.task_train_log_path=os.path.join(config.save_dir,'train_log.txt')

def train_wrapper(func):
    def wrapper(*args, **kwargs):
        if hasattr(args[2], 'dataloaders'): # logistic_ovo
            config, model, dataloader, *fold_idx = args
            dataloaders = dataloader.dataloaders
            net_best_weight = []
            fold_idx = fold_idx[0] if fold_idx else None
            
            for net_idx,net in enumerate(model.bk): # train each classifier
                data = dataloaders[net_idx]
                func(config, net, data, fold_idx, net_idx)
                net_best_weight.append(config.weight_save_path)

            for net_idx, best_weight_path in enumerate(net_best_weight): # aggregate best model
                model_dict = model.bk[net_idx].state_dict()
                saved_dict = torch.load(best_weight_path, map_location=config.device) 
                model_dict.update(saved_dict)
                model.bk[net_idx].load_state_dict(model_dict)
            
            val_loader = dataloader.origin_val_loader.val_loader
            acc,top5_acc=cal_acc(config, val_loader, model)
            print('-'*40)
            print(f'top1 acc: {acc:.3f}\ntop5 acc: {top5_acc:.3f}\n')

            config.save_dir = os.path.dirname(config.save_dir)
            config.weight_save_path = os.path.join(config.save_dir,'best.pth')
            torch.save(model.state_dict(), config.weight_save_path)
        else:
            acc = func(*args, **kwargs)
        return acc
    return wrapper

@train_wrapper
def train(config, model, dataloader, fold=None, net_idx=None):
    '''
    net_idx: only effect when config.model is 'logistic',use to distinguish different classifiers
    '''
    init_logger(config,fold,net_idx)

    writer=SummaryWriter(config.save_dir)
    
    criterion = loss_set[config.loss_func.lower()]()
    params = model.module.parameters() if config.ddp else model.parameters()
    try:
        optimizer = optim_set[config.optimizer.lower()](params, 
                                                        lr=config.lr, 
                                                        weight_decay=config.weight_decay,
                                                        momentum=config.momentum)
    except:
        optimizer = optim_set[config.optimizer.lower()](params, 
                                                        lr=config.lr, 
                                                        weight_decay=config.weight_decay)

    print("\033[0;33;40mtraining...\033[0m")
    time_start=time()
    time_collect=torch.zeros(config.epoch)
    for i in range(config.epoch):
        epoch_start_time=time()
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
            print("\repoch: {}/{}, iters: {}/{}, loss: {:.3f}".format(
                str(i+1).zfill(len(str(config.epoch))), 
                config.epoch, 
                str(batch+1).zfill(len(str(iters_num))), 
                iters_num, loss),
                end='')
            loss_collect[batch]=loss
        
        print()
        top1_acc,top5_acc=cal_acc(config, dataloader.val_loader, model)
        time_end=time()
        time_collect[i]=time_end-epoch_start_time
        epoch_loss=loss_collect.mean().item()
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
            if top1_acc>best_acc:
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
    parser.add_argument('--config_file', type=str, default=f'{sys.path[0]}/Config/softmax_pattern.json')   
    parser.add_argument('--record_dir', type=str, default=f'{sys.path[0]}/Results')
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

