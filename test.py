import torch,os,json,sys,argparse
import torch.nn as nn
import torch.distributed as dist
from utils import predict
from asset import *
from data_loader import universed_loader

def init_model(config):
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

    config.activate_func = activate_type

    return model

def test(config):    
    vectorizer, _, test_data = preprocess_set[config.vectorizer](config)

    dataloader = universed_loader(config, vectorizer, test_data=test_data)

    model=init_model(config)
    model_dict = model.state_dict()
    saved_dict = torch.load(config.best_weight_path, map_location=config.device) 
    model_dict.update(saved_dict)
    model.load_state_dict(model_dict)

    predict(config, dataloader.test_loader, model)

    if config.ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config_file', type=str, default=f'{sys.path[0]}/Config/softmax_pattern.json')   
    config = parser.parse_args()
    with open(config.config_file, 'r') as f:
        config.__dict__ = json.load(f) 
    test(config)