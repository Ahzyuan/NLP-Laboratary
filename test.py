import torch,argparse,os,json
from time import localtime
from model import TJ_model
from helper_func import cal_acc
from data_loader import TJ_dataloader

def test(weight_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default=r'E:\AIL\project\TJ_Palm\Config')   
    parser.add_argument('--dataset', type=str, default='TJ')   
    config = parser.parse_args()       
    with open(os.path.join(config.config_dir, 'hyp_{}.json'.format(config.dataset)), 'r') as f:
        config.__dict__ = json.load(f)    
    
    test_log_dir=r'E:\AIL\project\TJ_Palm\Results'
    if not os.path.exists(test_log_dir):
        os.makedirs(test_log_dir)

    dataloader = TJ_dataloader(config) 

    model=TJ_model(config).to(config.device)
    model_dict = model.state_dict()
    saved_dict = torch.load(weight_path, map_location=config.device) 
    model_dict.update(saved_dict)
    model.load_state_dict(model_dict)

    test_acc=cal_acc(config,dataloader.test_loader,model)
    struct_time=localtime()
    test_res='{}\\{}\\{} {}:{}  Top 1_acc: {:.3f}  Top 5_acc: {:.3f}\n'.format(*struct_time[:5],*test_acc)
    print(test_res)
    with open(os.path.join(test_log_dir, 'model_performance.txt'),'a',encoding='utf-8') as perform_writer:
        perform_writer.writelines(test_res)

if __name__=='__main__':
    best_weight=r'E:\AIL\project\TJ_Palm\Results\saved_model\best.pth'
    test(best_weight)