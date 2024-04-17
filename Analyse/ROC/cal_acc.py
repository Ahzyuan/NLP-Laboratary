import os
import numpy as np
from glob import glob

def cal_acc(path):
    labels = glob(path+'/*_label.npy')
    for label_path in labels:
        *model,_ = os.path.split(label_path)[-1].split('_')
        model = '_'.join(model) if isinstance(model,list) else model
        pred_path = os.path.join(path,f'{model}_pred.npy')

        label_onehot = np.load(label_path)
        pred_logit = np.load(pred_path)

        label = np.nonzero(label_onehot)[1]
        pred = np.argmax(pred_logit, 1)
        print('{}: {:.4f}'.format(model, sum(label==pred)/label.size)) 

if __name__=='__main__':
    cal_acc(r'C:\Users\Administrater\Desktop\tiger_npy')