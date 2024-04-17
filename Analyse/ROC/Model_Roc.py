import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import roc_curve, auc

def draw_roc(label, pred, save_path, type='macro', cls_name=None, show_cls=True, linewidth=2):
    '''
    label: should be one-hot label, shape: sample_num x class_num
    pred: prediction probabilities of each class, value should between (0,1),shape: sample_num x class_num
    type: 'micro' / 'macro' / 'both'. 'micro' uses row prediction data while macro uses column one.
    cls_name: the list of classes' name
    show_cls: whether to draw each class's line
    '''
    assert os.path.splitext(save_path)[-1] in ('.png','.jpg','.jpeg'),'Save_path should specify the img type.'
    assert type in ('micro', 'macro','both'), 'Only support \'micro\', \'macro\' and \'both\' for parameter \'type\'.'
    
    # pretreatment
    if not isinstance(label,dict):
        label = {'default_model':label}
        pred = {'default_model':pred}
    else:
        show_cls=False

    # collect all models' all type fpr/tpr/auc
    fpr, tpr, roc_auc = {},{},{}
    '''
    {'model_name1':{'cls': {cls_name1:1d array, / scalar for auc
                            cls_name2:1d array, / scalar for auc
                            ...},
                    'macro': 1d array, / scalar for auc
                    'macro_std': 1d array, / not exist in auc
                    'micro': 1d array, / scalar for auc,
                    'micro_std': 1d array}, / not exist in auc
     
     'model_name2':{'cls': {cls_name1:1d array, / scalar for auc
                            cls_name2:1d array, / scalar for auc
                            ...},
                    'macro': 1d array, / scalar for auc
                    'macro_std': 1d array, / not exist in auc
                    'micro': 1d array, / scalar for auc,
                    'micro_std': 1d array}, / not exist in auc
    ...}
    '''
    for model in label.keys():
        fpr[model], tpr[model], roc_auc[model] = {},{},{}
        
        model_label = label[model] # n,c
        model_pred = pred[model] # n,c
        
        if cls_name is None:
            cls_name = ['{:03d}'.format(_) for _ in range(model_pred.shape[-1])]
        n_classes = len(cls_name)
        assert  n_classes == model_pred.shape[-1], f'Wrong number of classes\' name.'
        assert model_pred.shape == model_label.shape, f'Shape not match! Labels\' is {label.shape} while Prediction is {pred.shape}.'
        
        # cal fpr/tpr
        if type in ('macro','both'): ## macro
            cls_fpr,cls_tpr,cls_auc = {},{},{}
            for id,cls in enumerate(cls_name):
                cls_fpr[cls], cls_tpr[cls], _ = roc_curve(model_label[:, id], model_pred[:, id])
                cls_auc[cls] = auc(cls_fpr[cls], cls_tpr[cls])
            
            fpr[model]['cls']=cls_fpr
            tpr[model]['cls']=cls_tpr
            roc_auc[model]['cls']=cls_auc

            ##  merge interpolated tpr by mean
            all_tpr = []
            all_fpr = np.unique(np.concatenate([cls_fpr[cls] for cls in cls_name])) ## aggregate all false positive rates
            for cls in cls_name:
                all_tpr.append(np.interp(all_fpr, cls_fpr[cls], cls_tpr[cls]))
            std_tpr = np.std(all_tpr, 0)

            fpr[model]["macro"] = all_fpr
            tpr[model]["macro"] = np.mean(all_tpr,0)
            roc_auc[model]["macro"] = auc(fpr[model]["macro"], tpr[model]["macro"])
            fpr[model]['macro_std'] = all_fpr
            tpr[model]['macro_std'] = std_tpr

        if type in ('micro','both'):    ## micro
            fpr[model]["micro"], tpr[model]["micro"], _ = roc_curve(model_label.ravel(), model_pred.ravel())
            roc_auc[model]["micro"] = auc(fpr[model]["micro"], tpr[model]["micro"])

    # Plot all ROC curves
    colors = np.random.rand((n_classes+2)*len(fpr),3)
    colors = np.maximum(colors,0.5)
    line_idx = -1
    plt.figure() ## figsize=(5, 5)
    for model,roc_data in fpr.items():
        for roc_type, fpr_data in roc_data.items():
            tpr_data = tpr[model][roc_type]
            if roc_type.endswith('_std') :
                prefix = roc_type.split('_')[0]
                type_tpr = tpr[model][prefix]
                std_tpr = tpr_data
                tprs_upper = np.minimum(type_tpr + std_tpr, 1)
                tprs_lower = np.maximum(type_tpr - std_tpr, 0)

                plt.fill_between(fpr_data,
                                 tprs_lower,
                                 tprs_upper,
                                 color=colors[line_idx], #"grey",
                                 alpha=0.2)
                                 #label=r"$\pm$ 1 std. dev.")
                
            elif roc_type == 'cls':
                if show_cls:
                    for cls, cls_fpr in fpr_data.items():
                        line_idx+=1
                        plt.plot(cls_fpr, 
                                 tpr_data[cls], 
                                 color=colors[line_idx], 
                                 lw=max(linewidth-1,1),
                                 alpha=0.3,
                                 label='Class {} (auc {:0.4f})'.format(cls, roc_auc[model][roc_type][cls]))
            
            else:
                line_idx+=1
                if type == 'both':
                    line_label = '{} ({}, auc {:0.4f})'.format(model, roc_type, roc_auc[model][roc_type])
                else:
                    line_label = '{} (auc {:0.4f})'.format(model, roc_auc[model][roc_type])
                plt.plot(fpr_data, 
                         tpr_data,
                         color=colors[line_idx], 
                         linestyle='-', 
                         linewidth=linewidth,
                         label=line_label)

    plt.plot([0, 1], [0, 1], 'k--', lw=max(linewidth-1,1),label='Chance level (auc 0.5)') # stochastic classify
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparison of Roc Curve')
    plt.legend(loc='lower right') 

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    #plt.show()

def organize_data(npy_dir):
    npy_list = glob(npy_dir+os.path.sep+'*.npy')
    assert len(npy_list)>0, f'Can\'t find any npy files in {npy_dir}.'

    label={}
    pred={}
    for file_path in npy_list:
        file_name = os.path.split(file_path)[-1]
        *model_name,type = file_name[:-4].split('_')
        model_name='+'.join(model_name)
        if type.lower() == 'pred':
            pred[model_name] = np.load(file_path)
        else:
            label[model_name] = np.load(file_path)
    
    return label,pred

if __name__=='__main__':
    npy_dir = r'E:\AIL\project\Pattern\Analyse\ROC\MNIO\origin'
    img_save_path = r'C:\Users\Administrater\Desktop\MNIO_analysis\roc\origin.png'

    label,pred = organize_data(npy_dir)
    while 1:
        draw_roc(label, pred, save_path=img_save_path, type='macro')
        input()