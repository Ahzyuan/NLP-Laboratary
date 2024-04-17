import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from glob import glob

def draw_confusion_matrix(ax, 
                          label_npy, pred_npy, cls_name=None, 
                          title="Confusion Matrix", is_text=False, is_axis=True, fontsize=6, 
                          save_csv_dir=None, save_img_dir=None, dpi=300):
    '''
    ax: 画图区域\n
    label_npy: 独热标签npy路径\n
    pred_npy: 预测概率npy路径\n
    cls_name: 类名列表, 比如['cat','dog','flower',...]。默认为自然数填充\n
    title: 图标题\n
    is_text: 是否在格子中显示数字\n
    is_axis: 是否显示全部坐标轴\n
    fontsize: 字体大小\n
    save_csv_dir: 保存混淆矩阵为csv格式的路径, None表示不保存\n
    save_img_dir: 保存混淆矩阵为图片格式的路径, None表示不保存\n
    dpi: 保存到文件的分辨率, 论文一般要求至少 300 dpi
    '''
    label_onehot = np.load(label_npy)
    pred_logit = np.load(pred_npy)
    label = np.nonzero(label_onehot)[1] 
    pred = np.argmax(pred_logit,1)
    if cls_name is None:
        cls_name = list(map(lambda x:'{:02d}'.format(x), range(label_onehot.shape[-1])))
    else:
        assert len(cls_name)==label_onehot.shape[-1]
    
    cm = confusion_matrix(y_true=label, y_pred=pred, normalize='true')
    if save_csv_dir is not None:
        np.savetxt(os.path.join(save_csv_dir,title+'.csv'), cm, fmt='%.4f', delimiter=',')

    im=ax.imshow(cm, cmap='Blues') # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    
    if is_text:
        for i in range(cls_name.__len__()):
            for j in range(cls_name.__len__()):
                value = format('%d' % cm[j, i])
                color = (1, 1, 1) if cm[j, i]>50 else (0, 0, 0)  # 对角线字体白色, 其他黑色
                plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    ax.set_title(title,fontsize=40)
    ax.set_yticks(range(len(cls_name)), cls_name, fontsize=fontsize) if is_axis else ax.set_yticks([])
    ax.set_xticks(range(len(cls_name)), cls_name, rotation=90, fontsize=fontsize)
    ax.tick_params(axis='both', direction='in', length=5)

    if save_img_dir is not None:
        fig_single, ax_single = plt.subplots()
        im_single=ax_single.imshow(cm, cmap='Blues')
        ax_single.set_title(title)
        ax_single.set_xlabel("Predict label")
        ax_single.set_ylabel("Truth label")
        ax_single.set_yticks(range(len(cls_name)), cls_name, fontsize=fontsize*0.4)
        ax_single.set_xticks(range(len(cls_name)), cls_name, rotation=90, fontsize=fontsize*0.4)
        ax.tick_params(axis='both', direction='in', length=5)

        plt.tight_layout()

        cax = fig_single.add_axes([0.82, 0.12, 0.02, 0.8])  # 定义colorbar的位置和尺寸
        colorbar=fig_single.colorbar(im_single, cax=cax)  # 添加colorbar，其中im是最后一个子图上的绘图对象
        colorbar.ax.tick_params(labelsize=fontsize*0.5)
        plt.savefig(os.path.join(save_img_dir, title+'.png'), bbox_inches='tight', dpi=dpi)
        plt.close()
    return im

if __name__=='__main__':
    npy_base = r'E:\AIL\project\Pattern\Analyse\ROC\MNIO\origin'
    
    save_path=os.path.join(r'C:\Users\Administrater\Desktop\MNIO_analysis\confusion_matrix',
                           os.path.split(npy_base)[-1])
    save_matrix_dir = os.path.join(save_path,'csv')
    os.makedirs(save_path,exist_ok=True)
    os.makedirs(save_matrix_dir,exist_ok=True)
    
    #os.makedirs(os.path.split(save_img)[0],exist_ok=True)
    

    labels = glob(npy_base+'/*_label.npy')
    fig, axs = plt.subplots(1, len(labels),figsize=(10*len(labels), 20))
    for id,label_path in enumerate(labels):
        *model, _ = os.path.split(label_path)[-1].split('_')
        pred_path = label_path.replace('label.npy','pred.npy')
        

        im = draw_confusion_matrix(ax=axs[id],
                                label_npy=label_path,
                                pred_npy=pred_path,
                                is_axis= False if id else True,
                                title='_'.join(model) if isinstance(model,list) else model,
                                fontsize=14, # 横纵坐标字体
                                save_csv_dir=save_matrix_dir,
                                save_img_dir=save_path) 
    
    save_all_img=os.path.join(save_path,'all_img')

    fig.text(-0.015, 0.5, 'Truth label', va='center', rotation='vertical',fontsize=40)
    fig.text(0.5, 0.2, 'Predict label', ha='center',fontsize=40)

    plt.tight_layout()

    cax = fig.add_axes([1, 0.25, 0.01, 0.5])  # 定义colorbar的位置和尺寸
    colorbar=fig.colorbar(im, cax=cax)  # 添加colorbar，其中im是最后一个子图上的绘图对象
    colorbar.ax.tick_params(labelsize=20)

    plt.savefig(save_all_img, bbox_inches='tight', dpi=300)