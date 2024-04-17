# 1. Opengait 注入代码

## 1.  修改 evaluator.py

1. 打开 `\OpenGait-master\opengait\evaluation\evaluator.py` 
2. 在原有的`single_view_gallery_evaluation()`函数内`view_num = len(view_list)`句**后**，插入下述代码

	```python
	import torch.nn.functional as F
	
	pseq_mask = np.isin(seq_type, [j for i in probe_seq_dict[dataset].values() for j in i])
	gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset])
	
	probe_feat = feature[pseq_mask, :]
	gallery_feat = feature[gseq_mask, :]
	probe_label = label[pseq_mask]
	gallery_label = label[gseq_mask]
	
	dist = cuda_dist(probe_feat, gallery_feat, metric)
	prob = F.softmax(-dist,1).cpu().numpy()
	
	all_cls = sorted(np.unique(gallery_label))
    proc_pred = np.zeros([len(prob),len(all_cls)])
    proc_label = np.zeros_like(proc_pred)

    for cls_id,cls in enumerate(all_cls):
        for sample_id,sample_prob in enumerate(prob):
            proc_pred[sample_id][cls_id] = np.mean(sample_prob[gallery_label==cls])
            if cls==probe_label[sample_id]:
                proc_label[sample_id,cls_id]=1
	```

3. 然后将`single_view_gallery_evaluation()`函数内的`result_dict = {}`上加一句

	```python
	npy_dict = {'proc_pred':proc_pred,'proc_label':proc_label}
	```

4. 将`single_view_gallery_evaluation()`函数的返回改为

	```python
	return result_dict,npy_dict
	```

## 2. 修改 base_model.py
1. 打开 `\OpenGait-master\opengait\modeling\base_model.py` 
2. 在文件的最后，将 `return eval_func(info_dict, dataset_name, **valid_args)` 改为以下：

	```python
	import os
	npy_dir = r'' # 指定一个文件夹，用于保存每个模型的预测数据
	os.makedirs(npy_dir,exist_ok=True)
	return_dict,npy_dict = eval_func(info_dict, dataset_name, **valid_args)
	model_name=model.cfgs['model_cfg']['model']
	model_name = 'GaitBase' if 'Base' in model_name else model_name
	for data_name,data in npy_dict.items():
	    if '@' not in data_name:
	        np.save(os.path.join(npy_dir,'{}_{}.npy'.format(model_name,data_name.split('_')[-1])),data)
	
	return return_dict
	```

3. 指定新代码里的`npy_dir`

# 2. 运行Opengait

- 在`\OpenGait-master\test.sh`中，配置好4个模型的运行语句
	- **执行测试**，而不是训练
	- **确保模型权重存在**，即`\OpenGait-master\output`对应模型权重文件中，存在轮数等于yaml文件`restore_hint`字段值的权重
	- 请查看各模型的数据集划分文件，**保持所有模型用于测试的个体相同**：即要求不同模型的yaml文件中，`dataset_partition`字段的值相同
- 测完4个模型后，在第二大步指定的文件夹中将有8个npy文件

# 3. 画图

- 打开`Model_Roc.py`，拉到最后
- 配置`npy_dir`为含8个npy文件的**文件夹路径**
- 配置`img_save_path`为图片保存路径，如`E:\abc\Roc_result.png`
- 运行`Model_Roc.py`，并查看`E:\abc\Roc_result.png`。<font size=2, color=pink>(图的配色是随机的，如果图线看不清，多运行几次这个`Model_Roc.py`)</font>

# 4. 注

以上所有注入代码不影响原始Opengait的运行，只不过在每次测试时会算多一点，可根据个人喜好决定完成后是否删除注入代码
