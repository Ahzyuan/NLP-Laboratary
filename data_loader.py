import torch
from itertools import combinations
from torch.utils.data import DataLoader,Dataset
from torch.nn.functional import one_hot

class universed_dataset(Dataset):
    def __init__(self, csv_data, config, vectorizer, data_column='Sentence',label_column='Category'): 
        if label_column and isinstance(csv_data[label_column].iloc[0],str):
            csv_data[label_column] = csv_data[label_column].apply(lambda x:config.cls_list.index(x))
        self.data = csv_data
        self.data_col = data_column
        self.label_col = label_column
        self.config = config
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.data)

    def correct_label(self,label):
        if 'logistic' in self.config.model:
            return torch.tensor(label, dtype=torch.float32)
        else: #if 'softmax' in self.config.model:
            if 'ce' in self.config.loss_func: 
                return torch.tensor(label, dtype=torch.long)
            else:
                return one_hot(torch.tensor([label]), len(self.config.cls_list)).squeeze().float()

    def __getitem__(self, idx):
        text = self.data.iloc[idx][self.data_col]
        bow_vector = self.vectorizer.transform([text]).toarray()[0] if self.config.vectorizer in ('bow','tfidf','ngram') else self.vectorizer(text)
        if self.label_col:
            label = self.data.iloc[idx][self.label_col]
            return torch.tensor(bow_vector, dtype=torch.float32), self.correct_label(label)
        else:
            return torch.tensor(bow_vector, dtype=torch.float32), idx

class universed_loader():
    def __init__(self, config, vectorizer, 
                 train_data=None, val_data=None, test_data=None,
                 data_column='Sentence',label_column='Category'):
        
        if config.model.lower() == 'logistic' and test_data is None:
            config.model = 'sub_logistic'
            self.origin_val_loader = universed_loader(config, vectorizer, val_data=val_data.copy())
            train_datas, val_datas = self.binary_data(config, train_data, val_data, label_column)
            self.dataloaders = [universed_loader(config, vectorizer, train_data, val_data) for train_data, val_data in zip(train_datas, val_datas)]
            config.model = 'logistic'
        else:   
            self.get_loader(config, vectorizer, train_data, val_data, test_data, data_column, label_column)
    

    def get_loader(self, config, vectorizer, train_data=None, val_data=None, test_data=None, data_column='Sentence',label_column='Category'):
        if train_data is not None:
            self.train_set=universed_dataset(train_data, config, vectorizer, data_column, label_column)
            self.train_loader=DataLoader(self.train_set,
                                         batch_size=config.batch_size,
                                         shuffle=True,
                                         num_workers=config.num_workers)
            
        if val_data is not None:
            self.val_set=universed_dataset(val_data, config, vectorizer, data_column, label_column)
            self.val_loader=DataLoader(self.val_set,
                                       batch_size=config.batch_size,
                                       shuffle=False,
                                       num_workers=config.num_workers)
        
        if test_data is not None:
            self.test_set=universed_dataset(test_data, config, vectorizer, data_column, label_column=None)
            self.test_loader=DataLoader(self.test_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=config.num_workers)

    def binary_data(self, config, train_data, val_data, label_column='Category'):
        dual_cls_list=list(combinations(config.cls_list,2)) 
        config.dual_cls_list=[(config.cls_list.index(i),config.cls_list.index(j))for i,j in dual_cls_list]
        
        train_label = train_data[label_column].copy()
        val_label = val_data[label_column].copy()
        train_datas, val_datas = [],[]

        for pick_cls1, pick_cls2 in dual_cls_list: 
            for set_id,label_list in enumerate([train_label,val_label]):
                mask1=label_list==pick_cls1 
                mask2=label_list==pick_cls2 

                if set_id:
                    now_data = val_data  
                    now_collector = val_datas
                else:
                    now_data = train_data  
                    now_collector = train_datas
                
                now_data[label_column][mask1]=0 
                now_data[label_column][mask2]=1 
                mask=mask1+mask2 
                now_collector.append(now_data[mask])
        
        return train_datas, val_datas