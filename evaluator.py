import torch
import numpy as np
import torch.nn as nn
import prettytable as ptb
from functools import reduce
from operator import mul
from time import time
from tqdm import tqdm

class Evaluator():
    def __init__(self, model, input_shape:tuple, float_type=32, tb_style=ptb.SINGLE_BORDER):
        '''
        input_shape: (bs=1, channel, h, w)
        '''
        assert len(input_shape)>=4,'Input should have shape lager than 3' # 保证有batch_size维度
        assert float_type in [16,32,64],f'float_type must be 16 or 32 or 64, but get {float_type}'
        
        self.model=model
        self.input=torch.rand(input_shape,dtype=eval(f'torch.float{float_type}'))
        self.element_byte=self.input.element_size()  # 计算浮点精度，即一个数要几个字节
        self.tb_style=tb_style

        self.clear_data()

# main usage

    def count_params(self, print_tb=True, save_path=None):
        '''
        only_reg: only count the params which require grad \n
        print_tb: whether to print the result table
        '''
        if save_path:
            assert save_path[-4:]=='.csv','Strongly recommend using csv format!'
        
        tb_fields=['id','Params Name','Requires_grad','Params_num']
        self.clear_data()
    
        for id,(name, parameter) in enumerate(self.model.named_parameters()): 
            p_ptr=parameter.data_ptr()
            if p_ptr in self.param_ptr:
                self.neglect_id.append(id)
                continue
            
            self.param_ptr.append(p_ptr)
            p_num=parameter.numel()
            p_reg=False
            if parameter.requires_grad:
                p_reg=True
                self.reg_num+=p_num
            self.params_data.append([id,name,p_reg,p_num])
            self.total_num+=p_num
    
        tb=self.draw_table(tb_fields,self.params_data)
    
        if print_tb:
            print(tb)
    
        if save_path:
            with open(save_path,'w') as w:
                w.writelines(tb.get_csv_string().replace('\n',''))
    
        if self.neglect_id:
            print('ID ',*self.neglect_id,' share memory with some of others!')
        word_length=len('Params (requires_grad)')
        print('# '+'Params (requires_grad)'.center(word_length)+' : {}  =  {} M  =  {} G'.format(self.reg_num,self.reg_num/1e6,self.reg_num/1e9))
        print('# '+'Params (total)'.center(word_length)+' : {}  =  {} M  =  {} G'.format(self.total_num,self.total_num/1e6,self.total_num/1e9))

    def cal_FandM(self, print_tb=True, save_path=None):  # to calculate FLOPs and MACs
        if save_path:
            assert save_path[-4:]=='.csv','Strongly recommend using csv format!'

        tb_fields=['Module','Kernel Size([H,W])','Bias','Output Shape','MACs','FLOPs']
        tb_title='Input: tensor({})'.format(list(self.input.shape))
        self.clear_data()

        for module_name,module in self.model.named_children():
            if module._modules:
                self.modules_dict.update(self.unfold_layer(module,module_name))
            else:
                self.modules_dict.update({str(id(module)):(module_name,module)})  # 展开模型所有sequential、modulelist、moduledict等，以 {层地址:(层名，层)}的形式存为字典

        handle_list=list(map(lambda x:self.regist_hook(x[1]),self.modules_dict.values()))
        logits=self.model(self.input)
        list(map(lambda x:x.remove(),handle_list))

        tb=self.draw_table(tb_fields,self.FandM_data)
        tb.title=tb_title
        tb.add_autoindex('Forward Step')

        if print_tb:
            print(tb)

        if save_path:
            with open(save_path,'w') as w:
                w.writelines(tb.get_csv_string().replace('\n',''))

        word_length=len('Total FLOPs')
        print('# '+'Total MACs'.rjust(word_length)+' : {}  =  {} M  =  {} G'.format(int(self.total_macs),self.total_macs/1e6,self.total_macs/1e9))
        print('# '+'Total FLOPs'.rjust(word_length)+' : {}  =  {} M  =  {} G'.format(int(self.total_flops),self.total_flops/1e6,self.total_flops/1e9))

    def count_memory(self, optimizer_type, print_tb=True, save_path=None):  # to calculate MAC
        if save_path:
            assert save_path[-4:]=='.csv','Strongly recommend using csv format!'
    
        tb_module_fields=['Module','Output Shape','Memory/MB']
        tb_param_fields=['Param Id','Param(Requires_Grad)','Floats Num','Memory/MB']
        tb_buffer_fields=['Buffer Id','Buffer','Floats Num','Memory/MB']
        tb_optim_fields=['Item Id','Optimizer Item','Params Num(Shared)','Memory/MB'] # Params Num为不共享内存的参数，Shared为共享参数，总参数等于两者之和
        tb_title='Input: tensor({},dtype=\'{}\')'.format(list(self.input.shape),str(self.input.dtype))
        self.clear_data()
    
        for module_name,module in self.model.named_children():
            if module._modules:
                self.modules_dict.update(self.unfold_layer(module,module_name))
            else:
                self.modules_dict.update({str(id(module)):(module_name,module)})  # 展开模型所有sequential、modulelist、moduledict等，以 {层地址:(层名，层)}的形式存为字典
    
        for no,(name, parameter) in enumerate(self.model.named_parameters()): 
            p_ptr=parameter.data_ptr()
            if p_ptr in self.param_ptr:
                self.neglect_id.append(id)
                continue
            
            self.param_ptr.append(p_ptr)
            p_num=parameter.numel()
            p_memory=p_num*self.element_byte # 计算各参数的字节数
            self.param_memory+=p_memory
            p_reg=True if parameter.requires_grad else False
            self.params_data.append([no+1,name+f'({p_reg})',p_num,'{:.4f}'.format(p_memory/1e6)])
    
        def hook_func(module,input,output):
            module_id=str(id(module))
            module_name=self.modules_dict[module_id][0]
            out_memory=output.numel()*self.element_byte # 计算中间层输出的字节数
            self.feat_memory+=out_memory
            self.modules_data.append([module_name,list(output.shape),'{:.4f}'.format(out_memory/1e6)])
    
        handle_list=list(map(lambda x:x[1].register_forward_hook(hook_func),self.modules_dict.values()))
        logits=self.model(self.input)
        list(map(lambda x:x.remove(),handle_list))

        for no,(name, buffer) in enumerate(self.model.named_buffers()): 
            b_num=buffer.numel()
            b_memory=b_num*self.element_byte # 计算各缓存参数的字节数
            self.buffer_memory+=b_memory
            self.buffers_data.append([no+1,name,b_num,'{:.4f}'.format(b_memory/1e6)])

        optimizer=optimizer_type(self.model.parameters(),lr=1e-4)
        optimizer.zero_grad()
        logits.max().backward()
        optimizer.step()
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.data_ptr() in self.param_ptr: 
                    self.shared_num+=1
                    continue
                self.param_ptr.append(param.data_ptr())
                self.op_param_num+=param.numel()
                self.op_param_memory+=self.op_param_num*self.element_byte
        self.op_data.append([1,'Param Groups',f'{self.op_param_num}({self.shared_num})','{:.4f}'.format(self.op_param_memory/1e6)])
    
        self.shared_num=0
        for param, state_dict in optimizer.state.items():
            if param.data_ptr() in self.param_ptr: 
                self.shared_num+=1
            else:
                self.param_ptr.append(param.data_ptr())
                self.state_memory+=param.numel()*self.element_byte
            for name,state_data in state_dict.items():
                if isinstance(state_data,torch.Tensor):
                    self.state_num+=1
                    self.state_memory+=state_data.numel()*self.element_byte
        self.op_data.append([2,'State',f'{self.state_num}({self.shared_num})','{:.4f}'.format(self.state_memory/1e6)])
    
        tb=self.draw_table(tb_module_fields,self.modules_data)
        tb.title=tb_title
        tb.add_autoindex('Forward Step')
    
        divider=['─'*len('Forward Step'),'─'*(len('Param(Requires_Grad)')+3),'─'*len('Param(Output Shape)'),'─'*len('Memory/MB)')]
        tb.add_row(divider)
        tb.add_row(tb_param_fields)
        tb.add_row(divider)
        tb.add_rows(self.params_data)
        
        tb.add_row(divider)
        tb.add_row(tb_buffer_fields)
        tb.add_row(divider)
        if self.buffers_data:
            tb.add_rows(self.buffers_data)
        else:
            tb.add_row(['None','None','None','None'])

        tb.add_row(divider)
        tb.add_row(tb_optim_fields)
        tb.add_row(divider)
        tb.add_rows(self.op_data)
    
        if print_tb:
            print(tb)
    
        if save_path:
            with open(save_path,'w') as w:
                w.writelines(tb.get_csv_string().replace('\n',''))
    
        word_length=len('Optimizer Params Memory')
        print('# '+'Model Params Memory'.rjust(word_length)+' : {:e} MB'.format(self.param_memory/1e6)) # 参数+梯度的内存
        print('# '+'Model Buffers Memory'.rjust(word_length)+' : {:e} MB'.format(self.buffer_memory/1e6))
        print('# '+'Features Memory'.rjust(word_length)+' : {:e} MB'.format(self.feat_memory/1e6))
        print('# '+'Optimizer Params Memory'.rjust(word_length)+' : {:e} MB'.format(self.op_param_memory/1e6)) 
        print('# '+'Optimizer State Memory'.rjust(word_length)+' : {:e} MB'.format(self.state_memory/1e6))
        print('# '+'Gradian Memory'.rjust(word_length)+' : {:e} MB'.format(self.param_memory/1e6))
        print('# '+'Total Memory'.rjust(word_length)+' : {:e} MB'.format((self.feat_memory+self.op_param_memory+self.state_memory+self.param_memory*2+self.buffer_memory)/1e6))

    def cal_ITTP(self, pick_device='cpu', warmup_iters=50, repeat=50): # to measure IT and TP
        '''
        input_shape: (bs,channel,h,w) \n
        warmup_iters: 预热时的前向传播数 \n
        repeat: 重复测量多少次
        '''
        device=torch.device(pick_device)
        self.model.to(device)
        input = torch.randn(self.input.shape).to(device)

        for _ in tqdm(range(warmup_iters),desc='Warming Up'):  # warm up
            self.model(input)

        IT_list=np.zeros(repeat)  # measure
        if device.type!='cpu':
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            for rep in tqdm(range(repeat),desc='Measuring'):
                if device.type=='cpu':
                    start_time=time()
                else:
                    starter.record()
                self.model(input)
                if device.type=='cpu':
                    end_time=time()
                else:
                    ender.record() 
                    torch.cuda.synchronize()  # WAIT FOR GPU SYNC

            IT_list[rep] = end_time-start_time if device.type=='cpu' else (starter.elapsed_time(ender))*1e-3 # 后者计时单位为ms

        IT=IT_list.mean()
        TP=1/IT
        print('-'*15+'Result'+'-'*15)
        print(f'Device: {pick_device}\nSample Shape: {self.input.shape}')
        print('IT(Inference Time): {:.6f} s'.format(IT))
        print('TP(Throughput): {:.6f} samples/s'.format(TP))

# help function

    def clear_data(self):
        self.modules_dict={}  # 存储展开的所有modules
        self.FandM_data,self.modules_data=[],[]  # 记录FLOPs和MACs表格信息; 记录记录中间层特征的表格信息 
        self.params_data,self.buffers_data=[],[]  # 记录总参数量表格信息; 记录缓存参数表格信息;
        self.op_data=[] # 记录优化器表格信息 
        self.neglect_id,self.param_ptr=[],[] # neglect_id: 记录不显示的共享参数id; param_ptr:存储参数的指针，避免同内存地址的参数被重复计算
        
        self.reg_num,self.total_num=0,0  # 记录requires_grad参数量; 记录总参数量
        self.total_macs,self.total_flops=0,0
        self.feat_memory,self.param_memory,self.buffer_memory,self.op_param_memory,self.state_memory=0,0,0,0,0
        self.op_param_num,self.shared_num,self.state_num=0,0,0

    def draw_table(self,fields,rows_data):
        tb=ptb.PrettyTable(fields)
        tb.set_style(self.tb_style)
        tb.add_rows(rows_data)
        return tb  

    def unfold_layer(self,module,loc_state:str): # 递归展开module
        '''
        loc_state: module的定位名称, 写入表格中的Module一列
        '''
        sub_modules=module._modules
        modules_dict={}

        if sub_modules:
            for name,sub_module in sub_modules.items():
                res=self.unfold_layer(sub_module,loc_state+f'.{name}')
                modules_dict.update(res)
        else:
            return {str(id(module)):(loc_state,module)}  # 以module的内存地址作为键名，用于前向传播时hook函数调取当前层的名称
        return modules_dict 

    def regist_hook(self,module):
        if isinstance(module,(nn.Conv1d,nn.Conv2d,nn.Conv3d)):
            h=module.register_forward_hook(self.conv_hook)
        elif isinstance(module,(nn.Sigmoid,nn.Tanh,nn.ReLU,nn.ReLU6,nn.SiLU,nn.PReLU,nn.RReLU,nn.LeakyReLU)):
            h=module.register_forward_hook(self.activate_hook)
        elif isinstance(module,(nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)):
            h=module.register_forward_hook(self.BN_hook)
        elif isinstance(module,nn.Linear):
            h=module.register_forward_hook(self.linear_hook)
        elif isinstance(module,(nn.MaxPool1d,nn.AvgPool1d,nn.MaxPool2d,nn.AvgPool2d,nn.MaxPool3d,nn.AvgPool3d)):
            h=module.register_forward_hook(self.pool_hook)
        else:
            h=module.register_forward_hook(self.not_support_hook)
        return h

# modules' FLOPs & MACs calculation principle 

    def conv_hook(self,module,input,output):
        c_in=input[0].shape[1]
        n = c_in * reduce(mul, module.kernel_size)
        m = output.numel()
        is_bias=1 if module.bias is not None else 0

        FLOPs=m*(2*n-1+is_bias)
        MACs=m*n
        self.total_macs+=MACs
        self.total_flops+=FLOPs

        module_name=self.modules_dict[str(id(module))][0]
        self.FandM_data.append([module_name,list(module.kernel_size),bool(is_bias),list(output[0].shape),MACs,FLOPs])
    
    def linear_hook(self,module,input,output):
        k=module.in_features
        l=module.out_features
        is_bias=1 if module.bias is not None else 0
        n=k

        FLOPs=l*(2*n-1+is_bias)
        MACs=l*n
        self.total_macs+=MACs
        self.total_flops+=FLOPs

        module_name=self.modules_dict[str(id(module))][0]
        self.FandM_data.append([module_name,'-',bool(is_bias),list(output[0].shape),MACs,FLOPs])

    def BN_hook(self,module,input,output):
        FLOPs=4*input[0].numel()
        MACs=0.5*FLOPs
        self.total_macs+=MACs
        self.total_flops+=FLOPs

        module_name=self.modules_dict[str(id(module))][0]
        self.FandM_data.append([module_name,'-','-',list(output[0].shape),MACs,FLOPs])

    def activate_hook(self,module,input,output):
        k=input[0].numel()
        if isinstance(module,(nn.Sigmoid,nn.PReLU,nn.RReLU,nn.LeakyReLU)):
            FLOPs=4*k
            MACs=2*k
        elif isinstance(module,(nn.Tanh)):
            FLOPs=9*k
            MACs=5*k
        elif isinstance(module,(nn.ReLU,nn.ReLU6)):
            FLOPs=k
            MACs=k
        else:
            FLOPs=5*k
            MACs=3*k

        self.total_macs+=MACs
        self.total_flops+=FLOPs

        module_name=self.modules_dict[str(id(module))][0]
        self.FandM_data.append([module_name,'-','-',list(output[0].shape),MACs,FLOPs])

    def pool_hook(self,module,input,output):
        k = module.kernel_size
        k = (k,) if isinstance(k,int) else k
        n = reduce(mul, k)-1
        m = output.numel()

        if isinstance(module,(nn.MaxPool1d,nn.MaxPool2d,nn.MaxPool3d)):
            FLOPs=n*m
        else:
            FLOPs=(2*n+1)*m
        MACs=n*m

        self.total_macs+=MACs
        self.total_flops+=FLOPs

        module_name=self.modules_dict[str(id(module))][0]
        self.FandM_data.append([module_name,list(k),'-',list(output[0].shape),MACs,FLOPs])

    def not_support_hook(self,module,input,output):
        module_name=self.modules_dict[str(id(module))][0]
        self.FandM_data.append([module_name,'-','-',list(output[0].shape),'Not Supported','Not Supported'])

#if __name__=='__main__':
#    from torchvision import models
#    import torch.optim as optim
#
#    model=models.alexnet()
#    op=optim.Adam(model.parameters(),lr=1e-4)
#
#    model_evaluator=Evaluator(model,(1,3,224,224))
#    model_evaluator.count_params()
#    input()
#    model_evaluator.count_memory(op)
#    input()
#    model_evaluator.cal_FandM()
#    input()
#    model_evaluator.cal_ITTP('cpu')
