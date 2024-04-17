import cv2,os
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

def explore_seqs(seq_root_path):
    '''
    get all seqs dir according to the seq_type
    '''
    seqs_list=[]
    for root,dir,files in os.walk(seq_root_path):
        if files:
            seqs_list.append(root)

    assert len(seqs_list)>0, f'No seqs found in {seq_root_path}!'
    return seqs_list

def frame_cog(img:np.ndarray):
    y_list, x_list =np.where(img!=0) # (row_idx,col_idx)
    y_cog , x_cog = y_list.mean(), x_list.mean()
    return (x_cog,y_cog)

def get_seq_cogs(seq_dir, source_type='img'):
    seq_files=glob(seq_dir+'/*')
    seq_files.sort()
    if source_type=='pkl':
        with open(seq_files,'rb') as pkl_data:
            seq_data=pk.load(pkl_data)
    else:
        seq_data=list(map(lambda x:cv2.imread(x,flags=cv2.IMREAD_GRAYSCALE),seq_files))    # t,64,64
        seq_data=np.array(seq_data)

    coords=np.array(list(map(frame_cog,seq_data)))
    x = coords[:, 0]
    y = coords[:, 1]
    frame_hw=seq_data.shape[1:]
    
    return (seq_data,coords,x,y,frame_hw)

def align_data(array1,array2):
    len_1=array1.shape[-1]
    len_2=len(array2)
    if len_1 > len_2:
        array2=np.append(array2, [np.nan]*(len_1 - len_2))
    elif len_1 < len_2:
        if array1.ndim==2:
            array1=np.c_[array1, np.array([np.nan]*(len_2 - len_1)*2).reshape(2,-1)]
        else:
            array1=np.append(array1, [np.nan]*(len_2 - len_1))
    else:
        pass
    
    if array1.ndim==2:
        return np.concatenate([array1,array2.reshape(1,-1)],0)
    else:
        return np.stack([array1,array2],0)

def plt_plot_2D(data_type, set_order, stack_data, data_meaning, save_dir):
    '''
    data_type: 'origin' / 'scale'
    set_order: e.g. ['casia','cow']
    stack_data: stack_x / stack_y
    data_meaning: 'y'/'x'
    save_dir
    '''
    x_range=stack_data.shape[-1]
    plt.figure(figsize=(x_range*0.2,5))
    plt.grid(True,linestyle='-', linewidth=0.5, color='gray', zorder=0)
    if data_type=='scale':
        plt.ylim(0,1)
    
    for idx,set_name in enumerate(set_order):
        data=stack_data[idx] if len(set_order)>1 else stack_data
        plt.scatter(range(x_range), data, s=10)
        plt.plot(range(x_range), data, '-', label=set_name)

    plt.xticks(range(0,x_range+1),rotation=90,fontsize=5)
    plt.xlim(-1,x_range)
    y_meaning='Y' if data_meaning=='y' else 'X'
    plt.xlabel('frame')
    plt.ylabel(y_meaning + ' coordinate of cog' )
    if len(set_order)>1:
        plt.legend()

    plt.tight_layout()
    save_name=f'COG_{y_meaning}.png' if data_type!='scale' else f'COG_{y_meaning}_SCALE.png'
    plt.savefig(os.path.join(save_dir,save_name),dpi=300,bbox_inches='tight')
    plt.close()

def plt_plot_3D(data_type, set_order, xy_dict, save_dir):
    '''
    xy_dict={'x':stack_x,
             'y':stack_y}
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if data_type=='scale':
        ax.set_xlim(0,1)
        ax.set_zlim(0,1)

    frame_list = np.arange(xy_dict['x'].shape[-1])
    for idx,set_name in enumerate(set_order):
        x_data=xy_dict['x'][idx] if len(set_order)>1 else xy_dict['x']
        y_data=xy_dict['y'][idx] if len(set_order)>1 else xy_dict['x']
        ax.scatter(x_data, frame_list, y_data,s=10)
        ax.plot(x_data,frame_list,y_data,linewidth=1,label=set_name)
    
    ax.set_xlabel('x coordinate of cog')
    ax.set_zlabel('y coordinate of cog')
    ax.set_ylabel('frame')
    if len(set_order)>1:
        ax.legend()

    plt.tight_layout()
    save_name='COG_Fluctuation_Plot.png' if data_type!='scale' else 'COG_Fluctuation_Plot_SCALE.png'
    plt.savefig(os.path.join(save_dir,save_name),dpi=300,bbox_inches='tight')
    #plt.show()
    plt.close()

def draw_cog_line(set_seqs, save_dir):
    '''
    set_seqs={'casia':(seq_data, # frame,h,w
                       coords, # frame,2
                       x, # frame,1
                       y, # frame,1
                       frame_hw), # (h,w)
              'cow':(seq_data,coords,x,y,frame_hw)}
    '''
    
    for set_name,set_seq in set_seqs.items():
        imgs,coords,x,y,frame_hw=set_seq
        assert len(imgs)==len(coords),'Some imgs miss the center of gravity!'
        
        if len(set_seqs)>1:
            set_save_dir = os.path.join(save_dir,set_name) # if multi set,save to each dir respectively
        else:
            set_save_dir = save_dir
        if not os.path.exists(set_save_dir):
            os.makedirs(set_save_dir,exist_ok=True)

        for id,img in enumerate(imgs):
            draw_img=img.copy()
            draw_img=cv2.cvtColor(draw_img, cv2.COLOR_GRAY2RGB) # to show color line

            x_cog, y_cog = int(x[id]),int(y[id])
            h,w = frame_hw

            hor_stp=(0,y_cog)
            hor_edp=(w,y_cog)
            ver_stp=(x_cog,0)
            ver_edp=(x_cog,h)

            cv2.line(draw_img, hor_stp, hor_edp, (0,255,0), 1)
            cv2.line(draw_img, ver_stp, ver_edp, (0,255,0), 1)
            cv2.imwrite(os.path.join(set_save_dir,f'{id}.png'),draw_img)

def draw_cog_xy(set_seqs, save_dir, scale=True, draw_3d=False):
    '''
    set_seqs={'casia':(seq_data, # frame,h,w
                       coords, # frame,2
                       x, # frame,1
                       y, # frame,1
                       frame_hw), # (h,w)
              'cow':(seq_data,coords,x,y,frame_hw)}
    '''
    draw_dict={'origin':{},'scale':{}} if scale else {'origin':{}}
    set_order = []
    for set_name,(seq_data,coords,x,y,frame_hw) in set_seqs.items():
        set_order.append(set_name)
        if not draw_dict['origin']:
            draw_dict['origin']['x']=x
            draw_dict['origin']['y']=y
            if scale:
                y_scale=y.copy()/frame_hw[0]
                x_scale=x.copy()/frame_hw[1]
                draw_dict['scale']['x']=x_scale
                draw_dict['scale']['y']=y_scale
        else:
            draw_dict['origin']['x']=align_data(draw_dict['origin']['x'],x)
            draw_dict['origin']['y']=align_data(draw_dict['origin']['y'],y)
            if scale:
                y_scale=y.copy()/frame_hw[0]
                x_scale=x.copy()/frame_hw[1]
                draw_dict['scale']['x']=align_data(draw_dict['scale']['x'],x_scale)
                draw_dict['scale']['y']=align_data(draw_dict['scale']['y'],y_scale)
    
    '''
    draw_dict={'origin':{'x':stack_x,
                         'y':stack_y}, 
               'scale':{'x':stack_scale_x,
                        'y':stack_scale_y}
               }
    '''

    for data_type, xy_dict in draw_dict.items():
        for axis,stack_data in xy_dict.items():
            plt_plot_2D(data_type, set_order, stack_data, data_meaning=axis, save_dir=save_dir)
        if draw_3d:
            plt_plot_3D(data_type, set_order, xy_dict, save_dir=save_dir)

def merge_type_xy(xy_info, save_dir, average = True, scale=True):
    '''
    xy_info:
    {'001':{'000':[(type, cog_x, cog_y, frame_hw),
                   (type, cog_x, cog_y, frame_hw),...],
            
            '045':[(type, cog_x, cog_y, frame_hw),
                   (type, cog_x, cog_y, frame_hw),...]
            }            
    }
    '''
    person_id = list(xy_info.keys())[0]

    to_draw={}
    for view, type_xy_list in xy_info[person_id].items():
        for (walk_type, cog_x, cog_y, frame_hw) in type_xy_list:
            if view not in to_draw:
                to_draw[view] = [[walk_type[:2]],
                                 cog_x, cog_y,
                                 cog_x/frame_hw[1], cog_y/frame_hw[0]]
            else:
                if walk_type[:2] in to_draw[view][0]:
                    continue
                else:
                    to_draw[view][0].append(walk_type[:2])
                       
                    to_draw[view][1] = align_data(to_draw[view][1],cog_x)
                    to_draw[view][2] = align_data(to_draw[view][2],cog_y)
                    to_draw[view][3] = align_data(to_draw[view][3],cog_x/frame_hw[1])
                    to_draw[view][4] = align_data(to_draw[view][4],cog_y/frame_hw[0])
        
    for view, stack_info in to_draw.items():
        type_list, *stack_datas = stack_info # satck_data: stack_x, stack_y, stack_x_scale, stack_y_scale
        for id, datas in enumerate(stack_datas):
            plt.figure(figsize=(10,5))
            plt.grid(True,linestyle='-', linewidth=0.5, color='gray', zorder=0)
            if id>=2:
                plt.ylim(0,1)

            for type_idx,data in enumerate(datas):
                #plt.scatter(range(len(data)), data,s=10)
                plt.plot(range(len(data)), data,'-',label=type_list[type_idx],alpha=0.5)

            plt.legend()
            plt.xticks(range(0,len(data)+1),rotation=90,fontsize=5)
            plt.xlim(-1,len(data)+1)
            y_meaning='X' if id%2==0 else 'Y'
            plt.xlabel('frame')
            plt.ylabel(y_meaning + ' coordinate of cog' ) 

            plt.tight_layout()
            save_name=f'stack_{y_meaning}_nbc-01.png' if id<2 else f'stack_{y_meaning}_Scale_nbc-01.png'
            save_path = os.path.join(save_dir,person_id,'Merge_by_type', view)
            os.makedirs(save_path,exist_ok=True)
            plt.savefig(os.path.join(save_path, save_name),dpi=300,bbox_inches='tight')
            plt.close()

if __name__ =='__main__':
    set_paths=[r'E:\AIL\Datasets\Gait\Thermal\mask_video\imgs',] # all sets should have same dir structure
               #r'E:\AIL\project\COG\Seqs\CASIAB'] #'/data/mutil-dataset/cow_gait'
    save_root = r'E:\AIL\project\COG\Results\Thermal'
    source_type='img' # all sets should have same item type
    seqs_lists=[]
    to_merge={}

    # pick the set which has least seqs
    for set_path in set_paths:
        seqs_lists.append(explore_seqs(set_path))
    min_set = min(seqs_lists,key=len)

    # draw
    for seq_dir in tqdm(min_set):
        set_name, person_id, walk_type, view = seq_dir.split(os.path.sep)[-4:]  # os.path.split(seq_dir)[-1]
        seq_save_path = os.path.join(save_root, person_id, walk_type, view)
        os.makedirs(seq_save_path, exist_ok=True)

        # get fellow dataset's coresponding seq's data
        set_seqs={set_name:get_seq_cogs(seq_dir, source_type)}
        for set_path in set_paths:
            if set_name in set_path:
                continue
            else:
                set_name = os.path.split(set_path)[-1]
                fellow_seq_dir = os.path.join(set_path, person_id, walk_type, view)
                if os.path.exists(fellow_seq_dir):
                    set_seqs[set_name]=get_seq_cogs(fellow_seq_dir, source_type)

        # draw cog in the source img
        #draw_cog_line(set_seqs, os.path.join(seq_save_path,'frame_cog'))
        # draw 2D x-frame line & 2D y-frame line / 3D x-y-frame line
        draw_cog_xy(set_seqs, seq_save_path, scale=True, draw_3d=False) # (frame_h,frame_w)

        #if person_id not in to_merge:
        #    if list(to_merge.keys()):  # 收集完一人的各序列信息就画
        #        merge_type_xy(to_merge,'./Results/cow')
        #        to_merge={}
        #    to_merge[person_id] = {view:[(walk_type, x.copy(), y.copy(), frame_hw)]}
        #else:
        #    if view not in to_merge[person_id]:
        #        to_merge[person_id][view] = [(walk_type, x.copy(), y.copy(), frame_hw)]
        #    else:
        #        to_merge[person_id][view].append((walk_type, x.copy(), y.copy(), frame_hw))
