import os
import numpy as np
from torch.utils.data import DataLoader,Dataset
from itertools import permutations

class WAXD_dataset(Dataset):
    def __init__(self,mode,saved_path):
        assert mode in ['train', 'test','valid'], "mode is 'train' or 'test'"
        
        ##If you need this data and the trained and saved model, please download it at XXXXX and place it in the npy_img file 
        train_input1_path=os.path.join(saved_path,'814noise1.npy')
        train_input2_path=os.path.join(saved_path,'814noise2.npy')
        train_input3_path=os.path.join(saved_path,'814noise3.npy')

        self.mode=mode

        if self.mode=='train':
            
            noise1=np.load(train_input1_path)
            noise2=np.load(train_input2_path)
            noise3=np.load(train_input3_path)

            # 生成数据的排列
            data_permutations = list(permutations([noise1, noise2, noise3], 2))

            # 将排列配对为新的input和target数据
            paired_data = [(input_data, target_data) for input_data, target_data in data_permutations]

            # 将所有input数据在axis 0 cat在一起
            self.input_ = np.concatenate([pair[0] for pair in paired_data], axis=0)
            print('after augment, data shape is {}' .format(self.input_.shape[0]))

            # 将所有target数据在axis 0 cat在一起
            self.target_ = np.concatenate([pair[1] for pair in paired_data], axis=0)


        else: # self.mode =='test'
            self.input1_=np.load(train_input1_path)
            self.input2_=np.load(train_input2_path)
            self.input3_=np.load(train_input3_path)


    def __len__(self):
        if self.mode=='train':
            return self.input_.shape[0]
        else: # self.mode =='test'
            return self.input1_.shape[0]

    def __getitem__(self, idx):

        if self.mode=='train':
            input_img=self.input_[idx]
            target_img=self.target_[idx]
            return input_img,target_img
        else: # self.mode =='test'
            input1_img=self.input1_[idx]
            input2_img=self.input2_[idx]
            input3_img=self.input3_[idx]
            output = np.stack([input1_img,input2_img,input3_img],axis=0)
        return output


        
def get_loader(mode='train',saved_path=None,batch_size=32):
    dataset_=WAXD_dataset(mode,saved_path)
    dataloader=DataLoader(dataset=dataset_,batch_size=batch_size,shuffle=(True if mode=='train' else False))
    return dataloader

