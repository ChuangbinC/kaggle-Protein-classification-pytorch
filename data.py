import random
import numpy as np # linear algebra
from PIL import Image
import torch
import os
from torch.utils.data import Dataset

random.seed(32)
np.random.seed(32)
torch.manual_seed(32)

class CellsDataset(Dataset):
    def __init__(self,X,y=None,transforms=None, nb_organelle=28,augument = False):
        self.nb_organelle = nb_organelle
        self.transforms = transforms
        self.X = X
        self.y = y
        self.augument = augument
    def read_rgby(self,image_path):
        Id = image_path.split('/')[-1].split('_')[0]
        path_dir = '/'.join(image_path.split('/')[:-1])
        
        images = np.zeros(shape=(512,512,4))
        colors = ['red','green','blue','yellow']
        for i,c in enumerate(colors):
            images[:,:,i]+= np.asarray(Image.open(path_dir + '/' + Id + '_' + c + ".png"))
            

        return images.astype(np.uint8)
        
    def __getitem__(self,index):
        path2img = self.X[index]
        image = self.read_rgby(path2img)
        
        if self.y is None:
            label = np.zeros(self.nb_organelle,dtype=np.int)
        else:
            label = np.eye(self.nb_organelle,dtype=np.float)[self.y[index]].sum(axis=0)
            
            
        if self.transforms:
            image = self.transforms(image)
        
        return image,label
    
    
    def __len__(self):
        return len(self.X)
