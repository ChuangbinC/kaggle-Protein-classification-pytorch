# coding: utf-8
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import time
from PIL import Image
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
from models.model import*
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from timeit import default_timer as timer
from tqdm import tqdm 
import os
from config import mconfig
from data import CellsDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'])
args = parser.parse_args()

random.seed(32)
random.seed(32)
np.random.seed(32)
torch.manual_seed(32)

# define data augment
class Rotate(object):
    def __init__(self,angle):
        self.angle = angle
    def __call__(self, img):
        return transforms.functional.rotate(img,self.angle)


def create_dataloader(data_sets,batch_size,shuffle=False,pin_memory=False):
    
    dataloader_sets = {}
    
    for key in data_sets.keys():
        if key == 'test':
            dataloader_sets[key] = DataLoader(data_sets[key],batch_size=1,pin_memory=True,shuffle=False,num_workers=4)
        else:
            dataloader_sets[key] = DataLoader(data_sets[key],batch_size=batch_size,pin_memory=pin_memory,shuffle=True)
    return dataloader_sets



def train(model,dataloader,criterion,optimizer,epoch,valid_loss,best_results,start,writer):
    losses = AverageMeter()
    f1 = AverageMeter()
    model.train()
    all_step = epoch*len(dataloader)
    for i,(images,labels) in enumerate(dataloader):
        images = images.cuda(non_blocking=True)
        labels = labels.float().cuda(non_blocking=True)
        output = model(images)
        
        loss = criterion(output,labels)
        losses.update(loss.item(),images.size(0))
        
        f1_btach = f1_score(labels,output.sigmoid().cpu() > 0.15,average='macro')
        f1.update(f1_btach,images.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('\r',end='',flush=True)
        message = '%s : progress rate:%3.1f in %3.1d epoch | losses is %0.3f  f1 score is %0.3f |valid loss is %0.3f  %0.4f | bestresults is %s  %s | time: %s' % (
                "train", i/len(dataloader) , epoch,
                losses.avg, f1.avg, 
                valid_loss[0], valid_loss[1], 
                str(best_results[0])[:8],str(best_results[1])[:8],
                time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
        writer.add_scalars('train_loss', {'train_loss': losses.avg}, all_step+i)
        writer.add_scalars('train_f1_score', {'train_f1_score': f1.avg},all_step+i)
        writer.add_scalars('optimizer LR', {'optimizer LR': optimizer.param_groups[0]['lr']},all_step+i)
    return [losses.avg,f1.avg]



def evaluate(model,dataloader,criterion,epoch,train_loss,best_results,start,writer):
    losses = AverageMeter()
    f1 = AverageMeter()
    model.cuda()
    model.eval()
    all_step = epoch*len(dataloader)
    with torch.no_grad():
        for i,(images,labels) in enumerate(dataloader):
            images = images.cuda(non_blocking=True)
            labels = labels.float().cuda(non_blocking=True)
            
            output = model(images)
            loss = criterion(output,labels)
            losses.update(loss.item(),images.size(0))
        
            f1_btach = f1_score(labels,output.sigmoid().cpu()>0.15,average='macro')
            f1.update(f1_btach,images.size(0))
            print('\r',end='',flush=True)
            message = '%s : progress rate:%5.1f in %6.1f epoch | losses is %0.3f  f1 score is %0.3f |valid loss is %0.3f  %0.4f | bestresults is %s  %s | time: %s' % (
                    "valid", i/len(dataloader), epoch,                    
                    train_loss[0], train_loss[1], 
                    losses.avg, f1.avg,
                    str(best_results[0])[:8],str(best_results[1])[:8],
                    time_to_str((timer() - start),'min'))
            print(message, end='',flush=True)
            writer.add_scalars('vaild_loss', {'vaild_loss': losses.avg}, all_step+i)
            writer.add_scalars('vaild_f1_score', {'vaild_f1_score': f1.avg}, all_step+i)
    return [losses.avg,f1.avg]

def test(test_loader,model,checkpointname):
    sample_submission_df = pd.read_csv("./data/sample_submission.csv")
    labels ,submissions= [],[]

    model.eval()
    submit_results = []
    for i,(input,_) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            image_var = input.cuda(non_blocking=True)
            y_pred = model(image_var)
            label = y_pred.sigmoid().cpu().data.numpy()
           
            labels.append(label > 0.15)

    for row in np.concatenate(labels):
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('./submit/%s_submission.csv'%checkpointname, index=None)


def main():
    use_gpu = torch.cuda.device_count()
    base_dir = mconfig.base_dir
    train_image_dir = os.path.join(base_dir,'train')
    test_image_dir = os.path.join(base_dir,'test')

    label_train = pd.read_csv(os.path.join(base_dir,'train.csv'))
    label_test = pd.read_csv(os.path.join(base_dir,'sample_submission.csv'))

    label_train['path'] = label_train['Id'].map(lambda x:os.path.join(train_image_dir,'{}_green.png'.format(x)))
    label_train['target_list'] = label_train['Target'].map(lambda x:[int(a) for a in x.split(' ')])
    label_test['path'] = label_test['Id'].map(lambda x: os.path.join(test_image_dir, '{}_green.png'.format(x)))
    
    X = label_train['path'].values
    Y = label_train['target_list'].values
    X_test = label_test['path']

    traindata_mean = np.array([0.08069, 0.05258, 0.05487])
    traindata_std  = np.array([0.13704, 0.10145, 0.15313])

    testdata_mean = np.array([0.05913, 0.0454 , 0.04066])
    testdata_std  = np.array([0.11734, 0.09503, 0.129])


    data_transforms = {
        'train':transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomChoice([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            Rotate(90),
            Rotate(180),
            Rotate(270),
            transforms.RandomAffine(0,shear=16),
            ]),
            transforms.ToTensor(),
            transforms.Normalize(traindata_mean,traindata_std),
            ]),
        'test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(testdata_mean, testdata_std),
            ]),
        }

    batch_size = mconfig.batch_size
    valid_size = mconfig.valid_size
    shuffle = True
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=valid_size, random_state=512)
    
    data_sets = {
        'train': CellsDataset(X_train,y_train,data_transforms['train'],augument= False),
        'valid': CellsDataset(X_valid,y_valid,data_transforms['test']),
        'test': CellsDataset(X_test,None,data_transforms['test']),
    }

    dataloader_sets = create_dataloader(data_sets, batch_size, shuffle, pin_memory=True)
    model = get_net()
    if use_gpu:
        model = torch.nn.DataParallel(model)
        model.cuda()
    else:
        print('using CPU!')

    if args.mode == 'test':
        checkpointname = mconfig.test_checkpoint_name
        best_model = torch.load(os.path.join(mconfig.checkpoints,checkpointname))
        model.load_state_dict(best_model["state_dict"])
        test(dataloader_sets['test'], model, checkpointname.split('-')[1])
    else:

        from tensorboardX import SummaryWriter
        from datetime import datetime
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        train_log_dir = os.path.join(mconfig.logs,TIMESTAMP) 
        writer = SummaryWriter(train_log_dir)

        optimizer = optim.SGD(model.parameters(),lr = mconfig.lr,momentum=0.9,weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss().cuda()

        best_results = [np.inf,0]
        val_metrics = [np.inf,0]
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
        start = timer()
        for epoch in range(0,mconfig.epoch):
            scheduler.step(epoch)
            train_metrics = train(model,dataloader_sets['train'],criterion,optimizer,epoch,val_metrics,best_results,start,writer)
            print('\r')
            val_metrics = evaluate(model,dataloader_sets['valid'],criterion,epoch,train_metrics,best_results,start,writer)
            is_best_loss = val_metrics[0] < best_results[0]
            best_results[0] = min(val_metrics[0],best_results[0])
            is_best_f1 = val_metrics[1] > best_results[1]
            best_results[1] = max(val_metrics[1],best_results[1])
            if is_best_loss and is_best_f1:
                torch.save({'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_loss': best_results[0],
                            'optimizer': optimizer.state_dict()},
                            './checkpoints/'+ 'm-' + "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now()) + '-' + str("%.4f" % best_results[0]) + '.pth.tar')
        print('\r')
        print('train finnish !!!')
        
if __name__ == "__main__":
    main()
