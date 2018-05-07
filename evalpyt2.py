import matplotlib.pyplot as plt
#import scipy
#from scipy import ndimage
from cv2 import imread
import numpy as np
#import sys
#sys.path.insert(0,'/data1/ravikiran/SketchObjPartSegmentation/src/caffe-switch/caffe/python')
#import caffe
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import deeplab_resnet2
from collections import OrderedDict
import os
#from os import walk
import torch.nn as nn

from docopt import docopt


docstr = """Evaluate ResNet-DeepLab trained on scenes (VOC 2012),a total of 21 labels including background

Usage: 
    evalpyt.py [options]

Options:
    -h, --help                  Print this message
    --visualize                 view outputs of each sketch
    --snapPrefix=<str>          Snapshot [default: VOC12_scenes_]
    --testGTpath=<str>          Ground truth path prefix [default: data/gt/]
    --testIMpath=<str>          Sketch images path prefix [default: data/img/]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 21]
    --gpu0=<int>                GPU number [default: 0] -1 for cpu mode
    --PSPNet                    Use the Pyramid Scene Parsing network
    --LISTpath=<str>            Input image number list file [default: data/list/val.txt]
    --coco                      Use the COCO dataset
    --GPUnormal                 Use GPUs without CUDA_VISIBLE_DEVICES
"""

args = docopt(docstr, version='v0.1')
print args

torch.backends.cudnn.enabled = False
gpu0 = int(args['--gpu0'])
if gpu0 >= 0 and args['--GPUnormal']:
    torch.cuda.set_device(gpu0)

max_label = int(args['--NoLabels'])-1 # labels from 0,1, ... 20(for VOC) 
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def get_iou(pred,gt):
    if pred.shape!= gt.shape:
        print 'pred shape',pred.shape, 'gt shape', gt.shape
    assert(pred.shape == gt.shape)    
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    count = np.zeros((max_label+1,))
    for j in range(max_label+1):
        x = np.where(pred==j)
        p_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        x = np.where(gt==j)
        GT_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        #pdb.set_trace()     
        n_jj = set.intersection(p_idx_j,GT_idx_j)
        u_jj = set.union(p_idx_j,GT_idx_j)
    
        
        if len(GT_idx_j)!=0:
            count[j] = float(len(n_jj))/float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:])/float(len(np.unique(gt))) 
    
    return Aiou

im_path = args['--testIMpath']
model = deeplab_resnet2.Res_Deeplab(int(args['--NoLabels']),args['--PSPNet'])

counter = 0
print('before')
if gpu0 >= 0:
    if args['--GPUnormal']:
        model.cuda(gpu0)
    else:
        model.cuda()
print('after')

model.eval()

snapPrefix = args['--snapPrefix']
gt_path = args['--testGTpath']
if not args['--LISTpath']:
    img_list = open('data/list/val.txt').readlines()
else:
    img_list = open(args['--LISTpath']).readlines()

for iter in (10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200):   #TODO set the (different iteration)models that you want to evaluate on. Models are saved during training after each 1000 iters by default.
    if gpu0 >= 0 and args['--GPUnormal']:
        saved_state_dict = torch.load(os.path.join('data/snapshots/',snapPrefix+str(iter)+'000.pth'))
    else:
        saved_state_dict = torch.load(os.path.join('data/snapshots/',snapPrefix+str(iter)+'000.pth'), map_location=lambda storage, loc: storage)
    if counter==0:
	print snapPrefix
    counter+=1
    model.load_state_dict(saved_state_dict)

    hist = np.zeros((max_label+1, max_label+1))
    pytorch_list = [];
    for i in img_list:
        counter+=1
        #print(counter)
        if args['--coco']:        
            img = np.zeros((640,640,3));
        else:
            img = np.zeros((513,513,3));

        if args['--coco']:
            img_temp = imread(os.path.join(im_path,i[:-1]+'.png')).astype(float)
        else:
            img_temp = imread(os.path.join(im_path,i[:-1]+'.jpg')).astype(float)
        img_original = img_temp
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        img[:img_temp.shape[0],:img_temp.shape[1],:] = img_temp
        gt = imread(os.path.join(gt_path,i[:-1]+'.png'),0)
        
        if args['--coco']:
            gt[gt==255] = 1
        else:
            gt[gt==255] = 0
            if int(args['--NoLabels']) == 2:
                gt[gt != 15] = 0
                gt[gt == 15] = 1

        if gpu0 >= 0:
            if args['--GPUnormal']:
                output = model(Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True).cuda(gpu0))
            else:
                output = model(Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True).cuda())
        else:
            output = model(Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True))
        
        if args['--coco']:
            interp = nn.UpsamplingBilinear2d(size=(640, 640))
        else:
            interp = nn.UpsamplingBilinear2d(size=(513, 513))
        if args['--PSPNet']:
            output = interp(output[0]).cpu().data[0].numpy()
        else:
            output = interp(output[3]).cpu().data[0].numpy()
        output = output[:,:img_temp.shape[0],:img_temp.shape[1]]
        
        output = output.transpose(1,2,0)
        output = np.argmax(output,axis = 2)
        #bck = np.zeros(output.shape,dtype=np.int64)
        #output = bck
        if args['--visualize']:
            plt.subplot(3, 1, 1)
            plt.imshow(img_original)
            plt.subplot(3, 1, 2)
            plt.imshow(gt)
            plt.subplot(3, 1, 3)
            plt.imshow(output)
            plt.show()

        iou_pytorch = get_iou(output,gt)       
        pytorch_list.append(iou_pytorch)
        hist += fast_hist(gt.flatten(),output.flatten(),max_label+1)
    miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print 'pytorch',iter,"Mean iou = ",np.sum(miou)/len(miou)
