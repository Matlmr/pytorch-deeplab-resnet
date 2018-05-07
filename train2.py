#from PyQt5 import QtCore, QtGui, QtWidgets

import torch, cv2
import torch.nn as nn
import numpy as np
#import pickle
import deeplab_resnet2 
#import cv2
from torch.autograd import Variable
import torch.optim as optim
#import scipy.misc
import torch.backends.cudnn as cudnn
#import sys
import os
#import matplotlib.pyplot as plt
from tqdm import *
import random
from docopt import docopt
import timeit
import torchvision.models as models
from torch.nn.functional import upsample

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

start = timeit.timeit
docstr = """Train ResNet-DeepLab on VOC12 (scenes) in pytorch using MSCOCO pretrained initialization 

Usage: 
    train.py [options]

Options:
    -h, --help                  Print this message
    --GTpath=<str>              Ground truth path prefix [default: data/gt/]
    --IMpath=<str>              Sketch images path prefix [default: data/img/]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 21]
    --LISTpath=<str>            Input image number list file [default: data/list/train_aug.txt]
    --lr=<float>                Learning Rate [default: 0.00025]
    -i, --iterSize=<int>        Num iters to accumulate gradients over [default: 10]
    --wtDecay=<float>           Weight decay during training [default: 0.0005]
    --gpu0=<int>                GPU number [default: 0]
    --maxIter=<int>             Maximum number of iterations [default: 20000]
    --savePath=<str>            Path to save network
    --PSPNet                    Use the Pyramid Scene Parsing network
    --savedDict=<str>           Path to pretrained network [default : data/MS_DeepLab_resnet_pretrained_COCO_init.pth]
    --GPUnormal                 Use GPUs without CUDA_VISIBLE_DEVICES
    --101                       Use Resnet101 instead of pretrained net
    --coco                      Use data from COCO
    --batchSize=<int>           The batch size used [default: 2]
"""

#    -b, --batchSize=<int>       num sample per batch [default: 1] currently only batch size of 1 is implemented, arbitrary batch size to be implemented soon
args = docopt(docstr, version='v0.1')
print(args)

cudnn.enabled = False
gpu0 = int(args['--gpu0'])
if args['--GPUnormal'] and gpu0 >= 0:
    torch.cuda.set_device(gpu0)

filename = args['--savePath']

def outS(i):
    """Given shape of input image as i,i,3 in deeplab-resnet model, this function
    returns j such that the shape of output blob of is j,j,21 (21 in case of VOC)"""
    j = int(i)
    j = (j+1)/2
    j = int(np.ceil((j+1)/2.0))
    j = (j+1)/2
    return j

def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list

def chunker(seq, size):
 return (seq[pos:pos+size] for pos in xrange(0,len(seq), size))

def resize_label_batch(label, size):
    label_resized = np.zeros((size,size,1,label.shape[3]))
    interp = nn.Upsample(size=(size, size),mode='bilinear')
    labelVar = Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)

    return label_resized

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def scale_im(img_temp,scale):
    new_dims = (  int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale)   )
    return cv2.resize(img_temp,new_dims).astype(float)

def scale_gt(img_temp,scale):
    new_dims = (  int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale)   )
    return cv2.resize(img_temp,new_dims,interpolation = cv2.INTER_NEAREST).astype(float)
   
def get_data_from_chunk_v2(chunk):
    gt_path =  args['--GTpath']
    img_path = args['--IMpath']
    max_im_size = int(512/1.3)
    scale = random.uniform(0.5, 1.3) #random.uniform(0.5,1.5) does not fit in a Titan X with the present version of pytorch, so we random scaling in the range (0.5,1.3), different than caffe implementation in that caffe used only 4 fixed scales. Refer to read me
    dim = int(scale*max_im_size)
    images = np.zeros((dim,dim,3,len(chunk)))
    gt = np.zeros((dim,dim,1,len(chunk)))
    for i,piece in enumerate(chunk):
        flip_p = random.uniform(0, 1)
        if 'simulant' in piece or args['--coco']:
            img_temp = cv2.imread(os.path.join(img_path,piece+'.png')).astype(float)
        else:
            img_temp = cv2.imread(os.path.join(img_path,piece+'.jpg')).astype(float)
        img_temp = cv2.resize(img_temp,(max_im_size,max_im_size)).astype(float)
        img_temp = scale_im(img_temp,scale)
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        img_temp = flip(img_temp,flip_p)
        images[:,:,:,i] = img_temp

        gt_temp = cv2.imread(os.path.join(gt_path,piece+'.png'))[:,:,0]
        if 'simulant' in piece:
            gt_temp[gt_temp != 0] = 1
        elif args['--coco']:
            gt_temp[gt_temp == 255] = 1
        else:
            gt_temp[gt_temp == 255] = 0
            if int(args['--NoLabels']) == 2:
                gt_temp[gt_temp != 15] = 0
                gt_temp[gt_temp == 15] = 1
        #print(gt_temp)        
        gt_temp = cv2.resize(gt_temp,(max_im_size,max_im_size) , interpolation = cv2.INTER_NEAREST)
        gt_temp = scale_gt(gt_temp,scale)
        gt_temp = flip(gt_temp,flip_p)
        gt[:,:,0,i] = gt_temp
    a = outS(max_im_size*scale)#41
    b = outS((max_im_size*0.5)*scale+1)#21
    labels = [resize_label_batch(gt,i) for i in [a,a,b,a]]
    images = images.transpose((3,2,0,1))
    gts = gt.transpose((3,2,0,1))
    images = torch.from_numpy(images).float()
    gts = torch.from_numpy(gts).float()
    return images, labels, gts

def loss_calc(out, label,gpu0):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label[:,:,0,:].transpose(2,0,1)
    label = torch.from_numpy(label).long()
    if args['--GPUnormal']:
        label = Variable(label).cuda(gpu0)
    elif gpu0 >= 0:
        label = Variable(label).cuda()
    else:
        label = Variable(label)
    m = nn.LogSoftmax()
    criterion = nn.NLLLoss2d()
    out = m(out)
    return criterion(out,label)

def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True, void_pixels=None):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    size_average: return per-element (pixel) average loss
    batch_average: return per-batch average loss
    void_pixels: pixels to ignore from the loss
    Returns:
    Tensor that evaluates the loss
    """
    assert(output.size() == label.size())

    labels = torch.ge(label, 0.5).float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos_pix = -torch.mul(labels, loss_val)
    loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

    if void_pixels is not None:
        w_void = torch.le(void_pixels, 0.5).float()
        loss_pos_pix = torch.mul(w_void, loss_pos_pix)
        loss_neg_pix = torch.mul(w_void, loss_neg_pix)
        num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()

    loss_pos = torch.sum(loss_pos_pix)
    loss_neg = torch.sum(loss_neg_pix)

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss


def lr_poly(base_lr, iter,max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)

    
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """

    b = []
    b.append(model.Scale.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

if not os.path.exists('data/snapshots'):
    os.makedirs('data/snapshots')


model = deeplab_resnet2.Res_Deeplab(int(args['--NoLabels']),args['--PSPNet'])

if args['--101']:
    if args['--GPUnormal']:
        saved_state_dict = torch.load('data/resnet101_imagenet.pth')
    else:
        saved_state_dict = torch.load('data/resnet101_imagenet.pth', map_location=lambda storage, loc: storage)
    model_full = models.resnet101()
    model_full.load_state_dict(saved_state_dict)
    model.load_pretrained_ms(model_full)
else:

    if not args['--savedDict']:
        if args['--GPUnormal']:
            saved_state_dict = torch.load('data/MS_DeepLab_resnet_pretrained_COCO_init.pth')
        else:
            saved_state_dict = torch.load('data/MS_DeepLab_resnet_pretrained_COCO_init.pth', map_location=lambda storage, loc: storage)
    else:
        if args['--GPUnormal']:
            saved_state_dict = torch.load(args['--savedDict'])
        else:
            saved_state_dict = torch.load(args['--savedDict'], map_location=lambda storage, loc: storage)

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}

    if int(args['--NoLabels'])!=21:
        for i in saved_state_dict:
            #Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            if i_parts[1]=='layer5':
                saved_state_dict[i] = model.state_dict()[i]

    # 2. overwrite entries in the existing state dict
    model_dict.update(saved_state_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

max_iter = int(args['--maxIter']) 
batch_size = int(args['--batchSize'])
weight_decay = float(args['--wtDecay'])
base_lr = float(args['--lr'])

model.float()
model.train() # use_global_stats = True

img_list = read_file(args['--LISTpath'])

data_list = []
for i in range(70):  # make list for 10 epocs, though we will only use the first max_iter*batch_size entries of this list
    np.random.shuffle(img_list)
    data_list.extend(img_list)
print('before')
if args['--GPUnormal']:
    model.cuda(gpu0)
elif gpu0 >= 0:
    model.cuda()
print('after')
criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': base_lr }, {'params': get_10x_lr_params(model), 'lr': 10*base_lr} ], lr = base_lr, momentum = 0.9,weight_decay = weight_decay)

optimizer.zero_grad()
data_gen = chunker(data_list, batch_size)

for iter in range(max_iter+1):
    
    chunk = data_gen.next()

    images, label, gts = get_data_from_chunk_v2(chunk)
    if args['--GPUnormal']:
        images = Variable(images).cuda(gpu0)
    elif gpu0 >= 0:
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
    else:
        images = Variable(images)
        gts = Variable(gts)

    if gpu0 >= 0:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    out = model(images)
    output = upsample(out[0], size=(images.shape[2], images.shape[3]), mode='bilinear')
    #loss = loss_calc(out[0], label[0],gpu0)
    loss = class_balanced_cross_entropy_loss(output, gts, size_average=True, batch_average=True)
    iter_size = int(args['--iterSize'])
    # for i in range(len(out)-1):
    #     loss = loss + loss_calc(out[i+1],label[i+1],gpu0)
    loss = loss/iter_size 
    loss.backward()

    if iter %1 == 0:
        print 'iter = ',iter, 'of',max_iter,'completed, loss = ', iter_size*(loss.data.cpu().numpy())

    if iter % iter_size == 0:
        optimizer.step()
        lr_ = base_lr  # lr_poly(base_lr,iter,max_iter,0.9)
        print '(poly lr policy) learning rate',lr_
        # optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_ }, {'params': get_10x_lr_params(model), 'lr': 10*lr_} ], lr = lr_, momentum = 0.9,weight_decay = weight_decay)
        optimizer.zero_grad()

    if iter % 10000 == 0 and iter!=0:
        print 'taking snapshot ...'
        #torch.save(model.state_dict(),'data/snapshots/VOC12_scenes_'+str(iter)+'.pth')
        torch.save(model.state_dict(),'data/snapshots/'+filename+'/'+str(iter)+'.pth')
end = timeit.timeit
print end-start,'seconds'
