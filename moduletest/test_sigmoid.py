import sys
sys.path.insert(0,'.')
import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import caffe

import pytorch_to_caffe

outdir = 'tmp'
name = 'sigmoid'
inputshape = (1, 3, 224, 224)
data = (np.random.random(inputshape) - 0.5) * 20.0
if not os.path.isdir(outdir):
    os.makedirs(outdir)

class sigmoidNet(nn.Module):
    def __init__(self) -> None:
        super(sigmoidNet, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.sigmoid(x)
        return x

def forward_pytorch(weightfile, data):
    input = data.copy()
    model = torch.load(weightfile)
    model.eval()
    input = torch.from_numpy(input)
    input = Variable(input)
    t0 = time.time()
    blobs = model.forward(input)
    blobs = blobs.data.numpy().flatten()
    t1 = time.time()
    return t1-t0, blobs, model.parameters()

def forward_caffe(protofile, weightfile, data):
    input=data.copy()
    caffe.set_mode_cpu()
    net = caffe.Net(protofile, weightfile, caffe.TEST)
    net.blobs['blob1'].reshape(1, 3, 224, 224)
    net.blobs['blob1'].data[...] = input
    t0 = time.time()
    output = net.forward()
    t1 = time.time()
    return t1-t0, net.blobs, net.params

def testsigmoid():
    # save torch model
    net = sigmoidNet()
    torch.save(net, os.path.join(outdir, '{}.pth'.format(name)))

    # torch outputs # 不能先转，再读取模型forward，有bug
    torchtime, torchoutput, torchparam = forward_pytorch(os.path.join(outdir, '{}.pth'.format(name)), data)
    
    # transform caffe model
    input=Variable(torch.ones(inputshape))
    pytorch_to_caffe.trans_net(net,input,name)
    pytorch_to_caffe.save_prototxt(os.path.join(outdir, '{}.prototxt'.format(name)))
    pytorch_to_caffe.save_caffemodel(os.path.join(outdir, '{}.caffemodel'.format(name)))

    # caffe outputs
    caffetime, caffeblobs, caffeparam = forward_caffe(os.path.join(outdir, '{}.prototxt'.format(name)), 
        os.path.join(outdir, '{}.caffemodel'.format(name)), data)
    caffeoutput = caffeblobs['sigmoid_blob1'].data[0][...].flatten()

    # print result
    print('='*100)
    print("inputdata min :   {:.10f}, max : {:.10f}, mean: {:.10f}".format(data.min(), data.max(), data.mean()))
    print("torchoutput min : {:.10f}, max : {:.10f}, mean: {:.10f}".format(torchoutput.min(), torchoutput.max(), torchoutput.mean()))
    print("caffeoutput min : {:.10f}, max : {:.10f}, mean: {:.10f}".format(caffeoutput.min(), caffeoutput.max(), caffeoutput.mean()))
    diff = abs(torchoutput - caffeoutput).sum()
    print('%s torchtime: %f caffetime: %f pytorch_shape: %-10s caffe_shape: %-10s output_mean_diff: %.10f' % ('sigmoid_blob1', torchtime, caffetime, torchoutput.shape, caffeoutput.shape, diff/torchoutput.size))
    if torchoutput.shape==caffeoutput.shape and diff/torchoutput.shape < 1e-5:
        return True
    return False

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--outdir',help='model and result output dir',default = "tmp",type=str)
    args=parser.parse_args()
    outdir=args.outdir
    
    result=testsigmoid()
    with open(os.path.join(outdir, 'test.csv'), 'a') as f:
        f.write('{:12s}  {}\n'.format(name, result))
    print('{:12s}  {}\n'.format(name, result))
