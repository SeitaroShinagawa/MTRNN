#!/usr/bin/env python

"""Chainer example: train a multiple timescale recurrent neural network (MTRNN) on MNIST

This is a minimal example to write a feed-forward net. It requires scikit-learn
to load MNIST dataset.

This implementation is based on pfnet/Chainer example feed-forward network


target                   x_1             x_2                          x_t+1
------------------------------------------------------------------------------------
output                   y_0             y_1                          y_t
<internal state>         _|_             _|_                          _|_
slow context Csc_0    ->|   |-> Csc_1 ->|   |-> Csc_2 -> ... Csc_t ->|   |-> Csc_t+1 -> ...
fast context Cfc_0(=0)->|_ _|-> Cfc_1 ->|_ _|-> Cfc_2 -> ... Cfc_t ->|_ _|-> Cfc_t+1 -> ...
                          |               |                            |
input                    x_0             x_1                          x_t

unrolling on time scale ---->
"""
import argparse

import numpy as np
import six

import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import random
import data
import time
from PIL import Image
import pickle

seed_def = 0
batchsize_def = 256
epoch_def = 20
hidden_def = 100
wdecay_def = 0.0
f_hidden_def = 100
s_hidden_def = 30
tau_io_def = 2.0
tau_fh_def = 5.0
tau_sh_def = 70.0
lr_def = 100.0

parser = argparse.ArgumentParser(description='RNN MNIST generation')
parser.add_argument('-S','--seed',default=seed_def, metavar='INT', type=int,
                    help='random seed (default: %d)' % seed_def)
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('-FH', '--fh', default=f_hidden_def, metavar='INT', type=int,
                    help='fast context hidden layer size (default: %d)' % f_hidden_def)
parser.add_argument('-SH', '--sh', default=s_hidden_def, metavar='INT', type=int,
                    help='slow context hidden layer size (default: %d)' % s_hidden_def)
parser.add_argument('-B', '--batchsize', default=batchsize_def, metavar='INT', type=int,
                    help='minibatch size (default: %d)' % batchsize_def)
parser.add_argument('-I', '--epoch', default=epoch_def, metavar='INT', type=int,
                    help='number of training epoch (default: %d)' % epoch_def)
parser.add_argument('-W', '--weightdecay', default=wdecay_def, metavar='FLOAT', type=float,
                    help='weight decay (default: %d)' % wdecay_def)
parser.add_argument('-IO', '--tau_io', default=tau_io_def, metavar='FLOAT', type=float,
                    help='tau_io (default: %f)' % tau_io_def)
parser.add_argument('-TFH', '--tau_fh', default=tau_fh_def, metavar='FLOAT', type=float,
                    help='tau_fh (default: %f)' % tau_fh_def)
parser.add_argument('-TSH', '--tau_sh', default=tau_sh_def, metavar='FLOAT', type=float,
                    help='tau_sh (default: %f)' % tau_sh_def)
parser.add_argument('-L', '--lr', default=lr_def, metavar='FLOAT', type=float,
                    help='lr (default: %f)' % lr_def)

args = parser.parse_args()


print '***Experiment settings***'
print 'ranomseed :',args.seed
print 'fast context hidden layer size :',args.fh
print 'slow context hidden layer size :',args.sh
print 'max epoch :',args.epoch
print 'mini batch size :',args.batchsize
print 'weight decay :',args.weightdecay
print 'Csc0 learning rate :',args.lr
print '*************************'

def random_set(randomseed):
	random.seed(randomseed)
	np.random.seed(randomseed)
	return 0
random_set(args.seed)

if args.gpu >= 0:
    print 'GPU mode'
    cuda.check_cuda_available()
else:
    print 'CPU mode'
xp = cuda.cupy if args.gpu >= 0 else np


# Prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

# Prepare multi-layer perceptron model

def rnn(io_size, fast_hidden_size, slow_hidden_size):
    model = chainer.FunctionSet(x_to_fh=F.Linear(io_size, fast_hidden_size),
                            fh_to_fh=F.Linear(fast_hidden_size, fast_hidden_size),
                            fh_to_sh=F.Linear(fast_hidden_size, slow_hidden_size),
                            sh_to_fh=F.Linear(slow_hidden_size, fast_hidden_size),
                            sh_to_sh=F.Linear(slow_hidden_size, slow_hidden_size),
                            fh_to_y=F.Linear(fast_hidden_size, io_size))
    for param in model.parameters:
        param[:] = np.random.uniform(-0.1, 0.1, param.shape)
    return model

#uncomment to create a new model
model=rnn(28,args.fh,args.sh)

#uncomment to load existed model
#model=pickle.load(open('modelname','r'))

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

def forward_one_step(model, x_data, u_io, u_fh, u_sh, tau_io, tau_fh, tau_sh, train=True):
    #original MTRNN have only sigmoid activation functions
    x = chainer.Variable(x_data)
    fh = F.sigmoid(u_fh)
    #fh = F.tanh(u_fh2)
    sh = F.sigmoid(u_sh)
    #sh = F.tanh(u_sh2)
    y = F.sigmoid(u_io)

    u_io2 = (1-1/tau_io)*u_io+(model.fh_to_y(fh))/tau_io
    u_fh2 = (1-1/tau_fh)*u_fh+(model.x_to_fh(x)+model.fh_to_fh(fh)+model.sh_to_fh(sh))/tau_fh
    u_sh2 = (1-1/tau_sh)*u_sh+(model.fh_to_sh(fh)+model.sh_to_sh(sh))/tau_sh
    return u_io2, u_fh2, u_sh2, y

def make_initial_state(batchsize, n_hidden, train=True):
    return np.array(np.zeros((batchsize, n_hidden)),dtype=np.float32)

#optional
def make_initial_state_random(batchsize, n_hidden, train=True):
    return np.array(np.random.uniform(-1,1,(batchsize, n_hidden)),dtype=np.float32)

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model) 

# Initialize slow context initial internal states (for each data)
Csc0_train = make_initial_state(len(x_train), args.sh)
Csc0_test = make_initial_state(len(x_test), args.sh)


# Learning loop
print 'training start ...'
for epoch in range(1,args.epoch+1):
    print 'epoch:', epoch
    now=time.time()
    cur=now
    err = xp.zeros(())

    perm = np.random.permutation(N) #random order
    #perm = range(N)                #in order

    for i in six.moves.range(0, N, args.batchsize): #i jumps 0->batchsize->2*batchsize->...->k*batchsize < N
        acc_loss = chainer.Variable(xp.zeros((),dtype=np.float32))
        x_batch = xp.asarray(x_train[perm[i:i + args.batchsize]])
        optimizer.zero_grads()
        
        Csc0 = chainer.Variable(xp.asarray(Csc0_train[perm[i:i + args.batchsize]]))     
        u_io = chainer.Variable(xp.asarray(make_initial_state(x_batch.shape[0], 28, train=True)))
        u_fh = chainer.Variable(xp.asarray(make_initial_state(x_batch.shape[0], args.fh, train=True)))
 
        j=0
        u_io, u_fh, u_sh, y= forward_one_step(model, x_batch[:,28*j:28*(j+1)],u_io, u_fh, Csc0, args.tau_io, args.tau_fh, args.tau_sh)
        loss_i = F.mean_squared_error(y,chainer.Variable(x_batch[:,28*(j+1):28*(j+2)]))
        acc_loss += loss_i
        err += loss_i.data.reshape(())*args.batchsize
        for j in range(1,27):
            u_io, u_fh, u_sh, y= forward_one_step(model, x_batch[:,28*j:28*(j+1)],u_io, u_fh, u_sh, args.tau_io, args.tau_fh, args.tau_sh)
            loss_i = F.mean_squared_error(y,chainer.Variable(x_batch[:,28*(j+1):28*(j+2)]))
            acc_loss += loss_i
            err += loss_i.data.reshape(())*args.batchsize
        acc_loss.backward()
        optimizer.update()
        optimizer.weight_decay(args.weightdecay)
        
        Csc0.data -= args.lr*Csc0.grad

        print 'now training ... :',i,'/%s'%(N),'\r',
        #if epoch == 1 and i == 0:
        #    with open("graph.dot", "w") as o:
        #        o.write(c.build_computational_graph((acc_loss, )).dump())
        #    with open("graph.wo_split.dot", "w") as o:
        #        g = c.build_computational_graph((acc_loss, ),
        #                                        remove_split=True)
        #        o.write(g.dump())
        #    print('graph generated')
    print 'Done a eopch.'
    error = err/N/27
    print 'train MSE = %f' %(error)
    now=time.time()
    print 'elapsed time:',now-cur
    
    with open('train.txt','aw') as f:
        print >> f, '%s %s %s' %(epoch, error, now-cur)

    if epoch % 1 == 0:      
        print 'evaluation...'
        now = time.time()
        cur = now
        err = xp.zeros(())
        perm = range(N_test)
        for i in six.moves.range(0, N_test, args.batchsize): #0->batchsize->2*batchsize->...->k*batchsize < N
            acc_loss = chainer.Variable(xp.zeros((),dtype=np.float32))
            x_batch = xp.asarray(x_test[perm[i:i + args.batchsize]])
                
            Csc0 = chainer.Variable(xp.asarray(Csc0_test[perm[i:i + args.batchsize]]))     
            u_io = chainer.Variable(xp.asarray(make_initial_state(x_batch.shape[0], 28, train=True)))
            u_fh = chainer.Variable(xp.asarray(make_initial_state(x_batch.shape[0], args.fh, train=True)))
 
            j=0
            u_io, u_fh, u_sh, y= forward_one_step(model, x_batch[:,28*j:28*(j+1)],u_io, u_fh, Csc0, args.tau_io, args.tau_fh, args.tau_sh)
            loss_i = F.mean_squared_error(y,chainer.Variable(x_batch[:,28*(j+1):28*(j+2)]))
            acc_loss += loss_i
            err += loss_i.data.reshape(())*args.batchsize
            for j in range(1,27):
                u_io, u_fh, u_sh, y= forward_one_step(model, x_batch[:,28*j:28*(j+1)],u_io, u_fh, u_sh, args.tau_io, args.tau_fh, args.tau_sh)
                loss_i = F.mean_squared_error(y,chainer.Variable(x_batch[:,28*(j+1):28*(j+2)]))
                acc_loss += loss_i
                err += loss_i.data.reshape(())*args.batchsize
            acc_loss.backward()
            Csc0.data -= args.lr*Csc0.grad
            
            print 'now evaluating ... :',i,'/%s'%(N_test),'\r',

        print 'Done a evaluation'
        error = err/N_test/27
        print 'evaluation MSE = %f' %(error)
        now=time.time()
        print 'elapsed time:',now-cur
        with open('evaluation.txt','aw') as f:
            print >> f, '%s %s %s' %(epoch, error, now-cur)

pickle.dump(model.to_cpu(),open('mnistMTRNN_model','w'))
pickle.dump(Csc0_train,open('Csc0_train','w')) 
pickle.dump(Csc0_test,open('Csc0_test','w'))
