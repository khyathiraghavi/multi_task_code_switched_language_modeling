#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
# arguments about the learning parameters and when and where to save the trained models
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=5, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=1, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=10, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=1, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=10, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
# Shuffling data after every epoch is preferable
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
# hyperparameters of the model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
#gpu usage: whether to use or not and the number of gpus
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


# load SST dataset
def sst(text_field, label_field,  **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(args.batch_size, 
                                                     len(dev_data), 
                                                     len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter 


# load MR dataset
def mr(text_field, label_field, **kargs):
    # MR data objects that contain torch text objects with actual text and labels
    train_data, dev_data = mydatasets.MR.splits(text_field, label_field)
    # coverts the words or tokens that are present in the text to integers
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(args.batch_size, len(dev_data)),
                                **kargs)
    # returns iterator objects in torch text
    return train_iter, dev_iter


"""
data is an API in torchtext
Torchtext library helps in pre-processing the data
Field is an inbuilt class that specifies how we want the data to be pre-processed
Here we want the text_field which is the actual text to be lower cased.
The label_field is not sequential data, since the original is predicting the sentiment of the data (lines?)
"""
# load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
# train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)


# update args and print
args.embed_num = len(text_field.vocab) #vocabulary size for the entire text
args.class_num = len(label_field.vocab) - 1 # the sentiment labels for each of the line in the files based on which file the data is read from 
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda # making the variable corresponding to no_cuda as None 
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
# kernel sizes that are used for series of convolution layers. In this case it is [3, 4, 5]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# args are a collection of all the arguments that are given in the command line argument parser
# Namespace(batch_size=64, class_num=2, cuda=True, device=-1, dropout=0.5, early_stop=1000, embed_dim=128, embed_num=136, epochs=256, kernel_num=100, kernel_sizes=[3, 4, 5], log_interval=1, lr=0.001, max_norm=3.0, predict=None, save_best=True, save_dir='snapshot/2018-03-19_16-45-13', save_interval=500, shuffle=False, snapshot=None, static=False, test=False, test_interval=100)

# the cnn model
cnn = model.CNN_Text(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

#print args.predict # None
#print args.test # False

#args.predict = "something"
# train or predict
if args.predict is not None:
    print "inside this"
    label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_iter, cnn, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_iter, dev_iter, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

