from torch.autograd import Variable
from constants import *

class Batching(object):
    def __init__(self):
        self.batching = True

    def batchify(self, data, batch_size, args):
        '''
        identify the number of batches possible with the batch size and then remove the extra data points.
        Distribute the data points equally across the batches
        '''
        num_batches = data.size(0) // batch_size
        data = data.narrow(0, 0, num_batches * batch_size)
        data = data.view(batch_size, -1).t().contiguous()
        if args.cuda:
            data = data.cuda()
        return data

    def get_batch(self, source, i, seq_len=None, evaluation=False):
        seq_len = min(seq_len if seq_len else BPTT, len(source) - 1 - i)
        data = Variable(source[i:i+seq_len], volatile=evaluation)
        target = Variable(source[i+1:i+1+seq_len].view(-1))
        return data, target