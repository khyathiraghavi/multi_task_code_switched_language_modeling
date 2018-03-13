import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter

import numpy as np
from sklearn.model_selection import train_test_split
import urllib
import shutil
import os
from datetime import datetime
from tqdm import tqdm_notebook as tqdm
#import matplotlib.pyplot as plt

# input file and hyperparameter settings
DATA_FILE = 'data.txt'
alphabet = 'abcdefghijklmnopqrstuvwxyz01234567890 \n.,!?'
ALPHABET_SIZE = len(alphabet)
BATCH_SIZE = 500
SEQ_LENGTH = 100
HIDDEN_SIZE = 64
RHN_LAYERS = 3
MLP_LAYERS = 2


# reading a remote data file
if not os.path.exists(DATA_FILE):
    with urllib.request.urlopen('https://www.gutenberg.org/files/521/521-0.txt') as remote:
        with open(DATA_FILE, 'wb') as file:
            shutil.copyfileobj(remote, file)
alphabet = np.array([ord(i) for i in alphabet])
alphabet_inv = np.array([-1] * 256)
for i, c in enumerate(alphabet):
    alphabet_inv[c] = i
    alphabet_inv[ord(chr(c).upper())] = i
data_src = np.fromfile(DATA_FILE, np.ubyte)
data = alphabet_inv[data_src]
data = data[data != -1]



class Bilinear (nn.Module):
    
    def __init__(self, in_features, out_features):
        super(Bilinear, self).__init__()
        self.linear_x = nn.Linear(in_features, out_features, bias=False)
        self.linear_h = nn.Linear(out_features, out_features, bias=True)

    def forward(self, x, h):
        return self.linear_x(x) + self.linear_h(h)



class RHN (nn.Module):
    
    def __init__(self, in_features, out_features, num_layers):
        super(RHN, self).__init__()
        #super().__init__()
        self.transform = []
        self.gate = []
        for i in range(num_layers):
            transform = Bilinear(in_features, out_features)
            gate = Bilinear(in_features, out_features)
            setattr(self, 'transform%d' % i, transform)
            setattr(self, 'gate%d' % i, gate)
            self.transform.append(transform)
            self.gate.append(gate)
        
    def forward(self, seq, h):
        for x in seq:
            for transform, gate in zip(self.transform, self.gate):
                gate_value = F.sigmoid(gate(x, h))
                h = F.tanh(transform(x, h)) * gate_value + h * (1 - gate_value)
        return h



class MLP (nn.Module):

    def __init__(self, in_features, out_features, num_layers):
        super(MLP, self).__init__()
        self.transform = []
        self.gate = []
        for i in range(num_layers - 1):
            transform = nn.Linear(in_features, in_features)
            gate = nn.Linear(in_features, in_features)
            setattr(self, 'transform%d' % i, transform)
            setattr(self, 'gate%d' % i, gate)
            self.transform.append(transform)
            self.gate.append(gate)
        self.output_layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        for transform, gate in zip(self.transform, self.gate):
            gate_value = F.sigmoid(gate(x))
            x = F.tanh(transform(x)) * gate_value + x * (1 - gate_value)
        x = self.output_layer(x)
        return x




def make_batch():
    x = torch.zeros((SEQ_LENGTH, BATCH_SIZE, ALPHABET_SIZE))
    indices = np.random.randint(0, len(data) - SEQ_LENGTH - 1, BATCH_SIZE)
    for i in range(SEQ_LENGTH):
        for j, c in enumerate(indices):
            x[i, j, data[c]] = 1
        indices += 1
    y = torch.LongTensor(data[indices])
    return Variable(x), Variable(y)


rhn = RHN(ALPHABET_SIZE, HIDDEN_SIZE, RHN_LAYERS)
mlp = MLP(HIDDEN_SIZE, ALPHABET_SIZE, MLP_LAYERS)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([ { 'params': rhn.parameters() }, { 'params': mlp.parameters() }])
h = Variable(torch.zeros(BATCH_SIZE, HIDDEN_SIZE))


x, y = make_batch()

#t = tqdm(range(2000), desc='progress')
t = range(20)
'''
fig = plt.figure()
plt.xlabel('epoch')
plt.ylabel('cross entropy loss')
plt.title('Training loss')
plt.grid()
'''
prev_loss = np.inf
start_dt = datetime.now()
prev_dt = start_dt
for i in t:
    print i
    x, y = make_batch()

    optimizer.zero_grad()
    y_preds = mlp(rhn(x, h))
    loss = criterion(y_preds, y)
    loss.backward()
    #plt.plot([i, i + 1], [prev_loss, loss.data[0]], 'b-')
    #t.set_postfix(loss=loss.data[0])
    dt = datetime.now()
    #if (dt - prev_dt).total_seconds() >= 10:
    #    fig.canvas.draw()
    #    prev_dt = dt
    prev_loss = loss.data[0]
    optimizer.step()
    for p in optimizer.param_groups:
        p['lr'] /= 1.001
print('elapsed time: %d seconds, final loss: %s' % (datetime.now() - start_dt, loss.data[0]))


