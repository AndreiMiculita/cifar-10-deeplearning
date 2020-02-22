from matplotlib import pyplot as plt
import numpy as np 

from src.models import *
from src.utils import *

architectures = ['VGG']
activations = ['ELU','lRELU','ReLU','PReLU','CELU','GELU','Softplus','Softmax2d']
optimizers = ['Adam', 'RMSprop', 'SGD+mom']


loss_, acc_ = [], []
#test different activation functions
for act in activations:
	print('\nTesting {}: activations\n'.format(act))
	l_, a_ = vgg(activation=act)
	loss_.append([])
	acc_.append([])
	loss_[activations.index(act)] = l_
	acc_[activations.index(act)] = a_
	torch.cuda.empty_cache()
	
plt.title('model:VGG, optimizer: SGD+mom, activations:') 
plt.xlabel('epochs')
plt.ylabel('cross-entropy loss')
plt.plot([x*10 for x in loss_[0]], '-b', label='ELU')
plt.plot([x*10 for x in loss_[1]], '-g', label='leaky ReLU')
plt.plot([x*10 for x in loss_[2]], '-r', label='ReLU')
plt.plot([x*10 for x in loss_[3]], '-c', label='PReLU')
plt.plot([x*10 for x in loss_[4]], '-y', label='CELU')
plt.plot([x*10 for x in loss_[5]], '-m', label='Softplus')
plt.plot([x*10 for x in loss_[5]], '-chartreuse', label='Softmax2d')
plt.legend()
plt.show() 
