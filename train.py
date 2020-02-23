from src.models import *
from src.resnet import resnet_and_train
from src.utils import *

architectures = ['VGG']
activations = ['ELU', 'lRELU', 'ReLU', 'PReLU', 'CELU', 'Softplus']
optimizers = ['Adam', 'RMSprop', 'SGD+mom']

loss_, acc_ = [], []
# test different activation functions
for act in activations:
    print('\nTesting {}: activations\n'.format(act))
    l_, a_, tr, ts, time_taken = resnet_and_train(act)
    loss_.append([])
    acc_.append([])
    loss_[activations.index(act)] = l_
    acc_[activations.index(act)] = a_

    with open('{}acc.csv'.format(act), 'w') as f:
        save_acc = [str(tr),',', str(ts),',',time_taken,"\n"]
        f.writelines(save_acc)

    torch.cuda.empty_cache()


plt.title('model:resnet, optimizer: SGD+mom, activations:')
plt.xlabel('epochs')
plt.ylabel('cross-entropy loss')
plt.plot([x * 10 for x in loss_[0]], '-b', label='ELU')
plt.plot([x * 10 for x in loss_[1]], '-g', label='leaky ReLU')
plt.plot([x * 10 for x in loss_[2]], '-r', label='ReLU')
plt.plot([x * 10 for x in loss_[3]], '-c', label='PReLU')
plt.plot([x * 10 for x in loss_[4]], '-y', label='CELU')
plt.plot([x * 10 for x in loss_[5]], '-m', label='Softplus')
plt.legend()
plt.show()
