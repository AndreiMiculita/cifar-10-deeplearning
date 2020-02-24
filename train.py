from src.models import *
from src.resnet import resnet_and_train
from src.utils import *

architectures = [alexnet]
activations = [nn.ELU(), nn.LeakyReLU(), nn.ReLU(inplace=True), nn.PReLU(), nn.CELU(), nn.Softplus()]
optimizers = [torch.optim.Adam, torch.optim.RMSprop, torch.optim.SGD]

loss_, acc_ = [], []
# test different activation functions
with open('accuracies.csv', 'a') as f:
    print("architecture,optimizer,activation,training accuracy,testing accuracy,time elapsed\n", file=f)
    for arch in architectures:
        for opt in optimizers:
            for act in activations:
                print("testing ", arch, " ", opt, " ", act)
                l_, a_, tr, ts, time_taken = arch(act, opt)
                loss_.append([])
                acc_.append([])
                loss_[activations.index(act)] = l_
                acc_[activations.index(act)] = a_

                print(arch, opt, act, str(tr), ',', str(ts), ',', time_taken, "\n", file=f)
                torch.cuda.empty_cache()

            plt.title('model: Alexnet, optimizer: {}'.format(opt) +', activations:')
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
