import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import datasets, transforms

from src.utils import *

# define model hyperparameters
batch_size = 256
epochs = 20
learning_rate = .01
img_crop_size = 64
print_step = 120
weight_decay = 0.01

global activation
lookup_act = {
    'ReLU'       : nn.ReLU(inplace=True),
    'ELU'        : nn.ELU(),
    'lRELU'      : nn.LeakyReLU(),
    'PReLU'      : nn.PReLU(),
    'CELU'       : nn.CELU(),
    'Softplus'   : nn.Softplus(),
    'Softmax2d'  : nn.Softmax2d()
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# define AlexNet Architecture
class AlexNet(nn.Module):
    def __init__(self, num_classes, activation):
        super(AlexNet, self).__init__()
        self.activation = self.lookup(activation)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            self.activation,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            self.activation,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.activation,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            self.activation,
            nn.Dropout(),
            nn.Linear(4096, 4096),
            self.activation,
            nn.Linear(4096, num_classes),
        )

    # takes in a module and applies the specified weight initialization
    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def lookup(self, f):  # activation function wrapping
        if f == 'ReLU':
            return nn.ReLU(inplace=True)
        elif f == 'ELU':
            return nn.ELU
        elif f == 'lReLU':
            return nn.LeakyReLU(negative_slope=0.01)
        elif f == 'GeLU':
            return nn.GELU


def alexnet(activations, optim):
    # if not sys.argv[1:]:
    #   print('\nGive arguements. Usage:\n')
    #   return 0
    # else:

    activation = 'ReLU'
    optim = 'Adam'
    model = AlexNet(num_classes=10, activation=activation)
    # model.weights_init_uniform()

    print("Using model: \n", model)

    # use gpu tensors if available
    device = 'cpu'
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
        print('Using cuda.\n')
    else:
        print('Using cpu.\n')

    # data preproc stage - img format: {batch_size X 3 X img_size X img_size}
    # convert to Tensor objects and normalize to floats in [0,1]
    transform = transforms.Compose([
        transforms.Resize(img_crop_size, interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Try data augmentation
    # transform = transforms.Compose([
    #     ImgAugTransform(),
    #     lambda x: PIL.Image.fromarray(x),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # define dataset - here CIFAR10
    train_data = datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data/', train=False, transform=transform)

    # shuffle and batch data inside DataLoader objects
    trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # define loss function and optimization algorithm
    loss_fn = nn.CrossEntropyLoss()  # here cross-entropy for multiclass classficiation
    if optim == 'SGD+mom':
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.93, weight_decay=weight_decay)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=weight_decay)
    elif optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=.001, alpha=.95, weight_decay=weight_decay)
    else:
        raise ValueError

    # train the model on the train set, while validating on the validation set
    train_losses, eval_losses = train(model, trainloader, testloader, optimizer, loss_fn, epochs, learning_rate, device)
    # make predictions for a test set
    accuracy = test(model, trainloader, loss_fn, device)
    print("Model accuracy on train set: %.1f %%" % accuracy)
    accuracy = test(model, testloader, loss_fn, device)
    print("Model accuracy on test set: %.1f %%" % accuracy)
    plt.title('model: AlexNet, activation: ReLu, optimization:'.format(optim))
    plt.xlabel('epochs')
    plt.ylabel('cross-entropy loss')
    plt.plot(train_losses, 'r--', label='train')
    plt.plot(eval_losses, 'b--', label='test')
    plt.legend()
    plt.show()
    return train_losses, eval_losses


class VGG(nn.Module):

    def __init__(self, features, activation, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.activation = lookup_act[activation]
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            self.activation,
            nn.Dropout(),
            nn.Linear(4096, 4096),
            self.activation,
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, activation, **kwargs):
    return VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), activation, **kwargs)


def vgg16(activation, pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, activation, **kwargs)


def vgg16_bn(activation, pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, activation, **kwargs)


# define VGG architecture
def vgg(activation):
    optim = 'SGD+mom'
    model = vgg16_bn(activation, num_classes=10)
    print("Using model: \n", model)

    # get the working directory where the data is saved
    import os
    datadir = os.getcwd()

    # use gpu tensors if available
    device = 'cpu'
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
        print('Using cuda.\n')
    else:
        print('Using cpu.\n')

    # data preproc stage - img format: {batch_size X 3 X img_size X img_size}
    # convert to Tensor objects and normalize to floats in [0,1]
    transform = transforms.Compose([
        transforms.Resize(img_crop_size, interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # define dataset - here CIFAR10
    train_data = datasets.CIFAR10(root='./data/', train=True, download=False, transform=transform)
    test_data = datasets.CIFAR10(root='./data/', train=False, transform=transform)

    # shuffle and batch data inside DataLoader objects
    trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # define loss function and optimization algorithm
    loss_fn = nn.CrossEntropyLoss()  # here cross-entropy for multiclass classficiation
    if optim == 'SGD+mom':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.95, weight_decay=weight_decay)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=weight_decay)
    elif optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=.0001, alpha=.95, weight_decay=weight_decay)
    else:
        raise ValueError

        # train the model on the train set, while validating on the validation set
    train_losses, eval_losses = train(model, trainloader, testloader, optimizer, loss_fn, epochs, learning_rate, device)
    # make predictions for a test set
    accuracy_train = test(model, trainloader, loss_fn, device)
    print("Model accuracy on train set: %.1f %%" % accuracy_train)
    accuracy_test = test(model, testloader, loss_fn, device)
    print("Model accuracy on test set: %.1f %%" % accuracy_test)
    # plt.xlabel('epochs')
    # plt.title('model: VGG, activations:{}'.format(activation) + 'optimization: SGD+mom').
    # plt.ylabel('cross-entropy loss')
    # plt.plot(train_losses, 'r--', label='train')
    # plt.plot(eval_losses, 'b--', label='test')
    # plt.legend()
    # plt.show()

    acc_train = accuracy_train.item()
    acc_test = accuracy_test.item()
    return train_losses, eval_losses, acc_train, acc_test
