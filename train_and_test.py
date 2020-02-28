import time

from src.alexnet import *
from src.resnet import *
from src.vgg import *
from src.utils import *
from torchvision import datasets, transforms


def train_and_test(model, optimizer):
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

    start = time.time()
    # train the model on the train set, while validating on the validation set
    train_losses, eval_losses = train(model, trainloader, testloader, optimizer, loss_fn, device)
    time_taken = time.time() - start

    # make predictions for a test set
    accuracy_train = test(model, trainloader, loss_fn, device)
    print("Model accuracy on train set: %.1f %%" % accuracy_train)
    accuracy_test = test(model, testloader, loss_fn, device)
    print("Model accuracy on test set: %.1f %%" % accuracy_test)

    acc_train = accuracy_train.item()
    acc_test = accuracy_test.item()
    return train_losses, eval_losses, acc_train, acc_test, str(int(time_taken))


if __name__ == "__main__":
    models = [vgg16_bn, AlexNet, resnet18]
    activations = [nn.ELU(), nn.LeakyReLU(), nn.ReLU(inplace=True), nn.PReLU(), nn.CELU(), nn.Softplus()]
    optimizers = [torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]

    loss_, acc_ = [], []
    # test different activation functions
    with open('accuracies_with_weight_decay.csv', 'a') as accuracies, open('loss_over_time_with_weight_decay.csv', 'a') as loss_over_time:
        print("architecture,optimizer,activation,training accuracy,testing accuracy,time elapsed\n", file=accuracies)
        for optimizer in optimizers:
            for model in models:
                for activation in activations:
                    print("testing ", model, " ", optimizer, " ", activation)
                    l_, a_, tr, ts, time_taken = train_and_test(model(activation=activation, num_classes=10), optimizer)
                    print(l_)
                    print(a_)

                    print(f'{model},{optimizer},{str(activation).replace(",",";")},{str(tr)},{str(ts)},{time_taken}', file=accuracies)
                    print(f'{model},{optimizer},{str(activation).replace(",",";")},{",".join(map(str,l_))}', file=loss_over_time)
                    print(f'{model},{optimizer},{str(activation).replace(",",";")},{",".join(map(str,a_))}', file=loss_over_time)

                    print(f'{model},{optimizer},{str(activation).replace(",",";")},{str(tr)},{str(ts)},{time_taken}')
                    print(f'{model},{optimizer},{str(activation).replace(",",";")},{",".join(map(str,l_))}')
                    print(f'{model},{optimizer},{str(activation).replace(",",";")},{",".join(map(str,a_))}')
                    torch.cuda.empty_cache()
