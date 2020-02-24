import torch.nn as nn


# define AlexNet Architecture
class AlexNet(nn.Module):
    def __init__(self, num_classes, activation):
        super(AlexNet, self).__init__()
        self.activation = activation
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
    @staticmethod
    def weights_init_uniform(m):
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
