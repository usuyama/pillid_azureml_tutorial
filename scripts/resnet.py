import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BasicResNet(nn.Module):
    def __init__(self, num_classes=100, network='resnet18', pretrained=True):
        super(BasicResNet, self).__init__()

        if network == 'resnet152':
            resnet_model = models.resnet152(pretrained=pretrained)
        elif network == 'resnet101':
            resnet_model = models.resnet101(pretrained=pretrained)
        elif network == 'resnet50':
            resnet_model = models.resnet50(pretrained=pretrained)
        elif network == 'resnet34':
            resnet_model = models.resnet34(pretrained=pretrained)
        elif network == 'resnet18':
            resnet_model = models.resnet18(pretrained=pretrained)
        else:
            raise Exception("{} model type not supported".format(network))

        self.resnet_model = resnet_model
        self.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
        self.resnet_model.fc = self.fc

    def forward(self, x):
        output = self.resnet_model.forward(x)

        return output


if __name__ == '__main__':
    model = BasicResNet()

    print(model.state_dict().keys())
