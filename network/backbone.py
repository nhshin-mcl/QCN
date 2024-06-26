import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
class BackboneV4(nn.Module):
    def __init__(self, cfg):
        super(BackboneV4, self).__init__()
        if cfg.backbone == 'resnet50':
            self.encoder = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.encoder.avgpool = nn.Identity()
            self.encoder.fc = nn.Identity()

            self.reduce_dim = nn.Sequential(
                nn.Conv2d(2048, cfg.reduced_dim, kernel_size=1, padding=0),
                nn.AdaptiveAvgPool2d(1),
            )

            self.reduce_dim[0].weight.data.normal_(0, 0.01)
            self.reduce_dim[0].bias.data.zero_()

        else:
            raise ValueError(f'[!] undefined backbone architecture has been given: {cfg.backbone}.')

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.backbone_name = cfg.backbone

    def _forward(self, im):

        f1 = self.encoder.maxpool(self.encoder.relu(self.encoder.bn1(self.encoder.conv1(im))))
        f2 = self.encoder.layer1(f1)
        f3 = self.encoder.layer2(f2)
        f4 = self.encoder.layer3(f3)
        f5 = self.encoder.layer4(f4)

        base_embs = self.reduce_dim(f5)

        return base_embs
