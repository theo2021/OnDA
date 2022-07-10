import torch.nn as nn
import torch

affine_par = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        # change
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        # change
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=padding,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(
                    inplanes,
                    num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                )
            )

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes, percentage=0.7):
        self.multi_level = multi_level
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True
        )  # change
        block_size = [64, 128, 256, 512]
        self.only_target = False
        self.layer1_main = self._make_layer(
            block, int(block_size[0] * percentage), layers[0]
        )
        self.layer1_aux = self._make_layer(
            block, block_size[0] - int(block_size[0] * percentage), layers[0]
        )

        self.layer2_main = self._make_layer(
            block, int(block_size[1] * percentage), layers[1], stride=2
        )
        self.layer2_aux = self._make_layer(
            block, block_size[1] - int(block_size[1] * percentage), layers[1], stride=2
        )

        self.layer3_main = self._make_layer(
            block, int(block_size[2] * percentage), layers[2], stride=1, dilation=2
        )
        self.layer3_aux = self._make_layer(
            block,
            block_size[2] - int(block_size[2] * percentage),
            layers[2],
            stride=1,
            dilation=2,
        )

        self.layer4_main = self._make_layer(
            block, int(block_size[3] * percentage), layers[3], stride=1, dilation=4
        )
        self.layer4_aux = self._make_layer(
            block,
            block_size[3] - int(block_size[3] * percentage),
            layers[3],
            stride=1,
            dilation=4,
        )

        main_out = int(block_size[3] * percentage) * 4
        aux_out = (block_size[3] - int(block_size[3] * percentage)) * 4
        self.source_classifier = ClassifierModule(
            main_out, [6, 12, 18, 24], [6, 12, 18, 24], num_classes
        )
        self.target_classifier = ClassifierModule(
            aux_out, [6, 12, 18, 24], [6, 12, 18, 24], num_classes
        )
        self.main_modules = [
            self.conv1,
            self.layer1_main,
            self.layer2_main,
            self.layer3_main,
            self.layer4_main,
        ]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (
            stride != 1
            or self.inplanes != planes * block.expansion
            or dilation == 2
            or dilation == 4
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par),
            )
        for i in downsample._modules["1"].parameters():
            i.requires_grad = False
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, dilation=dilation, downsample=downsample
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x_main = self.layer1_aux(x)
        x_main = self.layer2_main(x_main)
        x_main = self.layer3_main(x_main)
        output = {}
        output["feature_common"] = x_main
        output["target_segmentation"] = self.target_classifier(x_main)
        if not self.only_target:
            x_aux = self.layer1_aux(x)
            x_aux = self.layer2_aux(x_aux)
            x_aux = self.layer3_aux(x_aux)
            output["feature_source"] = x_aux
            x_full = torch.cat((x_main, x_aux))
            output["source_segmentation"] = self.source_classifier(x_full)
        else:
            output["feature_source"] = None
            output["source_segmentation"] = None
        return output

    def feeze_layers(self, option=True):
        for m in self.main_modules:
            for par in m.parameters():
                par.requires_grad = not option

    def get_1x_lr_params_no_scale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        if self.multi_level:
            b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [
            {"params": self.get_1x_lr_params_no_scale(), "lr": lr},
            {"params": self.get_10x_lr_params(), "lr": 10 * lr},
        ]


def get_deeplab_v2(num_classes=19, multi_level=True):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes, multi_level)
    return model
