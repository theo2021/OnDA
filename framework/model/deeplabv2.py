import torch
import torch.nn as nn

affine_par = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        norm_module=nn.BatchNorm2d,
        norm_grad=False,
    ):
        super(Bottleneck, self).__init__()
        # change
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False
        )
        self.bn1 = norm_module(planes, affine=affine_par)
        if not norm_grad:
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
        self.bn2 = norm_module(planes, affine=affine_par)
        if not norm_grad:
            for i in self.bn2.parameters():
                i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_module(planes * 4, affine=affine_par)
        if not norm_grad:
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


# Inherited by ProDA to perform prototyping
class SEBlock(nn.Module):
    def __init__(self, inplanes, r=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.se = nn.Sequential(
            nn.Linear(inplanes, inplanes // r),
            nn.ReLU(inplace=True),
            nn.Linear(inplanes // r, inplanes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        xx = self.global_pool(x)
        xx = xx.view(xx.size(0), xx.size(1))
        se_weight = self.se(xx).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)


class Classifier_Module2(nn.Module):
    def __init__(
        self,
        inplanes,
        dilation_series,
        padding_series,
        num_classes,
        droprate=0.1,
        use_se=True,
    ):
        super(Classifier_Module2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv2d_list.append(
            nn.Sequential(
                *[
                    nn.Conv2d(
                        inplanes,
                        256,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        dilation=1,
                        bias=True,
                    ),
                    nn.GroupNorm(num_groups=32, num_channels=256, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
        )

        for dilation, padding in zip(dilation_series, padding_series):
            # self.conv2d_list.append(
            #    nn.BatchNorm2d(inplanes))
            self.conv2d_list.append(
                nn.Sequential(
                    *[
                        # nn.ReflectionPad2d(padding),
                        nn.Conv2d(
                            inplanes,
                            256,
                            kernel_size=3,
                            stride=1,
                            padding=padding,
                            dilation=dilation,
                            bias=True,
                        ),
                        nn.GroupNorm(num_groups=32, num_channels=256, affine=True),
                        nn.ReLU(inplace=True),
                    ]
                )
            )

        if use_se:
            self.bottleneck = nn.Sequential(
                *[
                    SEBlock(256 * (len(dilation_series) + 1)),
                    nn.Conv2d(
                        256 * (len(dilation_series) + 1),
                        256,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        bias=True,
                    ),
                    nn.GroupNorm(num_groups=32, num_channels=256, affine=True),
                ]
            )
        else:
            self.bottleneck = nn.Sequential(
                *[
                    nn.Conv2d(
                        256 * (len(dilation_series) + 1),
                        256,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        bias=True,
                    ),
                    nn.GroupNorm(num_groups=32, num_channels=256, affine=True),
                ]
            )

        self.head = nn.Sequential(
            *[
                nn.Dropout2d(droprate),
                nn.Conv2d(
                    256, num_classes, kernel_size=1, padding=0, dilation=1, bias=False
                ),
            ]
        )

        ##########init#######
        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                m.bias.data.zero_()
            elif (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.InstanceNorm2d)
                or isinstance(m, nn.GroupNorm)
            ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.bottleneck:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
                m.bias.data.zero_()
            elif (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.InstanceNorm2d)
                or isinstance(m, nn.GroupNorm)
                or isinstance(m, nn.LayerNorm)
            ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)

    def forward(self, x, get_feat=False):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat((out, self.conv2d_list[i + 1](x)), 1)
        out = self.bottleneck(out)
        if get_feat:
            out_dict = {}
            out = self.head[0](out)
            out_dict["feat"] = out
            out = self.head[1](out)
            out_dict["out"] = out
            return out_dict
        else:
            out = self.head(out)
            return out


class ResNetMulti(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes,
        multi_level,
        classifier_module="normal",
        norm_module=nn.BatchNorm2d,
        norm_grad=False,
    ):
        self.multi_level = multi_level
        self.feat = False
        if classifier_module == "ProDA":
            clf_module = Classifier_Module2
            self.feat = True
        elif classifier_module == "normal":
            clf_module = ClassifierModule
        else:
            print("Using default ADVENT CLF")
            clf_module = ClassifierModule
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_module(64, affine=affine_par)
        if not norm_grad:
            for i in self.bn1.parameters():
                i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True
        )  # change
        self.layer1 = self._make_layer(
            block, 64, layers[0], norm_module=norm_module, norm_grad=norm_grad
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            norm_module=norm_module,
            norm_grad=norm_grad,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=1,
            dilation=2,
            norm_module=norm_module,
            norm_grad=norm_grad,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=1,
            dilation=4,
            norm_module=norm_module,
            norm_grad=norm_grad,
        )
        if self.multi_level:
            self.layer5 = clf_module(
                1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes
            )
        self.layer6 = clf_module(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilation=1,
        norm_module=nn.BatchNorm2d,
        norm_grad=False,
    ):
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
                norm_module(planes * block.expansion, affine=affine_par),
            )
        if not norm_grad:
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.multi_level:
            if self.feat:
                x1 = self.layer5(x, True)  # produce segmap 1
            else:
                x1 = self.layer5(x)
        else:
            x1 = None
        x2 = self.layer4(x)
        if self.feat:
            x2 = self.layer6(x2, True)  # produce segmap 2
        else:
            x2 = self.layer6(x2)
        return x1, x2

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


def get_deeplab_v2(
    num_classes=19,
    multi_level=True,
    layers=[3, 4, 23, 3],
    classifier="normal",
    norm_module=nn.BatchNorm2d,
    norm_grad=False,
):
    model = ResNetMulti(
        Bottleneck,
        layers,
        num_classes,
        multi_level,
        classifier_module=classifier,
        norm_module=norm_module,
        norm_grad=norm_grad,
    )
    return model
