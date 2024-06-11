from torch import nn
from mmseg.ops import resize
import torch
from mmseg.models.decode_heads.segformer_head import SegFormerHead
from mmseg.models.backbones.mix_transformer import mit_b1


norm_cfg = dict(type='BN', requires_grad=True)

class ModifiedSegformerHead(SegFormerHead):
    
    def __init__(self, **kwargs):
        default_values = dict(in_channels=[64, 128, 320, 512],
                         in_index=[0, 1, 2, 3],
                         channels=128,
                         dropout_ratio=0.1,
                         num_classes=19,
                         norm_cfg=norm_cfg,
                         align_corners=False,
                         decoder_params=dict(embed_dim=256, conv_kernel_size=1))
        default_values.update(kwargs)
        super().__init__(**default_values)
        
    def forward(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     print(f.shape)

        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}, {self.linear_c[str(i)]}')
            _c[i] = self.linear_c[str(i)](x[i]).permute(0, 2, 1).contiguous()
            _c[i] = _c[i].reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if i != 0:
                _c[i] = resize(
                    _c[i],
                    size=x[0].size()[2:],
                    mode='bilinear',
                    align_corners=False)

        feat = self.linear_fuse(torch.cat(list(_c.values()), dim=1))

        if self.dropout is not None:
            x = self.dropout(feat)
        else:
            x = feat
        out = self.linear_pred(x)

        return None, {'feat': feat,
                'out': out}


class SegFormerMitB1Model(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.backbone = mit_b1()
        self.decode_head = ModifiedSegformerHead(**args)
        
    def forward(self, X):
        backbone_out = self.backbone(X)
        return self.decode_head(backbone_out)
    
    def optim_parameters(self, lr):
        return [
            {"params": self.backbone.parameters(), "lr": lr},
            {"params": self.decode_head.parameters(), "lr": 10 * lr},
        ]
    