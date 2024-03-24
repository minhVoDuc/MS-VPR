import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

class InterpolateModule(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super(InterpolateModule, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode)

class FeatureReuse(nn.Module):
    def __init__(self, channel_size = 1024, feat_size_3rd = 1600, feat_size_4th = 400):
        # pass
        super().__init__()
        self.layer_norm = nn.LayerNorm(feat_size_3rd)
        self.upsample = InterpolateModule(size=(channel_size,feat_size_4th))

    def forward(self, x_3rd, x_4th):
        x_3rd = self.layer_norm(x_3rd)
        x_3rd = self.upsample(x_3rd.unsqueeze(1)).squeeze(1)
        x = torch.concat((x_4th, x_3rd), dim=1)
        return x


class FeatureMixerLayer(nn.Module): 
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)

class MuscVPR(nn.Module):
    def __init__(self,
                 in_h=40,
                 in_w=40,
                 in_h_1=20,
                 in_w_1=20,
                 in_channels=2048,
                 out_channels=1024,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()

        self.in_h = in_h # height of 3rd conv feature maps
        self.in_w = in_w # width of 3rd conv input feature maps

        self.in_h_1 = in_h_1 # height of 4th conv feature maps
        self.in_w_1 = in_w_1 # width of 4th conv feature maps
        self.in_channels = in_channels # depth of combined feature maps
        
        self.out_channels = out_channels # depth wise projection dimension
        self.out_rows = out_rows # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        hw_3rd = in_h*in_w
        hw_4th = in_h_1*in_w_1
                     
        # first FeatureMixer block
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw_3rd, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])

        # second FeatureMixer block
        self.mix_1 = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw_4th, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])

        self.channel_proj = nn.Linear(512, 512)
        self.row_proj = nn.Linear(1600, 4)
        self.channel_proj_1 = nn.Linear(in_channels, out_channels)
        self.row_proj_1 = nn.Linear(in_h_1 * in_w_1, out_rows)

        # FeatureReuse block
        self.reuse = FeatureReuse()


    def forward(self, x_4th, x_3rd):
        # Flatten feat maps from both layers to make them of size (N, C, HxW) (N = batch size)
        x_3rd = x_3rd.flatten(2)
        x_4th = x_4th.flatten(2)

        # Apply Feature-Mixer on feat maps from 3rd layer
        x_3rd = self.mix(x_3rd)

        # FeatureReuse
        x = self.reuse(x_3rd, x_4th)

        # Final MixVPR (Feature-Mixer + Depth-wise & Row-wise projection)
        x = self.mix_1(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj_1(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj_1(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


# -------------------------------------------------------------------------------

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    x = torch.randn(40, 1024, 20, 20)
    x_pre = torch.randn(40, 512, 40, 40)
    agg = MuscVPR(
                 in_h=40,
                 in_w=40,
                 in_h_1=20,
                 in_w_1=20,
                 in_channels=2048,
                 out_channels=1024,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_rows=4,
                 ) 

    print_nb_params(agg)
    output = agg(x, x_pre)
    # print(x.type)
    # output= agg.forward(x, x_pre)
    print(output.shape)


if __name__ == '__main__':
    main()
