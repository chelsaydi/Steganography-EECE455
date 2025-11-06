import torch
import torch.nn as nn

def conv_block(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class Encoder(nn.Module):
    def __init__(self, payload_channels=1):
        super().__init__()
        c0 = 3 + payload_channels
        self.net = nn.Sequential(
            conv_block(c0, 64), conv_block(64, 64),
            conv_block(64, 128, s=2), conv_block(128, 128),
            conv_block(128, 256, s=2), conv_block(256, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            conv_block(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            conv_block(128, 64),
            nn.Conv2d(64, 3, 1)
        )

    def forward(self, cover, payload_map):
        x = torch.cat([cover, payload_map], dim=1)
        delta = self.net(x)
        stego = torch.clamp(cover + torch.tanh(delta) * 0.02, 0.0, 1.0)
        return stego

class Decoder(nn.Module):
    def __init__(self, num_bits: int):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 64), conv_block(64, 64),
            conv_block(64, 128, s=2), conv_block(128, 128),
            conv_block(128, 256, s=2), conv_block(256, 256),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Linear(512, num_bits))

    def forward(self, stego_or_noisy):
        f = self.features(stego_or_noisy)
        logits = self.head(f)
        return logits
