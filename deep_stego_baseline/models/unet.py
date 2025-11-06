import torch
import torch.nn as nn

def CBR(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class UNetEncoder(nn.Module):
    def __init__(self, payload_channels=1):
        super().__init__()
        c0 = 3 + payload_channels
        self.enc1 = nn.Sequential(CBR(c0,64), CBR(64,64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(64,128), CBR(128,128))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(CBR(128,256), CBR(256,256))
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(CBR(256+128,128), CBR(128,64))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(CBR(64+64,64), CBR(64,32))
        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, cover, payload_map):
        x = torch.cat([cover, payload_map], dim=1)
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        u2 = self.up2(e3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        delta = self.out(d1)
        stego = torch.clamp(cover + torch.tanh(delta)*0.02, 0.0, 1.0)
        return stego
